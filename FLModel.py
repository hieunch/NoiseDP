# Federated Learning Model in PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils import gaussian_noise, binomial_noise
from rdp_analysis import calibrating_sampled_gaussian
from qtorch.quant import float_quantize, fixed_point_quantize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from MLModel import *

import numpy as np
import copy

from torch.func import functional_call, vmap, grad
import torch.nn.functional as F


class FLClient(nn.Module):
    """ Client of Federated Learning framework.
        1. Receive global model from server
        2. Perform local training (compute gradients)
        3. Return local model (gradients) to server
    """
    def __init__(self, model, output_size, data, lr, E, batch_size, clip, n_bits, \
    			 noise_scheme, sigma, trials, success_prob, device=None):
        """
        :param model: ML model's training process should be implemented
        :param data: (tuple) dataset, all data in client side is used as training data
        :param lr: learning rate
        :param E: epoch of local update
        """
        super(FLClient, self).__init__()
        self.device = device
        self.BATCH_SIZE = batch_size
        self.torch_dataset = TensorDataset(torch.tensor(data[0]),
                                           torch.tensor(data[1]))
        self.data_size = len(self.torch_dataset)
        self.data_loader = DataLoader(
            dataset=self.torch_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        self.noise_scheme = noise_scheme
        self.sigma = sigma    # DP noise level (Gaussian)
        self.trials = trials
        self.success_prob = success_prob
        self.n_bits = n_bits
        self.lr = lr
        self.E = E
        self.clip = clip
        if model == 'scatter':
            self.model = ScatterLinear(81, (7, 7), input_norm="GroupNorm", num_groups=27).to(self.device)
        else:
            self.model = model(data[0].shape[1], output_size).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total params: {total_params}')

    def recv(self, model_param):
        """receive global model from aggregator (server)"""
        self.model.load_state_dict(copy.deepcopy(model_param))

    def compute_loss(self, params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = functional_call(self.model, (params, buffers), (batch,))
        loss = F.cross_entropy(predictions, targets.long())
        return loss

    def update(self):
        """local model update"""
        self.model.train()
        criterion = nn.CrossEntropyLoss()#reduction='none')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        # optimizer = torch.optim.Adam(self.model.parameters())
        
        for e in range(self.E):
            # randomly select q fraction samples from data
            # according to the privacy analysis of moments accountant
            # training "Lots" are sampled by poisson sampling
            # idx = np.where(np.random.rand(len(self.torch_dataset[:][0])) < self.q)[0]

            # sampled_dataset = TensorDataset(self.torch_dataset[idx][0], self.torch_dataset[idx][1])
            sampled_dataset = TensorDataset(self.torch_dataset[:][0], self.torch_dataset[:][1])
            sample_data_loader = DataLoader(
                dataset=sampled_dataset,
                # batch_size=self.BATCH_SIZE,
                # shuffle=True
                batch_size=len(self.torch_dataset[:][0]),
                shuffle=False
            )
            
            optimizer.zero_grad()

            clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
            for batch_x, batch_y in sample_data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                # pred_y = self.model(batch_x.float())
                # loss = criterion(pred_y, batch_y.long())
                # local_loss = loss.detach().cpu().item()
                
                # # # bound l2 sensitivity (gradient clipping)
                # # # clip each of the gradient in the "Lot"
                # # for i in range(loss.size()[0]):
                # #     loss[i].backward(retain_graph=True)
                # #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                # #     for name, param in self.model.named_parameters():
                # #         clipped_grads[name] += param.grad 
                # #     self.model.zero_grad()

                # loss.backward(retain_graph=True)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                # for name, param in self.model.named_parameters():
                #     clipped_grads[name] += param.grad
                # self.model.zero_grad()

                """
                In order to apply the Gaussian mechanism to the gradient computation, it is
                necessary to bound its sensitivity.
                This can be achieved via **per-sample gradient clipping**
                """
                params = {k: v.detach() for k, v in self.model.named_parameters()}
                buffers = {k: v.detach() for k, v in self.model.named_buffers()}
                ft_compute_grad = grad(self.compute_loss)
                ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
                ft_per_sample_grads = ft_compute_sample_grad(params, buffers, batch_x, batch_y)

                pred_y = self.model(batch_x.float())
                loss = criterion(pred_y, batch_y.long())
                local_loss = loss.detach().cpu().item()
                self.model.zero_grad()

                for name, param in self.model.named_parameters():
                    grads = ft_per_sample_grads[name]
                    reshaped_tensor = grads.view(grads.size(0), -1)
                    norms = torch.linalg.norm(reshaped_tensor, dim=1, keepdim=True)
                    scale = torch.clamp(self.clip / norms, max=1.0).view(-1, *[1] * (grads.dim() - 1))
                    grads = grads * scale

                    clipped_grads[name] = torch.mean(grads, dim=0)
            
            if self.noise_scheme == 'gaussian':
                # add Gaussian noise
                for name, param in self.model.named_parameters():
                    clipped_grads[name] += gaussian_noise(clipped_grads[name].shape, self.sigma, device=self.device)
                    clipped_grads[name] = fixed_point_quantize(clipped_grads[name], wl=self.n_bits, fl=self.n_bits-2, rounding="stochastic")
            elif self.noise_scheme == 'binomial':
                # add Binomial noise
                for name, param in self.model.named_parameters():

                    #### Clip
                    X_max = self.clip
                    clipped_grads[name] = torch.clamp(clipped_grads[name], min=-X_max, max=X_max)

                    #### Quantize
                    # clipped_grads[name] /= X_max
                    # clipped_grads[name] = fixed_point_quantize(clipped_grads[name], wl=0, fl=4, rounding="stochastic")
                    # clipped_grads[name] *= X_max
                    
                    #### Add noise
                    noise = binomial_noise(clipped_grads[name].shape, self.trials, self.success_prob, device=self.device)
                    clipped_grads[name] += noise * 2 * X_max / (1 << self.n_bits - 1)
                    # clipped_grads[name] = float_quantize(clipped_grads[name], exp=2, man=4, rounding="stochastic")
                    # clipped_grads[name] = fixed_point_quantize(clipped_grads[name], wl=self.n_bits, fl=self.n_bits-2, rounding="stochastic")

            for name, param in self.model.named_parameters():
                param.grad = clipped_grads[name]
            
            # update local model
            optimizer.step()

            return local_loss



class FLServer(nn.Module):
    """ Server of Federated Learning
        1. Receive model (or gradients) from clients
        2. Aggregate local models (or gradients)
        3. Compute global model, broadcast global model to clients
    """
    def __init__(self, fl_param):
        super(FLServer, self).__init__()
        self.device = fl_param['device']
        self.client_num = fl_param['client_num']
        self.C = fl_param['C']      # (float) C in [0, 1]
        self.clip = fl_param['clip']
        self.T = fl_param['tot_T']  # total number of global iterations (communication rounds)
        self.n_update = 0

        self.data = []
        self.target = []
        for sample in fl_param['data'][self.client_num:]:
            self.data += [torch.tensor(sample[0]).to(self.device)]    # test set
            self.target += [torch.tensor(sample[1]).to(self.device)]  # target label

        self.input_size = int(self.data[0].shape[1])
        self.lr = fl_param['lr']
        
        # compute noise using moments accountant
        # self.sigma = compute_noise(1, fl_param['q'], fl_param['eps'], fl_param['E']*fl_param['tot_T'], fl_param['delta'], 1e-5)
        
        # calibration with subsampeld Gaussian mechanism under composition 
        # self.sigma = calibrating_sampled_gaussian(fl_param['q'], fl_param['eps'], fl_param['delta'], iters=fl_param['E']*fl_param['tot_T'], err=1e-3)
        # self.sigma = fl_param['sigma']
        # print("noise scale =", self.sigma)

        self.is_adaptive = fl_param['n_bits'] == 'adapt'
        self.n_bits = 4 if self.is_adaptive else fl_param['n_bits']
        self.initial_loss = None
        
        self.clients = [FLClient(fl_param['model'],
                                 fl_param['output_size'],
                                 fl_param['data'][i],
                                 fl_param['lr'],
                                 fl_param['E'],
                                 fl_param['batch_size'],
                                 fl_param['clip'],
                                 self.n_bits,
                                 fl_param['noise_scheme'],
                                 fl_param['sigma'] if 'sigma' in fl_param else None,
                                 fl_param['trials'] if 'trials' in fl_param else None,
                                 fl_param['success_prob'] if 'success_prob' in fl_param else None,
                                 self.device)
                        for i in range(self.client_num)]
        
        if fl_param['model'] == 'scatter':
            self.global_model = ScatterLinear(81, (7, 7), input_norm="GroupNorm", num_groups=27).to(self.device)
        else:
            self.global_model = fl_param['model'](self.input_size, fl_param['output_size']).to(self.device)
        
        self.weight = np.array([client.data_size * 1.0 for client in self.clients])
        self.broadcast(self.global_model.state_dict())

    def aggregated(self, idxs_users):
        """FedAvg"""
        model_par = [self.clients[idx].model.state_dict() for idx in idxs_users]
        new_par = copy.deepcopy(model_par[0])
        for name in new_par:
            new_par[name] = torch.zeros(new_par[name].shape).to(self.device)
        for idx, par in enumerate(model_par):
            w = self.weight[idxs_users[idx]] / np.sum(self.weight[:])
            for name in new_par:
                # new_par[name] += par[name] * (self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users]))
                new_par[name] += par[name] * (w / self.C)
        self.global_model.load_state_dict(copy.deepcopy(new_par))
        return self.global_model.state_dict().copy()

    def broadcast(self, new_par):
        """Send aggregated model to all clients"""
        for client in self.clients:
            client.recv(new_par.copy())

    def test_acc(self):
        self.global_model.eval()
        correct = 0
        tot_sample = 0
        for i in range(len(self.data)):
            t_pred_y = self.global_model(self.data[i])
            _, predicted = torch.max(t_pred_y, 1)
            correct += (predicted == self.target[i]).sum().item()
            tot_sample += self.target[i].size(0)
        acc = correct / tot_sample
        return acc

    def evaluate_model(self, batch_size=32, num_workers=4):
        self.global_model.eval()

        # Create a DataLoader for parallel data loading
        dataset = TensorDataset(torch.cat(self.data, 0), torch.cat(self.target, 0))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        all_targets = []

        with torch.no_grad():  # Disable gradient calculation for efficiency
          for data, target in data_loader:
              t_pred_y = self.global_model(data)
              _, predicted = torch.max(t_pred_y, 1)
              all_preds.extend(predicted.cpu().numpy())
              all_targets.extend(target.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        f1 = f1_score(all_targets, all_preds, average='weighted')
  
        return accuracy, precision, recall, f1

    def global_update(self):
        # idxs_users = np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False)
        idxs_users = np.sort(np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False))
        loss = 0
        for idx in idxs_users:
            tmp = self.clients[idx].update()
            loss += tmp
        loss /= idxs_users.size
        if self.initial_loss is None:
            self.initial_loss = loss
        self.broadcast(self.aggregated(idxs_users))
        if self.is_adaptive:
            print(loss, np.sqrt((self.initial_loss-0.55)/np.abs(loss-0.55)))
            self.n_bits = int(np.ceil(np.log2(np.sqrt((self.initial_loss-0.55)/np.abs(loss-0.55))*(1 << self.n_bits)+1)))
            self.set_n_bits()
        metrics = self.evaluate_model()
        metrics += (loss,)
        torch.cuda.empty_cache()
        return metrics

    def set_lr(self):
        for c in self.clients:
            c.lr = self.lr

    def set_n_bits(self):
        for c in self.clients:
            c.n_bits = self.n_bits
