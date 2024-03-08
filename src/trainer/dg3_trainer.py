
import os
import torch
from torch import nn
import time
import random
from progress.bar import Bar
from torch_geometric.loader import DataLoader
import copy
# from deepgate.arch.mlp import MLP
from deepgate.utils.utils import zero_normalization, AverageMeter, get_function_acc
from deepgate.utils.logger import Logger
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('/uac/gds/zyzheng23/projects/DeepGate3-Transformer/src')
from utils.utils import normalize_1
from utils.dag_utils import get_all_hops,get_random_hop
from utils.circuit_utils import complete_simulation, random_simulation

import networkx as nx
from scipy.optimize import linear_sum_assignment

TT_DIFF_RANGE = [0.2, 0.8]

def sample_structural_sim(subgraph, sample_cnt=100):
    stru_sim = []
    candidate_index = list(subgraph.keys())
    init_pair_idx = [random.sample(candidate_index, min(sample_cnt, len(candidate_index))), random.sample(candidate_index, min(sample_cnt, len(candidate_index)))]
    pair_idx = []
    for pair_k in range(sample_cnt):
        g1 = nx.DiGraph()
        g2 = nx.DiGraph()
        graph1 = subgraph[init_pair_idx[0][pair_k]]
        graph2 = subgraph[init_pair_idx[1][pair_k]]
        for edge_idx in range(len(graph1['edges'][0])):
            g1.add_edge(graph1['edges'][0][edge_idx], graph1['edges'][1][edge_idx])
        for edge_idx in range(len(graph2['edges'][0])):
            g2.add_edge(graph2['edges'][0][edge_idx], graph2['edges'][1][edge_idx])
        one_sim = nx.graph_edit_distance(g1, g2, timeout=0.1)
        stru_sim.append(one_sim)
        pair_idx.append([init_pair_idx[0][pair_k], init_pair_idx[1][pair_k]])
    stru_sim = torch.tensor(stru_sim)
    pair_idx = torch.tensor(pair_idx)
    return stru_sim, pair_idx

def sample_functional_tt(subgraph, sample_cnt=100):
    tt_list = []
    no_pi_list = []
    sample_list = []
    candidate_index = list(subgraph.keys())
    sample_list = random.sample(candidate_index, min(sample_cnt, len(candidate_index)))
    for idx in sample_list:
        if idx not in subgraph:
            continue
        g = subgraph[idx]
        tt_bin, no_pi = complete_simulation(g)
        tt_list.append(tt_bin)
        no_pi_list.append(no_pi)
    
    return tt_list, no_pi_list, sample_list

def DeepGate2_Tasks(graph, sample_cnt = 100):
    prob, full_states, level_list, fanin_list = random_simulation(graph, 1024)
    # PI Cover
    pi_cover = [[] for _ in range(len(prob))]
    for level in range(len(level_list)):
        for idx in level_list[level]:
            if level == 0:
                pi_cover[idx].append(idx)
            tmp_pi_cover = []
            for pre_k in fanin_list[idx]:
                tmp_pi_cover += pi_cover[pre_k]
            tmp_pi_cover = list(set(tmp_pi_cover))
            pi_cover[idx] += tmp_pi_cover
    # Sample 
    sample_idx = []
    tt_sim_list = []
    for _ in range(sample_cnt):
        while True:
            node_a = random.randint(0, len(prob)-1)
            node_b = random.randint(0, len(prob)-1)
            if node_a == node_b:
                continue
            if pi_cover[node_a] != pi_cover[node_b]:
                continue
            if abs(prob[node_a] - prob[node_b]) > 0.1:
                continue
            tt_dis = (full_states[node_a] != full_states[node_b]).sum() / len(full_states[node_a])
            if tt_dis > 0.2 and tt_dis < 0.8:
                continue
            if tt_dis == 0 or tt_dis == 1:
                continue
            break
        sample_idx.append([node_a, node_b])
        tt_sim_list.append(1-tt_dis)
    
    tt_index = torch.tensor(sample_idx)
    tt_sim = torch.tensor(tt_sim_list)
    return prob, tt_index, tt_sim

class Trainer():
    def __init__(self, 
                 args, 
                 model, 
                 training_id = 'default',
                 save_dir = './exp', 
                 lr = 1e-4,
                 loss_weight = [3.0, 1.0, 2.0],
                 emb_dim = 128, 
                 device = 'cpu', 
                 batch_size=32, 
                 num_workers=8, 
                 distributed = False, 
                 loss = 'l2'
                 ):
        super(Trainer, self).__init__()
        # Config
        self.args = args
        self.emb_dim = emb_dim
        self.device = device
        self.lr = lr
        self.lr_step = -1
        self.loss_weight = loss_weight
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.log_dir = os.path.join(save_dir, training_id)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        # Log Path
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        self.log_path = os.path.join(self.log_dir, 'log-{}.txt'.format(time_str))
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        self.loss = loss
        self.hop_per_circuit = 4
        # Distributed Training 
        self.local_rank = 0
        if self.distributed:
            if 'LOCAL_RANK' in os.environ:
                self.local_rank = int(os.environ['LOCAL_RANK'])
            self.device = 'cuda:%d' % self.local_rank
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            print('Training in distributed mode. Device {}, Process {:}, total {:}.'.format(
                self.device, self.rank, self.world_size
            ))
        else:
            print('Training in single device: ', self.device)
        
        # Loss 
        self.consis_loss_func = nn.MSELoss().to(self.device)
        if self.loss == 'l2':
            self.loss_func = nn.MSELoss().to(self.device)
        elif self.loss == 'l1':
            self.loss_func = nn.L1Loss().to(self.device)
        else:
            raise NotImplementedError
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss().to(self.device)
        self.l1_loss = nn.L1Loss().to(self.device)
        self.ce = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.cos_sim = nn.CosineSimilarity(dim=2, eps=1e-6).to(self.device)
        # self.reg_loss = nn.L1Loss().to(self.device)
        # self.clf_loss = nn.BCELoss().to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Model
        self.model = model.to(self.device)
        self.model_epoch = 0
        
        
        # Temp Data 
        self.stru_sim_tmp = {}
        
        # Logger
        if self.local_rank == 0:
            self.logger = Logger(self.log_path)
    
    def set_training_args(self, loss_weight=[], lr=-1, lr_step=-1, device='null'):
        if len(loss_weight) == 3 and loss_weight != self.loss_weight:
            print('[INFO] Update loss_weight from {} to {}'.format(self.loss_weight, loss_weight))
            self.loss_weight = loss_weight
        if lr > 0 and lr != self.lr:
            print('[INFO] Update learning rate from {} to {}'.format(self.lr, lr))
            self.lr = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        if lr_step > 0 and lr_step != self.lr_step:
            print('[INFO] Update learning rate step from {} to {}'.format(self.lr_step, lr_step))
            self.lr_step = lr_step
        if device != 'null' and device != self.device:
            print('[INFO] Update device from {} to {}'.format(self.device, device))
            self.device = device
            self.model = self.model.to(self.device)
            # self.reg_loss = self.reg_loss.to(self.device)
            # self.clf_loss = self.clf_loss.to(self.device)
            self.optimizer = self.optimizer
            # self.readout_rc = self.readout_rc.to(self.device)

    def save(self, filename):
        path = os.path.join(self.log_dir, filename)
        data = {
            'epoch': self.model_epoch, 
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(data, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in self.optimizer.param_groups:
            self.lr = param_group['lr']
        self.model_epoch = checkpoint['epoch']
        self.model.load(path)
        print('[INFO] Continue training from epoch {:}'.format(self.model_epoch))
        return path
    
    def resume(self):
        model_path = os.path.join(self.log_dir, 'model_last.pth')
        if self.local_rank == 0:
            print('[INFO] Load checkpoint from: ', model_path)
        if os.path.exists(model_path):
            self.load(model_path)
            return True
        else:
            return False

    def run_batch(self, batch):
        # Get all subgraph (k-hops)
        subgraph = get_all_hops(batch, self.args.k_hop) 
        # Get embeddings: hs/hf node-level, hop_hs/hop_hf graph-level
        hs, hf, hop_hs, hop_hf = self.model(batch, subgraph)
        
        # DG2 Tasks
        prob, tt_index, tt_sim = DeepGate2_Tasks(batch)
        pred_prob = self.model.prob_pred(hf).to(self.device)
        prob = prob.unsqueeze(1).to(self.device)
        l_fprob = self.l1_loss(pred_prob, prob)
        pred_tt_sim = torch.cosine_similarity(hf[tt_index[:, 0]], hf[tt_index[:, 1]], eps=1e-8)
        pred_tt_sim = normalize_1(pred_tt_sim).float().to(self.device)
        tt_sim = normalize_1(tt_sim).float().to(self.device)
        l_fttsim = self.l1_loss(pred_tt_sim, tt_sim)
        
        # Functional Tasks (Graph mask prediction)
        l_ftt = 0
        tt_list, no_pi_list, sample_list = sample_functional_tt(subgraph, 100)
        for graph_k, idx in enumerate(sample_list):
            if no_pi_list[graph_k] > 6:
                continue
            pred_tt = self.model.pred_tt(hop_hs[idx], no_pi_list[graph_k])
            label_tt = torch.tensor(tt_list[graph_k])
            while len(label_tt) < 64:
                label_tt = torch.cat([label_tt, label_tt])
            pred_tt = pred_tt.to(self.device)
            label_tt = label_tt.to(self.device)
            l_ftt += self.bce(pred_tt, label_tt.float())
        l_ftt /= len(sample_list)

        # Structural Tasks
        stru_sim, pair_idx = sample_structural_sim(subgraph, 32)
        hs_sim = torch.cosine_similarity(hop_hs[pair_idx[:, 0]], hop_hs[pair_idx[:, 1]], eps=1e-8)
        stru_sim = normalize_1(stru_sim).float().to(self.device)
        hs_sim = normalize_1(hs_sim).float().to(self.device)
        l_ssim = self.l1_loss(hs_sim, stru_sim)
        
        loss_status = {
            'prob': l_fprob,
            'tt_sim': l_fttsim,
            'tt_cls': l_ftt,
            'g_sim': l_ssim,
        }
        
        return loss_status

    def run_batch_mask(self, batch):
        hs, hf, pred_prob, pred_hop_tt = self.model(batch)
        
        # DG2 Tasks
        l_fprob = self.l1_loss(pred_prob, batch.prob.unsqueeze(1).to(self.device))
        pred_tt_sim = torch.cosine_similarity(hf[batch.tt_pair_index[0]], hf[batch.tt_pair_index[1]], eps=1e-8)
        pred_tt_sim = normalize_1(pred_tt_sim).float().to(self.device)
        tt_sim = normalize_1(batch.tt_sim).float().to(self.device)
        # l_fttsim = self.l1_loss(pred_tt_sim, tt_sim)
        
        # Graph mask prediction 
        pred_hop_tt_prob = nn.Sigmoid()(pred_hop_tt).to(self.device)
        pred_tt = torch.where(pred_hop_tt_prob > 0.5, 1, 0)
        print(pred_hop_tt_prob.shape, batch.hop_tt.shape)
        print(pred_hop_tt_prob.device, batch.hop_tt.device)
        l_ftt = self.l1_loss(pred_hop_tt_prob, batch.hop_tt.float())
        hamming_dist = torch.mean(torch.abs(pred_tt.float()-batch.hop_tt.float())).cpu()
        
        loss_status = {
            'prob': l_fprob,
            'tt_sim': 0,
            'tt_cls': l_ftt,
            'g_sim': 0,
        }
        
        return loss_status, hamming_dist


    def run_batch_baseline(self, batch):
        # Get all subgraph (k-hops)

        subgraph = get_random_hop(batch, self.args.k_hop, hop_per_circuit=self.hop_per_circuit)
        # Get embeddings: hs/hf node-level, hop_hs/hop_hf graph-level
        logits = self.model(batch, subgraph)
        
        # Functional Tasks (Graph mask prediction)
        l_ftt = 0
        hamming_dist = 0
        tt_list, no_pi_list, sample_list = sample_functional_tt(subgraph, 100)
        for graph_k, idx in enumerate(sample_list):
            if no_pi_list[graph_k] > 6:
                continue
            pred_prob = nn.Sigmoid()(logits[idx]).to(self.device)
            pred_tt = torch.where(pred_prob>0.5,1,0)
            label_tt = torch.tensor(tt_list[graph_k])
            while len(label_tt) < 64:
                label_tt = torch.cat([label_tt, label_tt])
            pred_tt = pred_tt.to(self.device)
            label_tt = label_tt.to(self.device)
            l_ftt += self.bce(pred_prob, label_tt.float())
            hamming_dist += torch.mean(torch.abs(pred_tt.float()-label_tt.float()))
        l_ftt /= len(sample_list)
        hamming_dist /= len(sample_list)

        loss_status = {
            'prob': 0,
            'tt_sim': 0,
            'tt_cls': l_ftt,
            'g_sim': 0,
        }
        
        return loss_status,hamming_dist

    
    def train(self, num_epoch, train_dataset, val_dataset):
        # Distribute Dataset
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                    num_workers=self.num_workers, sampler=train_sampler)
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                     num_workers=self.num_workers, sampler=val_sampler)
        else:
            train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
        
        # AverageMeter
        print(f'save model to {self.log_dir}')
        # self.save(os.path.join(self.log_dir, 'model_-1.pth'))
        self.save('model_last.pth')
        # Train
        print('[INFO] Start training, lr = {:.4f}'.format(self.optimizer.param_groups[0]['lr']))
        for epoch in range(num_epoch): 
            for phase in ['train', 'val']:
            # for phase in ['val']:
                if phase == 'train':
                    dataset = train_dataset
                    self.model.train()
                    self.model.to(self.device)
                else:
                    dataset = val_dataset
                    self.model.eval()
                    self.model.to(self.device)
                if self.local_rank == 0:
                    bar = Bar('{} {:}/{:}'.format(phase, epoch, num_epoch), max=len(dataset))
                hamming_list = []
                lprob = []
                lall = []
                lttcls = []
                # lttsim=[]
                # for iter_id, batch in enumerate(dataset):
                    # print(torch.mean(batch.hop_tt.float()))
                    
                if self.local_rank == 0:
                    bar = Bar('{} {:}/{:}'.format(phase, epoch, num_epoch), max=len(dataset))

                for iter_id, batch in enumerate(dataset):
                    time_stamp = time.time()
                    batch = batch.to(self.device)                    
                    loss_dict,hamming_dist = self.run_batch_mask(batch)

                    hamming_list.append(hamming_dist)
                    lprob.append(loss_dict['prob'].item())
                    loss = (loss_dict['prob'] * self.args.w_prob + \
                            loss_dict['tt_sim'] * self.args.w_tt_sim + \
                            loss_dict['tt_cls'] * self.args.w_tt_cls + \
                            loss_dict['g_sim'] * self.args.w_g_sim) / (self.args.w_prob + self.args.w_tt_sim + self.args.w_tt_cls + self.args.w_g_sim)
                    lall.append(loss.item())
                    lttcls.append(loss_dict['tt_cls'].item())
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    if self.local_rank == 0:
                        Bar.suffix = '[{:}/{:}] |Tot: {total:} |ETA: {eta:} '.format(iter_id, len(dataset), total=bar.elapsed_td, eta=bar.eta_td)
                        Bar.suffix += '|Prob: {:.4f} |TTCLS: {:.4f} |Loss: {:.4f} |Dist: {:.4f}'.format(
                            torch.mean(torch.tensor(lprob)).item(), torch.mean(torch.tensor(lttcls)).item(),
                            torch.mean(torch.tensor(lall)).item(), torch.mean(torch.tensor(hamming_list)).item()
                        )
                        bar.next()
                        # bar.suffix = '({phase}) Epoch: {epoch} | Iter: {iter} | Time: {time:.4f}'.format(
                        #     phase=phase, epoch=epoch, iter=iter_id, time=time.time()-time_stamp
                        # )
                        # for loss_key in loss_dict:
                        #     if loss_dict[loss_key] !=0:
                        #         bar.suffix += ' | {}: {:.4f}'.format(loss_key, loss_dict[loss_key].item())
                        # bar.suffix += ' | hamming_dist: {:.4f}'.format(hamming_dist)
                        # bar.next()
                        # output_log = '({phase}) Epoch: {epoch} | Iter: {iter} | Time: {time:.4f}'.format(
                        #     phase=phase, epoch=epoch, iter=iter_id, time=time.time()-time_stamp
                        # )
                        # for loss_key in loss_dict:
                        #     if loss_dict[loss_key] !=0:
                        #         output_log += ' | {}: {:.4f}'.format(loss_key, loss_dict[loss_key].item())
                        # output_log += ' | hamming_dist: {:.4f}'.format(hamming_dist)
                        # print(output_log)
                print(f'overall hamming distance:{torch.mean(torch.tensor(hamming_list))}')
                print(f'overall probability loss:{torch.mean(torch.tensor(lprob))}')
                if self.local_rank == 0:
                    self.logger.write('{} Epoch: {:}/{:}| Prob: {:.4f}| TTCLS: {:.4f}| Loss: {:.4f}| Dist: {:.4f}'.format(
                        phase, epoch, num_epoch, 
                        torch.mean(torch.tensor(lprob)).item(), torch.mean(torch.tensor(lttcls)).item(),
                        torch.mean(torch.tensor(lall)).item(), torch.mean(torch.tensor(hamming_list)).item()
                    ))
            
            # Learning rate decay
            self.model_epoch += 1
            if self.lr_step > 0 and self.model_epoch % self.lr_step == 0:
                self.lr *= 0.1
                if self.local_rank == 0:
                    print('[INFO] Learning rate decay to {}'.format(self.lr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                    
        # del train_dataset
        # del val_dataset
        

