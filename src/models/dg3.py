import torch 
import deepgate as dg
import torch.nn as nn 

from .mlp import MLP
from .dg2 import DeepGate2

from .plain_tf import Plain_Transformer
from .hop_tf import Hop_Transformer
from .path_tf import Path_Transformer
from .baseline_tf import Baseline_Transformer
from .mlp import MLP
from .tf_pool import tf_Pooling
import numpy as np
_transformer_factory = {
    'baseline': None,
    'plain': Plain_Transformer,
    'hop': Hop_Transformer, 
    'path': Path_Transformer
}

import torch.nn as nn

class DeepGate3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_tt_len = 64
        self.hidden = 128
        self.max_path_len = 257
        self.tf_arch = args.tf_arch
        # Tokenizer
        self.tokenizer = DeepGate2()
        self.tokenizer.load_pretrained(args.pretrained_model_path)

        #special token
        self.cls_token = nn.Parameter(torch.randn([self.hidden,]))
        self.cls_path_token = nn.Parameter(torch.randn([self.hidden,]))
        self.dc_token = nn.Parameter(torch.randn([self.hidden,]))
        self.zero_token = nn.Parameter(torch.randn([self.hidden,]))
        self.one_token = nn.Parameter(torch.randn([self.hidden,]))
        self.pad_token = torch.zeros([self.hidden,]) # dont learn
        self.pool_max_length = 10
        self.PositionalEmbedding = nn.Embedding(10,self.hidden)
        self.Path_Pos = nn.Embedding(self.max_path_len,self.hidden)
        
        # Transformer 
        if args.tf_arch != 'baseline':
            self.transformer = _transformer_factory[args.tf_arch](args)
        

        #pooling layer
        pool_layer = nn.TransformerEncoderLayer(d_model=self.hidden, nhead=4, batch_first=True)
        self.hop_func_tf = nn.TransformerEncoder(pool_layer, num_layers=1)
        self.hop_struc_tf = nn.TransformerEncoder(pool_layer, num_layers=1)
        self.path_struc_tf = nn.TransformerEncoder(pool_layer, num_layers=1)
        
        

        # Prediction 
        self.hop_head = nn.Sequential(nn.Linear(self.hidden, self.hidden*4),
                        nn.ReLU(),
                        nn.LayerNorm(self.hidden*4),
                        nn.Linear(self.hidden*4, self.max_tt_len))
        self.readout_prob = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_path_len = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_path_and_ratio = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_hop_level = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_level = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_num = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.connect_head = MLP(
            dim_in=self.args.token_emb*2, dim_hidden=self.args.mlp_hidden, dim_pred=3, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.on_path_head = MLP(
            dim_in=self.args.token_emb*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.on_hop_head = MLP(
            dim_in=self.args.token_emb*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_path_len = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )

        #Similarity
        self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    
    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        state_dict_ = checkpoint['state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = self.state_dict()
        
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)
        
        
    def forward(self, g):
        hs, hf = self.tokenizer(g)
        hf = hf.detach()
        hs = hs.detach()
        # Refine-Transformer 
        if self.tf_arch != 'baseline':

            hf_tf, hs_tf = self.transformer(g, hs, hf)
            #function
            hf = hf + hf_tf
            #structure
            hs = hs + hs_tf

        #=========================================================
        #======================GATE-level=========================
        #=========================================================
            
        #gate-level pretrain task : predict pari-wise TT sim
        gate_tt_sim = self.sim(hf[g.tt_pair_index[0]],hf[g.tt_pair_index[1]])

        #gate-level pretrain task : predict global probability
        prob = self.readout_prob(hf)

        #gate-level pretrain task : predict global level
        pred_level = self.readout_level(hs)

        #gate-level pretrain task : predict connection
        gates = hs[g.connect_pair_index]
        gates = gates.permute(1,2,0).reshape(-1,self.hidden*2)
        pred_connect = self.connect_head(gates)

        #=========================================================
        #======================PATH-level=========================
        #=========================================================

        #path-level pretrain task : on-path prediction, path num prediction
        path_hs =  torch.cat([self.cls_path_token.reshape([1,1,-1]).repeat(g.paths.shape[0],1,1),hs[g.paths]],dim=1)
        # False = compute attention, True = mask 
        path_mask = torch.stack([torch.cat([torch.zeros([g.paths_len[i]+1]),torch.ones([256-g.paths_len[i]])],dim=0) for i in range(g.paths.shape[0])]) 
        path_mask = torch.where(path_mask==1, True, False).to(hs.device)
        pos = torch.arange(path_hs.shape[1]).unsqueeze(0).repeat(path_hs.shape[0],1).to(hs.device)
        path_hs = path_hs + self.Path_Pos(pos)
        path_hs = self.path_struc_tf(path_hs, src_key_padding_mask = path_mask)[:, 0]

        #on-path prediction
        on_path_emb = torch.cat([hs[g.ninp_node_index],path_hs[g.ninp_path_index]],dim=1)
        on_path_logits = self.on_path_head(on_path_emb)

        #path num prediction
        pred_path_len = self.readout_path_len(path_hs)
        pred_path_and_ratio = self.readout_path_and_ratio(path_hs)  # Predict the ratio of AND gates in the path
        # pred_path_gate1 = self.readout_gate1(path_hs)
        # pred_path_gate2 = self.readout_gate2(path_hs)

        #=========================================================
        #======================GRAPH-level========================
        #=========================================================

        #graph-level pretrain task : predict truth table & pair-wise TT sim
        hop_hf = []
        hf_masks = []
        for i in range(g.hop_po.shape[0]):
            pi_idx = g.hop_pi[i][g.hop_pi_stats[i]!=-1].squeeze(-1)
            pi_hop_stats = g.hop_pi_stats[i]
            pi_emb = hf[pi_idx]
            pi_emb = []
            for j in range(8):
                if pi_hop_stats[j] == -1:
                    continue
                elif pi_hop_stats[j] == 0:
                    pi_emb.append(self.zero_token)
                elif pi_hop_stats[j] == 1:
                    pi_emb.append(self.one_token)
                elif pi_hop_stats[j] == 2:
                    pi_emb.append(hf[g.hop_pi[i][j]])
            # add dont care token
            while len(pi_emb) < 6:
                pi_emb.insert(0,self.dc_token)
            # pad seq to fixed length
            hf_mask = [1 for _ in range(len(pi_emb))]
            while len(pi_emb) < 8:
                pi_emb.append(self.pad_token.to(hf.device))
                hf_mask.append(0)
            pi_emb = torch.stack(pi_emb) # 8 128
            po_emb = hf[g.hop_po[i]] # 1 128
            hop_hf.append(torch.cat([self.cls_token.unsqueeze(0),pi_emb,po_emb], dim=0)) 
            hf_mask.insert(0,1)
            hf_mask.append(1)
            hf_masks.append(torch.tensor(hf_mask))

        hop_hf = torch.stack(hop_hf) #bs seq_len hidden
        pos = torch.arange(hop_hf.shape[1]).unsqueeze(0).repeat(hop_hf.shape[0],1).to(hf.device)
        hop_hf = hop_hf + self.PositionalEmbedding(pos)

        hf_masks = 1 - torch.stack(hf_masks).to(hf.device).float() #bs seq_len 
        hf_masks = torch.where(hf_masks==1, True, False).to(hf.device)
        hop_hf = self.hop_func_tf(hop_hf,src_key_padding_mask = hf_masks)
        hop_hf = hop_hf[:,0]

        #pair-wise TT sim prediction
        hop_tt_sim = self.sim(hop_hf[g.hop_forward_index[g.hop_pair_index[0]]], hop_hf[g.hop_forward_index[g.hop_pair_index[0]]])
        # truth table prediction
        hop_tt = self.hop_head(hop_hf)

        #graph-level pretrain task : PPA prediction
        hop_hs = []
        hs_masks = []
        for i in range(g.hop_po.shape[0]):
            pi_idx = g.hop_pi[i][g.hop_pi_stats[i]!=-1].squeeze(-1)
            pi_hop_stats = g.hop_pi_stats[i]
            pi_emb = hs[pi_idx]
            pi_emb = []
            for j in range(8):
                if pi_hop_stats[j] == -1:
                    continue
                else:
                    pi_emb.append(hs[g.hop_pi[i][j]])
            # pad seq to fixed length
            hs_mask = [1 for _ in range(len(pi_emb))]
            while len(pi_emb) < 8:
                pi_emb.append(self.pad_token.to(hs.device))
                hs_mask.append(0)

            pi_emb = torch.stack(pi_emb) # 8 128
            po_emb = hs[g.hop_po[i]] # 1 128
            hop_hs.append(torch.cat([self.cls_token.unsqueeze(0),pi_emb,po_emb], dim=0)) 
            hs_mask.insert(0,1)
            hs_mask.append(1)
            hs_masks.append(torch.tensor(hs_mask))

        hop_hs = torch.stack(hop_hs) #bs seq_len hidden
        pos = torch.arange(hop_hs.shape[1]).unsqueeze(0).repeat(hop_hs.shape[0],1).to(hs.device)
        hop_hs = hop_hs + self.PositionalEmbedding(pos)

        hs_masks = 1 - torch.stack(hs_masks).to(hs.device).float() #bs seq_len 
        hs_masks = torch.where(hs_masks==1, True, False).to(hs.device)

        hop_hs = self.hop_struc_tf(hop_hs,src_key_padding_mask = hs_masks)
        hop_hs = hop_hs[:,0]
        #pari-wise GED prediction 
        pred_GED = self.sim(hop_hs[g.hop_pair_index[0]], hop_hs[g.hop_pair_index[1]])

        #gate number prediction
        pred_hop_num = self.readout_num(hop_hs)

        #hop level prediction
        pred_hop_level = self.readout_hop_level(hop_hs)

        #graph-level pretrain task : on hop prediction
        on_hop_emb = torch.cat([hs[g.ninh_node_index],path_hs[g.ninh_hop_index]],dim=1)
        on_hop_logits = self.on_hop_head(on_hop_emb)
        
        result = {
            'emb':
            {
                'hf':hf,
                'hs':hs,
            },
            'node':
            {
                'prob':prob,
                'level':pred_level,
                'connect':pred_connect,
                'tt_sim':gate_tt_sim,
            },
            'path':{
                'on_path':on_path_logits,
                'length':pred_path_len,
                'AND': pred_path_and_ratio, 
                # 'AND': pred_path_gate1,
                # 'NOT': pred_path_gate2,
            },
            'hop':{
                'tt':hop_tt,
                'tt_sim':hop_tt_sim,
                'area':pred_hop_num,
                'time':pred_hop_level,
                'on_hop':on_hop_logits,
                'GED':pred_GED,
            }
        }
        return result
    
