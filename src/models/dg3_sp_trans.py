import torch 
import deepgate as dg
import torch.nn as nn 

from .mlp import MLP
from .dg2 import DeepGate2

from .plain_tf import Plain_Transformer
from .plain_tf_linear import Sparse_Transformer
from .hop_tf import Hop_Transformer
from .path_tf import Path_Transformer
from .baseline_tf import Baseline_Transformer
from .hybrid_tf import Hybrid_Transformer
from .mlp import MLP
from .history import History
# from .tf_pool import tf_Pooling
from dg_datasets.dg3_multi_parser import OrderedData
import math
from torch_scatter import scatter_add, scatter_mean, scatter_max
import numpy as np
import torch.nn.functional as F
_transformer_factory = {
    'baseline': None,
    'plain': Plain_Transformer,
    'hop': Hop_Transformer, 
    'path': Path_Transformer,
    'hybrid': Hybrid_Transformer,
    'sparse': Sparse_Transformer,
}

import torch.nn as nn
import time
import copy

from .gnn_layers import get_simple_gnn_layer, EDGE_GNN_TYPES


def build_graph(g, area_nodes, area_nodes_stats, area_faninout_cone, prob):
    area_g = OrderedData()
    nodes = area_nodes[area_nodes != -1]
    pi_mask = (area_nodes_stats == 1)[:len(nodes)]
    area_g.nodes = nodes
    area_g.gate = g.gate[nodes]
    area_g.gate[pi_mask] = 0
    # print(prob.shape)
    # print(nodes.shape)
    # print(torch.max(nodes))
    area_g.prob = prob[nodes]
    area_g.forward_level = g.forward_level[nodes]
    area_g.backward_level = g.backward_level[nodes]
    area_g.forward_index = torch.tensor(range(len(nodes)))
    area_g.backward_index = torch.tensor(range(len(nodes)))
    
    # Edge_index
    glo_to_area = {}
    for i, node in enumerate(nodes):
        glo_to_area[node.item()] = i
    area_edge_index = []
    for edge in g.edge_index.t():
        if edge[0].item() in glo_to_area and edge[1].item() in glo_to_area:
            area_edge_index.append([glo_to_area[edge[0].item()], glo_to_area[edge[1].item()]])
    area_edge_index = torch.tensor(area_edge_index).t()
    area_g.edge_index = area_edge_index
    
    area_g.fanin_fanout_cones = area_faninout_cone
    area_g.batch = torch.zeros(len(nodes), dtype=torch.long)
    return area_g

def merge_area_g(batch_g, g):
    no_nodes = batch_g.nodes.shape[0]
    batch_g.nodes = torch.cat([batch_g.nodes, g.nodes])
    batch_g.gate = torch.cat([batch_g.gate, g.gate])
    batch_g.prob = torch.cat([batch_g.prob, g.prob])
    batch_g.forward_level = torch.cat([batch_g.forward_level, g.forward_level])
    batch_g.backward_level = torch.cat([batch_g.backward_level, g.backward_level])
    batch_g.edge_index = torch.cat([batch_g.edge_index, g.edge_index + no_nodes], dim=1)
    batch_g.fanin_fanout_cones = torch.cat([batch_g.fanin_fanout_cones, g.fanin_fanout_cones], dim=0)
    batch_g.batch = torch.cat([batch_g.batch, torch.tensor([batch_g.batch.max() + 1] * len(g.nodes)).to(batch_g.batch.device)])
    
    batch_g.forward_index = torch.tensor(range(len(batch_g.nodes))).to(batch_g.batch.device)
    batch_g.backward_index = torch.tensor(range(len(batch_g.nodes))).to(batch_g.batch.device)
    
    return batch_g
    
def generate_orthogonal_vectors(n, dim):
    if n < dim * 8:
        # Choice 1: Generate n random orthogonal vectors in R^dim
        # Generate an initial random vector
        v0 = np.random.randn(dim)
        v0 /= np.linalg.norm(v0)
        # Generate n-1 additional vectors
        vectors = [v0]
        for i in range(n-1):
            while True:
                # Generate a random vector
                v = np.random.randn(dim)

                # Project the vector onto the subspace spanned by the previous vectors
                for j in range(i+1):
                    v -= np.dot(v, vectors[j]) * vectors[j]

                if np.linalg.norm(v) > 0:
                    # Normalize the vector
                    v /= np.linalg.norm(v)
                    break

            # Append the vector to the list
            vectors.append(v)
    else: 
        # Choice 2: Generate n random vectors:
        vectors = np.random.rand(n, dim) - 0.5
        for i in range(n):
            vectors[i] = vectors[i] / np.linalg.norm(vectors[i])

    return vectors

def generate_hs_init(G, hs, no_dim):
    if G.batch == None:
        batch_size = 1
    else:
        batch_size = G.batch.max().item() + 1
    for batch_idx in range(batch_size):
        if G.batch == None:
            pi_mask = (G.forward_level == 0)
        else:
            pi_mask = (G.batch == batch_idx) & (G.forward_level == 0)
        pi_node = G.forward_index[pi_mask]
        pi_vec = generate_orthogonal_vectors(len(pi_node), no_dim)
        hs[pi_node] = torch.tensor(np.array(pi_vec), dtype=torch.float)
    
    return hs

class StructureExtractor(nn.Module):

    def __init__(self, embed_dim, gnn_type="gcn", num_layers=1,
                 batch_norm=True, concat=True, khopgnn=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn
        self.concat = concat
        self.gnn_type = gnn_type
        layers = []
        for _ in range(num_layers):
            layers.append(get_simple_gnn_layer(gnn_type, embed_dim, **kwargs))
        self.gcn = nn.ModuleList(layers)

        self.relu = nn.GELU()
        self.batch_norm = batch_norm
        inner_dim = (num_layers + 1) * embed_dim if concat else embed_dim

        if batch_norm:
            self.bn = nn.BatchNorm1d(inner_dim)

        self.out_proj = nn.Linear(inner_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr=None,
            subgraph_indicator_index=None, agg="sum"):
        x_cat = [x]
        for gcn_layer in self.gcn:
            # if self.gnn_type == "attn":
            #     x = gcn_layer(x, edge_index, None, edge_attr=edge_attr)
            if self.gnn_type in EDGE_GNN_TYPES:
                if edge_attr is None:
                    x = self.relu(gcn_layer(x, edge_index))
                else:
                    x = self.relu(gcn_layer(x, edge_index, edge_attr=edge_attr))
            else:
                x = self.relu(gcn_layer(x, edge_index))

            if self.concat:
                x_cat.append(x)

        if self.concat:
            x = torch.cat(x_cat, dim=-1)

        if self.khopgnn:
            if agg == "sum":
                x = scatter_add(x, subgraph_indicator_index, dim=0)
            elif agg == "mean":
                x = scatter_mean(x, subgraph_indicator_index, dim=0)
            return x
        if self.num_layers > 0 and self.batch_norm:
            x = self.bn(x)

        x = self.out_proj(x)
        return x

class DeepGate4(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_tt_len = 64
        self.hidden = 128
        self.max_path_len = 257
        self.pool_depth = 2
        self.dim_feedforward = self.hidden*4
        self.tf_arch = args.tf_arch
        # Tokenizer
        self.tokenizer = DeepGate2()
        # self.tokenizer.load_pretrained(args.pretrained_model_path)

        #special token
        self.cls_token_hf = nn.Parameter(torch.randn([self.hidden,]))
        self.cls_token_hs = nn.Parameter(torch.randn([self.hidden,]))
        self.cls_path_token = nn.Parameter(torch.randn([self.hidden,]))
        self.dc_token = nn.Parameter(torch.randn([self.hidden,]))
        self.zero_token = nn.Parameter(torch.randn([self.hidden,]))
        self.one_token = nn.Parameter(torch.randn([self.hidden,]))
        self.pad_token = torch.zeros([self.hidden,]) # don't learn
        self.pool_max_length = 10
        self.hf_PositionalEmbedding = nn.Embedding(33,self.hidden)
        self.hs_PositionalEmbedding = nn.Embedding(33,self.hidden)
        self.Path_Pos = nn.Embedding(self.max_path_len,self.hidden)

        # Offline Global Embedding
        self.hf_history = History(num_embeddings=150000, embedding_dim=self.hidden)
        self.hs_history = History(num_embeddings=150000, embedding_dim=self.hidden)
        self.mk = torch.zeros(150000)

        # Structure model
        self.sinu_pe = self.sinuous_positional_encoding(1000, self.hidden)
        self.abs_pe_embedding = nn.Linear(self.hidden, self.hidden)
        self.structure_extractor = StructureExtractor(self.hidden, num_layers=3)

        # Refine Transformer 
        if args.tf_arch != 'baseline' :
            self.transformer = _transformer_factory[args.tf_arch](args, hidden=self.hidden)


        # Pooling Transformer
        # pool_layer = nn.TransformerEncoderLayer(d_model=self.hidden, nhead=4, dim_feedforward=self.dim_feedforward, batch_first=True)
        # self.hop_func_tf = nn.TransformerEncoder(pool_layer, num_layers=self.pool_depth)
        # self.hop_struc_tf = nn.TransformerEncoder(pool_layer, num_layers=self.pool_depth)
        # self.path_struc_tf = nn.TransformerEncoder(pool_layer, num_layers=3)
        
        self.init_MLP()
        self.sigmoid = nn.Sigmoid()
        
    def sinuous_positional_encoding(self, seq_len, d_model):

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))


        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe
    
    def init_MLP(self):
        self.hop_head = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=self.max_tt_len, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )         
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
        self.gate_dis = MLP(
            dim_in=self.args.token_emb*4, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.proj_gate_ttsim = MLP(
            dim_in=self.args.token_emb*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.proj_hop_ttsim = MLP(
            dim_in=self.args.token_emb*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.proj_GED = MLP(
            dim_in=self.args.token_emb*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
    
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

    def reset_parameters(self, g):
        #set to zero
        self.hf_history.reset_parameters()
        self.hs_history.reset_parameters()
        self.mk.fill_(0)

        PI_idx = g.forward_index[g.forward_level==0].cpu()

        #init hs

        # #generate orthogonal vectors as initialized hf
        hs = generate_orthogonal_vectors(len(PI_idx), self.hidden)
        hs = torch.tensor(np.array(hs)).cpu().float()
        self.hs_history.push(hs, PI_idx)

        # # use speical token(level 0) as initailized hs
        # hs = self.hs0.unsqueeze(0).repeat(len(PI_idx),1)
        # hs = torch.tensor(np.array(hs)).cpu().float()
        # self.hs_history.push(hs, PI_idx)

        #init hf
        prob_mask = copy.deepcopy(g.prob)
        prob_mask = prob_mask.unsqueeze(-1)
        prob_mask = prob_mask[PI_idx]
        hf = prob_mask.repeat(1, self.hidden).clone()
        hf = hf.float()
        self.hf_history.push(hf, PI_idx)

        #init mask
        self.mk[PI_idx] = 1

    def forward(self, g, skip_path=False, skip_hop=False, large_ckt=False, phase='train'):
        # assert large_ckt, 'the model is designed for large circuit'

        """
        mk: 0 denotes need to update, 1 denotes the node have been updated
        
        """

        device = g.gate.device
        g.nodes = g.nodes.cpu()

        if large_ckt==False:
            # Refine-Transformer 
            hs, hf = self.tokenizer(g, g.prob)
            if self.tf_arch != 'baseline':
                hf_tf, hs_tf = self.transformer(g, hf, hs)
                #function
                hf = hf + hf_tf
                #structure
                hs = hs + hs_tf      
        else:
            hf_detach = self.hf_history.pull(g.nodes)
            hs_detach = self.hs_history.pull(g.nodes)

            abs_pe = self.sinu_pe[g.forward_level.cpu()].to(device)

            lhs = self.abs_pe_embedding(abs_pe)
            lhs = self.structure_extractor(lhs, g.edge_index)
            
            # t1 = time.time()
            # hs, hf = self.tokenizer(g, g.prob, hf_detach, hs_detach+lhs, self.mk)

            # hs, hf = self.tokenizer(g, g.prob, hf_detach, hs_detach, self.mk)

            hs, hf = self.tokenizer(g, g.prob, hf_detach, hs_detach, lhs, self.mk)


            # t2 = time.time()
            # print(f'dg2 runtime:{t2-t1}s')

            # Add local structure embedding

            # hs = hs + lhs

            if self.tf_arch!='baseline':

                if self.tf_arch!='sparse':
                    # hf_tf, hs_tf = self.transformer(g, hf.clone(), hs.clone(), self.mk)
                    hf_tf, hs_tf = self.transformer(g, hf.clone(), hs.clone()+lhs, self.mk)
                else:
                    hf_tf, hs_tf = self.transformer(g, hf.clone(), hs.clone(), self.mk)
                # hf_tf, hs_tf = self.transformer(g, hf.clone(), hs.clone())
                
                hf = hf + hf_tf
                hs = hs + hs_tf
                # t3 = time.time()
                # print(f'transformer runtime:{t3-t2}s')

            #update once
            update_idx = g.nodes[self.mk[g.nodes]==0]

            update_hf = []
            update_hs = []
            for idx in update_idx:
                update_hf.append(torch.mean(hf[g.nodes==idx], dim=0))
                update_hs.append(torch.mean(hs[g.nodes==idx], dim=0))
            update_hf = torch.stack(update_hf)
            update_hs = torch.stack(update_hs)

            self.hf_history.push(x = update_hf, n_id = update_idx)
            self.hs_history.push(x = update_hs, n_id = update_idx)
            
            self.mk[g.nodes] = 1
        
        
        if phase=='test':
            return None
        #=========================================================
        #======================GATE-level=========================
        #=========================================================

        #gate-level pretrain task : predict global probability
        prob = self.readout_prob(hf)
        prob = F.sigmoid(prob)

        #gate-level pretrain task : predict global level
        pred_level = self.readout_level(hs)
        
        update_prob = []
        update_level = []
        for idx in update_idx:
            update_prob.append(torch.mean(prob[g.nodes==idx], dim=0))
            update_level.append(torch.mean(pred_level[g.nodes==idx], dim=0))

        update_prob = torch.stack(update_prob)
        update_level = torch.stack(update_level)

        #=========================================================
        #=======================pair-wise=========================
        #=========================================================

        # con_pair_emb = None 
        # con_label = None
        # pred_con = None
        # pred_tt_sim = None
        # tt_pair_emb = None
        # tt_label = None
        
        tt_pair_emb = []
        tt_label = []
        dest_emb = []
        src_emb = self.hf_history.pull(g.tt_pair_index[0,:][self.mk[g.tt_pair_index[0,:].cpu()]==1].cpu()).squeeze(-1)


        # use miter to get the tt sim
        # XNOR(A,B)=(A and B)or(!A and !B)
        

        for i in range(g.tt_pair_index.shape[1]):
            src = g.tt_pair_index[0,i]
            dest = g.tt_pair_index[1,i]
            if self.mk[src]==1:
                # src_emb = self.hf_history.pull(src.cpu()).squeeze()
                dest_emb.append(torch.mean(hf[g.nodes==dest.cpu()], dim=0))
                tt_label.append(g.tt_sim[i])

        

        if dest_emb != []:
            tt_pair_emb = torch.cat([src_emb,torch.stack(dest_emb)],dim=1)
            tt_label = torch.stack(tt_label)
        else:
            tt_pair_emb = None
            tt_label = None

        if tt_pair_emb is None:
            pred_tt_sim = None
        elif tt_pair_emb.shape[0]==1:
            tt_pair_emb = tt_pair_emb.repeat(2,1)
            tt_label = tt_label.repeat(2)

            # pred_tt_sim = torch.abs(self.sigmoid(tt_pair_emb[:,:self.hidden])-self.sigmoid(tt_pair_emb[:,self.hidden:])).mean(dim=1)
            pred_tt_sim = self.proj_gate_ttsim(tt_pair_emb)
            # pred_tt_sim = self.sigmoid(pred_tt_sim)
            pred_tt_sim = torch.clamp(pred_tt_sim, 0., 1.)
        else:

            # pred_tt_sim = torch.abs(self.sigmoid(tt_pair_emb[:,:self.hidden])-self.sigmoid(tt_pair_emb[:,self.hidden:])).mean(dim=1)
            pred_tt_sim = self.proj_gate_ttsim(tt_pair_emb)
            # pred_tt_sim = self.sigmoid(pred_tt_sim)
            pred_tt_sim = torch.clamp(pred_tt_sim, 0., 1.)
            

        con_pair_emb = []
        con_label = []
        dest_emb = []

        src_emb = self.hs_history.pull(g.connect_pair_index[0,:][self.mk[g.connect_pair_index[0,:].cpu()]==1].cpu()).squeeze(-1)

        for i in range(g.connect_pair_index.shape[1]):
            src = g.connect_pair_index[0,i]
            dest = g.connect_pair_index[1,i]
            if self.mk[src]==1:
                # src_emb = self.hs_history.pull(src.cpu()).squeeze()
                dest_emb.append(torch.mean(hs[g.nodes==dest.cpu()], dim=0))
                con_label.append(g.connect_label[i])

        if dest_emb != []:
            con_pair_emb = torch.cat([src_emb,torch.stack(dest_emb)],dim=1)
            con_label = torch.stack(con_label)
        else:
            con_pair_emb = None 
            con_label = None
        
        if con_pair_emb is None:
            pred_con = None
        elif con_pair_emb.shape[0]==1:
            con_pair_emb = con_pair_emb.repeat(2,1)
            con_label = con_label.repeat(2)
            pred_con = self.connect_head(con_pair_emb)
            pred_con = self.sigmoid(pred_con)
        else :
            pred_con = self.connect_head(con_pair_emb)
            pred_con = self.sigmoid(pred_con)




        result = {
            # 'emb':
            # {
            #     'hf':hf,
            #     'hs':hs,
            # },
            'node':
            {
                'update_idx':update_idx,
                'prob':update_prob,
                'level':update_level,
            },
            'pair':
            {
                'pred_tt_sim':pred_tt_sim,
                'tt_label':tt_label,
                'pred_con':pred_con,
                'con_label':con_label,
            },
        }
        return result

