import torch 
import deepgate as dg
import torch.nn as nn 
from .pool import PoolNet
from .mlp import MLP
from .dg2 import DeepGate2

from .plain_tf import Plain_Transformer
from .hop_tf import Hop_Transformer
from .mlp import MLP
from .tf_pool import tf_Pooling

_transformer_factory = {
    'plain': Plain_Transformer,
    'hop': Hop_Transformer, 
}

class DeepGate3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Tokenizer
        self.tokenizer = DeepGate2()
        self.tokenizer.load_pretrained(args.pretrained_model_path)
        
        # Transformer 
        self.transformer = _transformer_factory[args.tf_arch](args)
        
        # Prediction 
        self.hs_mask_token = nn.Parameter(torch.zeros([self.args.token_emb,]))
        self.hf_mask_token = nn.Parameter(torch.zeros([self.args.token_emb,]))
        self.mask_pred_hs = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=self.args.token_emb, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer=self.args.act_layer
        )
        self.mask_pred_hf = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=self.args.token_emb, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer=self.args.act_layer
        )
        self.hs_pool = tf_Pooling(args)
        self.hf_pool = tf_Pooling(args)
        self.tt_pred = [nn.Sequential(nn.Linear(self.args.token_emb, 1), nn.Sigmoid()) for _ in range(64)]
        self.prob_pred = nn.Sequential(nn.Linear(self.args.token_emb, self.args.token_emb), nn.ReLU(), nn.Linear(self.args.token_emb, 1), nn.ReLU())
        
    def forward(self, g, subgraph):
        
        # Tokenizer
        hs, hf = self.tokenizer(g)
        
        # Transformer 
        # tf_hs, tf_hf = hs, hf
        tf_hs, tf_hf = self.transformer(hs, hf, subgraph)
        
        # Pooling 
        hop_hs = torch.zeros(len(g.gate), self.args.token_emb).to(self.args.device)
        hop_hf = torch.zeros(len(g.gate), self.args.token_emb).to(self.args.device)
        for idx in subgraph.keys():
            hop_hs[idx] = self.hs_pool(torch.cat([tf_hs[subgraph[idx]['pos'].long()], tf_hs[subgraph[idx]['pis'].long()]], dim=0))
            hop_hf[idx] = self.hf_pool(torch.cat([tf_hf[subgraph[idx]['pos'].long()], tf_hf[subgraph[idx]['pis'].long()]], dim=0))
        
        return tf_hs, tf_hf, hop_hs, hop_hf
    
    def pred_tt(self, graph_emb, no_pi):
        tt = []
        for pi in range(int(pow(2, no_pi))):
            tt.append(self.tt_pred[pi](graph_emb))
        tt = torch.tensor(tt).squeeze()
        return tt
    
    def pred_prob(self, hf):
        prob = self.prob_pred(hf)
        return prob