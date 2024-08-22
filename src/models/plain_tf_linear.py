import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, FusedGATConv
from torch.nn import Linear, LayerNorm
from torch_geometric.utils import add_self_loops

# class GATTransformerEncoderLayer(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, heads=8, concat=True, dropout=0.1, ff_hidden_dim=128):
#         super(GATTransformerEncoderLayer, self).__init__()
        
#         # GAT multi-head attention
#         self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=concat)
        
#         # Feed-forward network (FFN)
#         self.ffn = torch.nn.Sequential(
#             Linear(out_channels*heads if concat else out_channels, ff_hidden_dim),
#             torch.nn.ReLU(),
#             Linear(ff_hidden_dim, out_channels*heads if concat else out_channels)
#         )
        
#         # Layer normalization
#         self.norm1 = LayerNorm(out_channels*heads if concat else out_channels)
#         self.norm2 = LayerNorm(out_channels*heads if concat else out_channels)
        
#         # Dropout
#         self.dropout = torch.nn.Dropout(dropout)

#     def forward(self, x, edge_index):
#         # GAT layer with residual connection
#         x_residual = x.clone()
#         x = self.gat(x, edge_index)
#         x = self.dropout(x)
#         x = x + x_residual  # Residual connection
#         x = self.norm1(x)   # Layer normalization
        
#         # Feed-forward network with residual connection
#         x_residual = x.clone()
#         x = self.ffn(x)
#         x = self.dropout(x)
#         x = x + x_residual  # Residual connection
#         x = self.norm2(x)   # Layer normalization
        
#         return x



# class Sparse_Transformer(torch.nn.Module):
#     def __init__(self, args, hidden, num_layers=4, heads=4, concat=True, dropout=0.):
#         super(Sparse_Transformer, self).__init__()
 
#         in_channels = hidden
#         out_channels = in_channels // heads

#         ff_hidden_dim = 4 * hidden

#         self.num_layers = num_layers
#         self.hf_layers = torch.nn.ModuleList([
#             GATTransformerEncoderLayer(in_channels if i == 0 else out_channels*heads if concat else out_channels,
#                                        out_channels, heads=heads, concat=concat, dropout=dropout, ff_hidden_dim=ff_hidden_dim)
#             for i in range(num_layers)
#         ])
#         self.hs_layers = torch.nn.ModuleList([
#             GATTransformerEncoderLayer(in_channels if i == 0 else out_channels*heads if concat else out_channels,
#                                        out_channels, heads=heads, concat=concat, dropout=dropout, ff_hidden_dim=ff_hidden_dim)
#             for i in range(num_layers)
#         ])

#     def forward(self, g, hf, hs):

#         bs = g.batch.max()+1
#         prefix_idx = []
#         for i in range(bs):
#             prefix_idx.append(torch.sum(g.batch<i))
#         prefix_idx = torch.stack(prefix_idx)

#         # corr_m = g.fanin_fanout_cones.reshape(bs,511,511)
#         corr_m = g.fanin_fanout_cones.reshape(bs,512,512)
#         # print(corr_m)
#         corr_m = torch.where(corr_m == 1, False, True)
#         virtual_edge = torch.argwhere(corr_m == False) #(nodes, 3) 3 indices(bs, row, col)
#         # print("sparsity: ", virtual_edge.shape[0] / torch.numel(corr_m))
#         virtual_edge = virtual_edge[:,1:] + prefix_idx[virtual_edge[:,0]].unsqueeze(-1).repeat(1,2)
#         virtual_edge = torch.stack([virtual_edge[:,1],virtual_edge[:,0]],dim=-1).transpose(0,1)

#         # virtual_edge = virtual_edge.T
#         # virtual_edge = virtual_edge[mk[g.nodes[virtual_edge[:,1].cpu()]]==0]
#         # virtual_edge = virtual_edge.T

#         if virtual_edge.shape[1] == 0:
#             return hf, hs
        
#         for i in range(self.num_layers):
    
#             hf = self.hf_layers[i](hf+hs, virtual_edge)
#             hs = self.hs_layers[i](hs, virtual_edge)

#         return hf, hs

class GATTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, concat=True, dropout=0.1, ff_hidden_dim=128):
        super(GATTransformerEncoderLayer, self).__init__()
        
        # GAT multi-head attention
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=concat)
        # self.gat = FusedGATConv((in_channels, in_channels), out_channels, heads=heads, dropout=dropout, concat=concat, add_self_loops=False)
        
        # Feed-forward network (FFN)
        self.ffn = torch.nn.Sequential(
            Linear(out_channels*heads if concat else out_channels, ff_hidden_dim),
            torch.nn.ReLU(),
            Linear(ff_hidden_dim, out_channels*heads if concat else out_channels)
        )
        
        # Layer normalization
        self.norm1 = LayerNorm(out_channels*heads if concat else out_channels)
        self.norm2 = LayerNorm(out_channels*heads if concat else out_channels)
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # GAT layer with residual connection
        x_residual = x.clone()
        # (rowptr, col), (row, colptr) , perm = self.gat.to_graph_format(edge_index)
        # x = self.gat(x, (rowptr, col), (row, colptr) , perm)
        x = self.gat(x, edge_index)
        x = self.dropout(x)
        # print(x.shape)
        # print(x_residual.shape)
        x = x + x_residual  # Residual connection
        x = self.norm1(x)   # Layer normalization
        
        # Feed-forward network with residual connection
        x_residual = x.clone()
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_residual  # Residual connection
        x = self.norm2(x)   # Layer normalization
        
        return x



class Sparse_Transformer(torch.nn.Module):
    def __init__(self, args, hidden, num_layers=4, heads=4, concat=True, dropout=0.):
        super(Sparse_Transformer, self).__init__()
 
        in_channels = hidden
        out_channels = in_channels // heads

        ff_hidden_dim = 4 * hidden

        self.num_layers = num_layers
        self.hf_layers = torch.nn.ModuleList([
            GATTransformerEncoderLayer(in_channels if i == 0 else out_channels*heads if concat else out_channels,
                                       out_channels, heads=heads, concat=concat, dropout=dropout, ff_hidden_dim=ff_hidden_dim)
            for i in range(num_layers)
        ])
        self.hs_layers = torch.nn.ModuleList([
            GATTransformerEncoderLayer(in_channels if i == 0 else out_channels*heads if concat else out_channels,
                                       out_channels, heads=heads, concat=concat, dropout=dropout, ff_hidden_dim=ff_hidden_dim)
            for i in range(num_layers)
        ])

    def forward(self, g, hf, hs):

        bs = g.batch.max()+1
        prefix_idx = []
        for i in range(bs):
            prefix_idx.append(torch.sum(g.batch<i))
        prefix_idx = torch.stack(prefix_idx)

        # corr_m = g.fanin_fanout_cones.reshape(bs,511,511)
        corr_m = g.fanin_fanout_cones.reshape(bs,512,512)
        # print(corr_m)
        corr_m = torch.where(corr_m == 1, False, True)
        virtual_edge = torch.argwhere(corr_m == False) #(nodes, 3) 3 indices(bs, row, col)
        # print("sparsity: ", virtual_edge.shape[0] / torch.numel(corr_m))
        virtual_edge = virtual_edge[:,1:] + prefix_idx[virtual_edge[:,0]].unsqueeze(-1).repeat(1,2)
        virtual_edge = torch.stack([virtual_edge[:,1],virtual_edge[:,0]],dim=-1).transpose(0,1)
        virtual_edge, _ = add_self_loops(virtual_edge)
        # virtual_edge = virtual_edge.T
        # virtual_edge = virtual_edge[mk[g.nodes[virtual_edge[:,1].cpu()]]==0]
        # virtual_edge = virtual_edge.T

        if virtual_edge.shape[1] == 0:
            return hf, hs
        
        for i in range(self.num_layers):
    
            hf = self.hf_layers[i](hf+hs, virtual_edge)
            hs = self.hs_layers[i](hs, virtual_edge)

        return hf, hs
