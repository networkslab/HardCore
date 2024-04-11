import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn.pytorch as dglnn

class SATInstanceEncoderHetero(nn.Module):
    def __init__(self, hidden_dim, num_layers, feat_type, add_clause_pe=False, flip_literal_flag=False, device=torch.device('cpu'), act='stack' ):
        super().__init__()
        self.act = act
        if feat_type == 'pe':
            in_dim = 10 
        else:
            in_dim = 3
        self.device = device
        layers = []
        layers.append(dglnn.HeteroGraphConv({
            'in': dglnn.GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True),
            'contain': dglnn.GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True),
            'flip': dglnn.GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)},
            aggregate=self.act)
        )
        # layers.append(dglnn.HeteroGraphConv({
        #     'in': dglnn.GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True),
        #     'pair': dglnn.GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)},
        #     aggregate='mean')
        # )
        for i in range(num_layers-1):
            layers.append(dglnn.HeteroGraphConv({
                'in': dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True),
                'contain': dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True),
                'flip': dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)},
                aggregate=self.act)
            )
        # for i in range(num_layers-1):
        #     layers.append(dglnn.HeteroGraphConv({
        #         'in': dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True),
        #         'pair': dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)},
        #         aggregate='mean')
        #     )
        self.layers = nn.Sequential(*layers)
        if self.act == 'stack':
            self.fc_out = nn.Linear(hidden_dim*(2**num_layers), 1)    # Project concat(literal, clause) to hidden_dim
        elif self.act == 'mean':
            self.fc_out = nn.Linear(hidden_dim, 1)   
        self.flip_literal_flag = flip_literal_flag

    def forward(self, g, h):
       
        for layer in self.layers:
            # print(0, list(h['pos_lit']))
            h = layer(g, h)
            # print(1, (h['clause'].shape))
            # input()
            h = {k: F.relu(v) for k, v in h.items()}
            # print(2, h['clause'].shape)
            # print('--------------------------')
            if self.flip_literal_flag:
                tmp = h['pos_lit']
                h['pos_lit'] = h['neg_lit']
                h['neg_lit'] = tmp
        # with g.local_scope():
        #     g.ndata['h'] = h
        #     # Calculate graph representation by average readout.
        #     pos_lit_pooled = dgl.mean_nodes(g, 'h', ntype='pos_lit')
        #     neg_lit_pooled = dgl.mean_nodes(g, 'h', ntype='neg_lit')
        #     clause_pooled = dgl.mean_nodes(g, 'h', ntype='clause')
        # literal_pooled = torch.stack([pos_lit_pooled, neg_lit_pooled], dim=0).mean(dim=0)
        # literal_clause = torch.concat([literal_pooled, clause_pooled], dim=-1)
        # print(h['clause'].shape)
        h = h['clause']
        if self.act == 'sack':
            h = torch.reshape(h,(h.shape[0], 2**(len(self.layers))*h.shape[-1]))
        hg = self.fc_out(h)
        # print(hg.shape)
        return torch.sigmoid(hg)

class SATInstanceEncoderHetero_softmax(nn.Module):
    def __init__(self, hidden_dim, num_layers, feat_type, add_clause_pe=False, flip_literal_flag=False):
        super().__init__()
        if feat_type == 'pe':
            in_dim = 10 
        else:
            in_dim = 3
        layers = []
        layers.append(dglnn.HeteroGraphConv({
            'in': dglnn.GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True),
            'contain': dglnn.GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True),
            'flip': dglnn.GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)},
            aggregate='stack')
        )
        # layers.append(dglnn.HeteroGraphConv({
        #     'in': dglnn.GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True),
        #     'pair': dglnn.GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)},
        #     aggregate='mean')
        # )
        for i in range(num_layers-1):
            layers.append(dglnn.HeteroGraphConv({
                'in': dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True),
                'contain': dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True),
                'flip': dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)},
                aggregate='stack')
            )
        # for i in range(num_layers-1):
        #     layers.append(dglnn.HeteroGraphConv({
        #         'in': dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True),
        #         'pair': dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)},
        #         aggregate='mean')
        #     )
        self.layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim*(2**num_layers), 2)    # Project concat(literal, clause) to hidden_dim
        self.flip_literal_flag = flip_literal_flag

    def forward(self, g, h):
        for layer in self.layers:
            # print(0, list(h['pos_lit']))
            h = layer(g, h)
            # print(1, (h['clause'].shape))
            # input()
            h = {k: F.relu(v) for k, v in h.items()}
            # print(2, h['clause'].shape)
            # print('--------------------------')
            if self.flip_literal_flag:
                tmp = h['pos_lit']
                h['pos_lit'] = h['neg_lit']
                h['neg_lit'] = tmp
        # with g.local_scope():
        #     g.ndata['h'] = h
        #     # Calculate graph representation by average readout.
        #     pos_lit_pooled = dgl.mean_nodes(g, 'h', ntype='pos_lit')
        #     neg_lit_pooled = dgl.mean_nodes(g, 'h', ntype='neg_lit')
        #     clause_pooled = dgl.mean_nodes(g, 'h', ntype='clause')
        # literal_pooled = torch.stack([pos_lit_pooled, neg_lit_pooled], dim=0).mean(dim=0)
        # literal_clause = torch.concat([literal_pooled, clause_pooled], dim=-1)
        # print(h['clause'].shape)
        h = h['clause']
        h = torch.reshape(h,(h.shape[0], 2**(len(self.layers))*h.shape[-1]))
        hg = self.fc_out(h)
        # print(hg.shape)
        return torch.nn.functional.softmax(hg)

