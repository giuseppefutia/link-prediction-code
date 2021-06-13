import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# GCN Message Passing

def gcn_msg(edge):
    msg = edge.src['h'] * edge.src['norm']
    return {'m': msg}

def gcn_reduce(node):
    accum = torch.sum(node.mailbox['m'], 1) * node.data['norm']
    return {'h': accum}

# GAT Message Passing

def gat_msg(edge):
    src = edge.src['h']
    att = edge.data['e']
    return {'m': src, 'e': att}

def gat_reduce(node):
    alpha = F.softmax(node.mailbox['e'], dim=1)
    acc = torch.sum(alpha * node.mailbox['m'], dim=1)
    return {'h': acc}


class NodeApplyModule(nn.Module):
    # Update node features with linear transformation
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)
        
    def forward(self, node):
        h = self.lin(node.data['h'])
        return {'h': h}


class GCNLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 activation,
                 dropout):
        
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        
        self.activation = activation
        self.node_update = NodeApplyModule(in_dim, out_dim)
    
    def forward(self, g, features):
        g.ndata['h'] = features
        g.apply_nodes(self.node_update)
        g.update_all(gcn_msg, gcn_reduce)
        h = g.ndata['h']
        h = self.activation(h)
        h = self.dropout(h)
        return h

    def __repr__(self):
        return '{}(in_dim={}, out_dim={})'.format(self.__class__.__name__,
                                            self.in_dim,
                                            self.out_dim)


class GATLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 activation,
                 dropout):
        
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        
        self.activation = activation
        self.node_update = NodeApplyModule(in_dim, out_dim)

    def edge_attention(self, edge):
        z2 = torch.cat([edge.src['h'], edge.dst['h']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}
    
    def forward(self, g, features):
        g.ndata['h'] = features
        g.apply_edges(self.edge_attention)
        g.update_all(gat_msg, gat_reduce)
       
        h = g.ndata['h']
        h = self.activation(h) # F.elu as default
        h = self.dropout(h)
        return h
     
    def __repr__(self):
        return '{}(in_dim={}, out_dim={})'.format(self.__class__.__name__,
                                                  self.in_dim,
                                                  self.out_dim)


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x):
        y = self.lin(x)
        return y


class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, activation, dropout, out_dim, gnn_model):
        super().__init__()
        self.input = nn.Embedding(in_dim, hidden_dim)
        self.gnn_model = gnn_model
        if self.gnn_model == 'GCN':
            self.gnn1 = GCNLayer(hidden_dim, hidden_dim, activation, dropout)
            self.gnn2 = GCNLayer(hidden_dim, hidden_dim, activation, dropout)
        else: 
            self.gnn = GATLayer(hidden_dim, hidden_dim, F.elu, dropout)
        self.output = MLPLayer(hidden_dim, out_dim)
    
    def forward(self, g, in_feat):
        h = self.input(in_feat)
        if self.gnn_model == 'GCN':
            h = self.gnn1(g, h)
            h = self.gnn2(g, h)
        else:
            h = self.gnn(g, h)
        return h
    
    def edge_predictor(self, h_i, h_j):
        x = torch.cat([h_i, h_j], dim=1)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x
    
    def loss(self, pos_out, neg_out):
        pos_loss = -torch.log(pos_out + 1e-15).mean()  # positive samples
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()  # negative samples
        loss = pos_loss + neg_loss
        return loss