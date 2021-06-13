import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dgl.data import register_data_args
from dgl.sampling import sample_neighbors
from dgl.subgraph import edge_subgraph

from net import Model
from utils import load_dataset, gpu_setup, save_model, save_features, count_parameters

def train_single_epoch(g, dataset, in_feat, model, optimizer, device, num_node_samples):
    model.train()
    total_loss = total_examples = 0
    train_mask = g.edata['train_mask']
    sub_g = edge_subgraph(g, train_mask)
    print('\n----- Training Graph:')
    print(sub_g)
    print()

    # Edges sampling for scalability
    print('Perform edge sampling...')
    perm = torch.randperm(g.nodes().size(0))
    idx = perm[:num_node_samples]
    spl_g = sample_neighbors(sub_g, idx, 100)
    src = torch.reshape(spl_g.edges()[0], (-1, 1))
    dst = torch.reshape(spl_g.edges()[1], (-1, 1))
    sampled_edges = torch.cat((src, dst),1).to(device)

    print('Number of sampled nodes: %d' % sub_g.num_nodes())
    print('Number of training edges: %d' % sub_g.num_edges())
    print('Number of sampled edges: %d' % spl_g.num_edges())

    for batch in DataLoader(dataset=sampled_edges, batch_size=1000, shuffle=True):
        optimizer.zero_grad()
        g = g.to(device)
        h = model(g, in_feat)

        # Positive samples
        edge = batch.t()
        pos_out = model.edge_predictor(h[edge[0]], h[edge[1]])

        # Negative samples
        edge = torch.randint(0, in_feat.size(0), edge.size(), dtype=torch.long)
        neg_out = model.edge_predictor(h[edge[0]], h[edge[1]])
        
        loss = model.loss(pos_out, neg_out)
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.detach().item() * num_examples
        total_examples += num_examples
        
    return total_loss/total_examples, optimizer, loss


def eval(g, dataset, evaluator, in_feat, model, device, batch_size=100):
    model.eval()

    with torch.no_grad():
        g = g.to(device)

        h = model(g, in_feat)

        pos_train_edges = dataset['train'].to(device)
        pos_valid_edges = dataset['valid'].to(device)
        neg_valid_edges = dataset['valid_neg'].to(device)
        pos_test_edges = dataset['test'].to(device)
        neg_test_edges = dataset['test_neg'].to(device)

        pos_train_preds = []
        for batch in DataLoader(range(pos_train_edges.size(0)), batch_size):
            edge = pos_train_edges[batch].t()
            pos_train_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_train_pred = torch.cat(pos_train_preds, dim=0)

        pos_valid_preds = []
        for batch in DataLoader(range(pos_valid_edges.size(0)), batch_size):
            edge = pos_valid_edges[batch].t()
            pos_valid_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

        neg_valid_preds = []
        for batch in DataLoader(range(neg_valid_edges.size(0)), batch_size):
            edge = neg_valid_edges[batch].t()
            neg_valid_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

        pos_test_preds = []
        for batch in DataLoader(range(pos_test_edges.size(0)), batch_size):
            edge = pos_test_edges[batch].t()
            pos_test_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for batch in DataLoader(range(neg_test_edges.size(0)), batch_size):
            edge = neg_test_edges[batch].t()
            neg_test_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)
    
    train_hits = []
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits.append(
            evaluator.eval({
                'y_pred_pos': pos_train_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
        )
    
    valid_hits = []
    for K in [10, 50, 100]:
        evaluator.K = K
        valid_hits.append(
            evaluator.eval({
                'y_pred_pos': pos_valid_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
        )
    
    test_hits = []
    for K in [10, 50, 100]:
        evaluator.K = K
        test_hits.append(
            evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']
        )

    return train_hits, valid_hits, test_hits


def train_process(g, dataset, evaluator, in_feat, model, optimizer, epochs, device, num_node_samples):
    t0 = time.time()
    per_epoch_time = []

    print('\nStart the training process!')

    print("\nNumber of training edges: %d" % dataset['train'].size()[0])
    print("Number of valid edges: %d" % dataset['valid'].size()[0])
    print("Number of test edges: %d" % dataset['test'].size()[0])
    
    for epoch in range(epochs):
        
        start = time.time()  
        
        # training
        epoch_train_loss, optimizer, loss = train_single_epoch(
                    g, dataset, in_feat, model, optimizer, device, num_node_samples)
        
        # evaluation
        '''
        epoch_train_hits, epoch_val_hits, epoch_test_hits = eval(
                    g, dataset, evaluator, in_feat, model, device)
        '''

        per_epoch_time.append(time.time()-start)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
                .format(epoch, time.time()-start, epoch_train_loss))
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'checkpoint.pth')
    
    train_hits, val_hits, test_hits = eval(
                    g, dataset, evaluator, in_feat, model, device)

    print("\n\nNumber of training edges: %d" % dataset['train'].size()[0])
    print("Number of valid edges: %d" % dataset['valid'].size()[0])
    print("Number of test edges: %d" % dataset['test'].size()[0])
    
    print(f"\n\nTest:\nHits@10: {test_hits[0]*100:.4f}% \nHits@50: {test_hits[1]*100:.4f}% \nHits@100: {test_hits[2]*100:.4f}% \n")
    print(f"Train:\nHits@10: {train_hits[0]*100:.4f}% \nHits@50: {train_hits[1]*100:.4f}% \nHits@100: {train_hits[2]*100:.4f}% \n")
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    # save features and model
    save_features(model.input.weight.data[:10, :]) # First ten node features
    save_model(model)
        
        
def main(args):
    g, train, valid, valid_neg, test, test_neg, evaluator = load_dataset()

    print('\n----- Full Graph:')
    print(g)
    print()

    dataset = {'train': train,
               'valid': valid,
               'valid_neg': valid_neg,
               'test': test,
               'test_neg': test_neg}

    torch.cuda.is_available() 
    device = gpu_setup(use_gpu=True, gpu_id=args.gpu)
    if device == 'cuda':
        g.to(device)
    
    # degree normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    norm.to(device)
    g.ndata['norm'] = norm.unsqueeze(1)
    
    in_dim = g.number_of_nodes()
    hidden_dim = args.hidden_dim
    activation = F.relu
    dropout = args.dropout
    out_dim = 1
    gnn_model = args.gnn_model

    in_feat = torch.arange(in_dim).to(device)

    model = Model(in_dim,
                  hidden_dim,
                  activation,
                  dropout,
                  out_dim,
                  gnn_model)
    
    model = model.to(device)

    print('\n----- Model:')
    print(model)
    
    learnable = count_parameters(model)
    print('Learnable parameters: %d' % learnable)    

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    
    epochs = args.n_epochs
    num_node_samples = args.n_samples

    train_process(g, dataset, evaluator, in_feat, model, optimizer, epochs, device, num_node_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN')
    register_data_args(parser)

    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    
    parser.add_argument("--gpu", type=int, default=1,
            help="gpu")
    
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    
    parser.add_argument("--n-epochs", type=int, default=50,
            help="number of training epochs")
    
    parser.add_argument("--hidden-dim", type=int, default=16,
            help="number of hidden gcn units")

    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    
    parser.add_argument("--n-samples", type=int, default=500,
            help="Number of nodes to sample (efficiency purposes)")
    
    parser.add_argument("--gnn-model", type=str, default="GCN",
            help="Graph Neural Network model")
    
    args = parser.parse_args()
    
    print(args)

    main(args)