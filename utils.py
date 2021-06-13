import os
import numpy as np
import torch

from ogb.linkproppred import DglLinkPropPredDataset, Evaluator


def save_model(model):
    torch.save(model, 'model.pth')
    print('Model saved!')


def save_features(tensor):
    node_features = tensor.tolist()
    with open('features.txt', 'w') as f:
        for item in node_features:
            f.write("%s\n" % item)


def count_parameters(model):
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return learnable_params


def load_dataset(name='ogbl-ddi'):
    dataset = DglLinkPropPredDataset(name = name)
    graph = dataset[0]
    split_edge = dataset.get_edge_split()

    train_edges = split_edge['train']['edge']
    valid_edges, valid_edges_neg = split_edge['valid']['edge'], split_edge['valid']['edge_neg']
    test_edges, test_edges_neg = split_edge['test']['edge'], split_edge['test']['edge_neg']

    evaluator = Evaluator(name)

    src = torch.reshape(graph.edges()[0], (-1, 1))
    dst = torch.reshape(graph.edges()[1], (-1, 1))
    graph_edges = torch.cat((src, dst), 1)

    train_mask = edge_mask(graph_edges, train_edges)
    val_mask = edge_mask(graph_edges, valid_edges)
    test_mask = edge_mask(graph_edges, test_edges)
    graph.edata['train_mask'] = train_mask
    graph.edata['val_mask'] = val_mask
    graph.edata['test_mask'] = test_mask
    
    return graph, train_edges, valid_edges, valid_edges_neg, test_edges, test_edges_neg, evaluator


def edge_mask(edges, subset):
    """Create edge mask for different datasets."""
    dim = torch.max(torch.unique(edges)).item() + 1
    space = torch.zeros(dim, dim)

    edges = edges.t()
    subset = subset.t()

    space[subset[0], subset[1]] = 1
    int_mask = space[edges[:][0], edges[:][1]]
    mask = int_mask > 0

    return mask


def gpu_setup(use_gpu=False, gpu_id=0):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print('Is cuda available?')

    if torch.cuda.is_available() and use_gpu:
        print('Cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('Cuda not available')
        device = torch.device("cpu")
    
    return device