import argparse

from loader import BioDataset
from dataloader import DataLoaderMasking 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred, GNNDecoder,MLP

import pandas as pd

from util import MaskEdge

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

#criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

def compute_accuracy(pred, target):
    #return float(torch.sum((pred.detach() > 0) == target.to(torch.uint8)).cpu().item())/(pred.shape[0]*pred.shape[1])
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def train(args, model_list, loader, optimizer_list, device):
    model, linear_pred_edges = model_list
    optimizer_model, optimizer_linear_pred_edges = optimizer_list

    model.train()
    linear_pred_edges.train()

    loss_accum = 0
    acc_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)

        ### predict the edge types.
        masked_edge_index = batch.edge_index[:, batch.masked_edge_idx]
        edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
        pred_edge = linear_pred_edges(edge_rep)

        #converting the binary classification to multiclass classification
        edge_label = torch.argmax(batch.mask_edge_label, dim = 1)

        acc_edge = compute_accuracy(pred_edge, edge_label)
        acc_accum += acc_edge

        optimizer_model.zero_grad()
        optimizer_linear_pred_edges.zero_grad()

        loss = criterion(pred_edge, edge_label)
        loss.backward()

        optimizer_model.step()
        optimizer_linear_pred_edges.step()

        loss_accum += float(loss.cpu().item())

    return loss_accum/(step + 1), acc_accum/(step + 1)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.15,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--model_file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f" %(args.num_layer, args.mask_rate))

    #set up dataset
    root_unsupervised = 'dataset/unsupervised'
    dataset = BioDataset(root_unsupervised, data_type='unsupervised', transform = MaskEdge(mask_rate = args.mask_rate))

    print(dataset)

    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device)

    if args.input_model_file is not None and args.input_model_file != "":
        model.load_state_dict(torch.load(args.input_model_file))
        print("Resume training from:", args.input_model_file)
        resume = True
    else:
        resume = False

    #set up models, one for pre-training and one for context embeddings
    MLP1=MLP(args.emb_dim).to(device)
    MLP2=MLP(args.emb_dim).to(device)
    atom_pred_decoder = GNNDecoder(args.emb_dim, args.bond_dim, JK=args.JK, gnn_type=args.gnn_type).to(device)
    if args.mask_edge:
        bond_pred_decoder = GNNDecoder(args.emb_dim, args.bond_dim, JK=args.JK, gnn_type=args.gnn_type)
        optimizer_dec_pred_bonds = optim.Adam(bond_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    else:
        bond_pred_decoder = None
        optimizer_dec_pred_bonds = None

    model_list = [model,MLP1,MLP2,atom_pred_decoder, bond_pred_decoder] 

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_atoms = optim.Adam(atom_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_MLP1=optim.Adam(MLP1.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_MLP2=optim.Adam(MLP2.parameters(), lr=args.lr, weight_decay=args.decay)
    if args.use_scheduler:
        print("--------- Use scheduler -----------")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.epochs) ) * 0.5
        scheduler_model = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=scheduler)
        scheduler_dec_pred_atoms = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_atoms, lr_lambda=scheduler)
        scheduler_dec_pred_bonds = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_bonds, lr_lambda=scheduler)
        scheduler_MLP1= torch.optim.lr_scheduler.LambdaLR(optimizer_MLP1, lr_lambda=scheduler)
        scheduler_MLP2= torch.optim.lr_scheduler.LambdaLR(optimizer_MLP2, lr_lambda=scheduler)
        scheduler_list = [scheduler_model, scheduler_MLP1,scheduler_MLP2,scheduler_dec_pred_atoms,scheduler_dec_pred_bonds, None]
    else:
        scheduler_model = None
        scheduler_MLP1=None
        scheduler_MLP2=None
        scheduler_dec_pred_atoms  = None
        scheduler_dec_pred_bonds  = None

    optimizer_list = [optimizer_model, optimizer_MLP1,optimizer_MLP2,optimizer_dec_pred_atoms, optimizer_dec_pred_bonds]

    output_file_temp = "/Bio/checkpoints/" + args.output_model_file + f"_{args.gnn_type}"
        

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        # train_loss, train_acc_atom, train_acc_bond = train(args, model_list, loader, optimizer_list, device)
        # print(train_loss, train_acc_atom, train_acc_bond)

        train_loss = train(args, model_list, loader, optimizer_list, device, alpha_l=args.alpha_l, loss_fn=args.loss_fn)
        if not resume:
            if epoch % 50 == 0:
                torch.save(model.state_dict(), output_file_temp + f"_{epoch}.pth")
        print(train_loss)
        if scheduler_model is not None:
            scheduler_model.step()
        if scheduler_MLP1 is not None:
            scheduler_MLP1.step()
        if scheduler_MLP2 is not None:
            scheduler_MLP2.step()
        if scheduler_dec_pred_atoms is not None:
            scheduler_dec_pred_atoms.step()
        if scheduler_dec_pred_bonds is not None:
            scheduler_dec_pred_bonds.step()

    output_file = "/Bio/checkpoints/" + args.output_model_file + f"_{args.gnn_type}"
    if resume:
        torch.save(model.state_dict(), args.input_model_file.rsplit(".", 1)[0] + f"_resume_{args.epochs}.pth")
    elif not args.output_model_file == "":
        torch.save(model.state_dict(), output_file + ".pth")

if __name__ == "__main__":
    main()