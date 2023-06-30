import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from loader import BioDataset
from dataloader import DataLoaderFinetune
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

class GINConv(MessagePassing):

    def __init__(self, emb_dim, aggr = "add", input_layer = False):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(2*emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))

        ### Mapping 0/1 edge features to embedding
        self.edge_encoder = torch.nn.Linear(9, emb_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 9)
        self_loop_attr[:,7] = 1 # attribute for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_encoder(edge_attr)

        if self.input_layer:
            x = self.input_node_embeddings(x.to(torch.int64).view(-1,))

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return torch.cat([x_j, edge_attr], dim = 1)

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr = "add", input_layer = False):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

        ### Mapping 0/1 edge features to embedding
        self.edge_encoder = torch.nn.Linear(9, emb_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 9)
        self_loop_attr[:,7] = 1 # attribute for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_encoder(edge_attr)

        if self.input_layer:
            x = self.input_node_embeddings(x.to(torch.int64).view(-1,))

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr = "add", input_layer = False):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        ### Mapping 0/1 edge features to embedding
        self.edge_encoder = torch.nn.Linear(9, heads * emb_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 9)
        self_loop_attr[:,7] = 1 # attribute for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_encoder(edge_attr)

        if self.input_layer:
            x = self.input_node_embeddings(x.to(torch.int64).view(-1,))

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean", input_layer = False):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        
        ### Mapping 0/1 edge features to embedding
        self.edge_encoder = torch.nn.Linear(9, emb_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 9)
        self_loop_attr[:,7] = 1 # attribute for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_encoder(edge_attr)

        if self.input_layer:
            x = self.input_node_embeddings(x.to(torch.int64).view(-1,))

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)


class GNN(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gat, graphsage, gcn
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of message-passing GNN convs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if layer == 0:
                input_layer = True
            else:
                input_layer = False

            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add", input_layer = input_layer))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, input_layer = input_layer))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim, input_layer = input_layer))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim, input_layer = input_layer))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, x, edge_index, edge_attr):
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            if layer == self.num_layer - 1:
                #remove relu from the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim = 0), dim = 0)[0]

        return node_representation


class GNN_graphpred(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_representation = self.gnn(x, edge_index, edge_attr)

        pooled = self.pool(node_representation, batch)
        center_node_rep = node_representation[data.center_node_idx]

        graph_rep = torch.cat([pooled, center_node_rep], dim = 1)

        return self.graph_pred_linear(graph_rep)


if __name__ == "__main__":
    pass



class GNNDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super().__init__()
        self._dec_type = gnn_type 
        if gnn_type == "gin":
            self.conv = GINConv(hidden_dim, out_dim, aggr = "add")
        elif gnn_type == "gcn":
            self.conv = GCNConv(hidden_dim, out_dim, aggr = "add")
        elif gnn_type == "linear":
            self.dec = torch.nn.Linear(hidden_dim, out_dim)
        else:
            raise NotImplementedError(f"{gnn_type}")
        self.dec_token = torch.nn.Parameter(torch.zeros([1, hidden_dim]))
        self.enc_to_dec = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)    
        self.activation = torch.nn.PReLU() 
        self.temp = 0.2


    def forward(self, x, edge_index, edge_attr, mask_node_indices):
        if self._dec_type == "linear":
            out = self.dec(x)
        else:
            x = self.activation(x)
            x = self.enc_to_dec(x)
            x[mask_node_indices] = 0
            # x[mask_node_indices] = self.dec_token
            out = self.conv(x, edge_index, edge_attr)
            # out = F.softmax(out, dim=-1) / self.temp
        return out

class GNNDecoder_new(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super().__init__()
        self._dec_type = gnn_type 
        if gnn_type == "gin":
            self.conv = GINConv(hidden_dim, out_dim, aggr = "add")
        elif gnn_type == "gcn":
            self.conv = GCNConv(hidden_dim, out_dim, aggr = "add")
        elif gnn_type == "linear":
            self.dec = torch.nn.Linear(hidden_dim, out_dim)
        else:
            raise NotImplementedError(f"{gnn_type}")
        self.dec_token = torch.nn.Parameter(torch.zeros([1, hidden_dim]))
        self.enc_to_dec = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)    
        self.activation = torch.nn.PReLU() 
        self.temp = 0.2
        self.mean_conv = GCNConv(hidden_dim, hidden_dim, aggr = "add")
        self.log_conv = GCNConv(hidden_dim, hidden_dim, aggr = "add") 


    def forward(self, x, edge_index, edge_attr,hidden_dim):
        x = self.activation(x, edge_index, edge_attr)
        mean = self.mean_conv(x, edge_index, edge_attr)
        log = self.log_conv(x, edge_index, edge_attr)
        gaussian_noise = torch.randn(edge_index[0].size(0), hidden_dim)
        z= gaussian_noise*torch.exp(self.logstd) + self.mean
        # x[mask_node_indices] = self.dec_token
        out = self.conv(x, edge_index, edge_attr)

        # out = F.softmax(out, dim=-1) / self.temp
        return out

class MLP(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()   
        self.activation = torch.nn.PReLU() 
        self.mlp_linear1 =torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp_linear2 =torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):

        x = self.activation(x)
        x = self.mlp_linear1(x)
        Z = self.mlp_linear2(x)
        return Z


    def __init__(self, num_layer,num_atom,num_tag, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        '''
        def encoding_mask_noise(self, g, x, mask_rate=0.3):
            return mask_index 
        '''

    #def forward(self, x, edge_index, edge_attr):
    def mask_noise(self, x, edge_index, edge_attr,mask_rate=0.2):
        num_nodes = len(x)
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        # random masking
        mask_index = perm[: num_mask_nodes]
        keep_index = perm[num_mask_nodes: ]
        x[mask_index]=0.0
        for node in edge_index[0],edge_index[1]:
            if node in mask_index:
                node=0
        noise_x=x
        noise_edge_index=edge_index


        return noise_x,noise_edge_index,mask_index

    #def make_noise(self,):;
    def forward(self, x, edge_index, edge_attr):
        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation;




class GNN_graphpred(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)
        #out = F.softmax(self.graph_pred_linear(self.pool(node_representation, batch)), dim=-1)
        out=self.graph_pred_linear(self.pool(node_representation, batch))
        return out

class GNN_graphpred_ib(torch.nn.Module):
    def __init__(self,num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred_ib, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        self.mlp=torch.nn.Linear(emb_dim,emb_dim)
        self.conv_mean = GINConv(emb_dim, emb_dim, aggr = "add")
        self.conv_logstddev = GINConv(emb_dim, emb_dim, aggr = "add")
        self.mlp1=torch.nn.Linear(emb_dim,emb_dim)
        self.mlp2=torch.nn.Linear(emb_dim,emb_dim)
        self.activation= torch.nn.PReLU() 
        self.mu_p = torch.nn.Parameter(torch.randn(emb_dim))
        self.std_p = torch.nn.Parameter(torch.randn(emb_dim))
     

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def kl_div(self, mu_q, std_q, mu_p, std_p):

        k = mu_q.size(1)
        #k=3000
        #print('mu_p.size(1)',mu_p.size(1))
        mu_diff = mu_p - mu_q
        mu_diff_sq = torch.mul(mu_diff, mu_diff)
        logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=1)
        logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=1)
        fs = torch.sum(torch.div(std_q ** 2, std_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=1)
        kl_divergence = (fs - k + logdet_std_p - logdet_std_q)*0.5
        return kl_divergence.mean()

    def forward(self, *argv):
        
        x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        node_representation = self.gnn(x, edge_index, edge_attr)
        #node_representation = self.mlp(node_representation)
        batch_size = node_representation.shape[0]
        #mean= self.conv_mean(node_representation,edge_index, edge_attr)
        #logstd= self.conv_logstddev(node_representation,edge_index, edge_attr)
        mean=self.conv_mean(node_representation,edge_index, edge_attr)
        logstd=self.conv_logstddev(node_representation,edge_index, edge_attr)
        gaussian_noise = torch.randn(node_representation.size(0), self.emb_dim).to(node_representation.device)

        z= gaussian_noise*torch.exp(logstd).to(node_representation.device) +mean
        mu_p = self.mu_p.view(1, -1).expand(batch_size, -1)
        std_p = torch.nn.functional.softplus(self.std_p.view(1, -1).expand(batch_size, -1))


        kl_loss = self.kl_div(mean, logstd, mu_p, std_p)
        return self.graph_pred_linear(self.pool(z, batch)),kl_loss

class Accumulator:  #@save
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class finetune_ib(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, ib_dim,ib,
    num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(finetune_ib, self).__init__()
        self.num_tasks = num_tasks
        self.ib_dim =ib_dim
        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        
        self.dropout = torch.nn.Dropout(drop_ratio)
        self.ib = ib
        self.JK = JK
        self.activation= torch.nn.PReLU()
        print('self.JK',self.JK)

        if self.ib :
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(emb_dim,emb_dim),
                self.activation,
                torch.nn.Linear(emb_dim,emb_dim),
                self.activation)


            self.emb2mu = torch.nn.Linear(emb_dim,self.ib_dim)
            self.emb2std = torch.nn.Linear(emb_dim,self.ib_dim)
            self.mu_p = torch.nn.Parameter(torch.randn(self.ib_dim))
            self.std_p = torch.nn.Parameter(torch.randn(self.ib_dim))
            self.pool_dim=1
        else:
            self.pool_dim=emb_dim
        self.compress=torch.nn.Linear(self.ib_dim,self.pool_dim)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * self.pool_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(self.pool_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * self.pool_dim, set2set_iter)
            else:
                self.pool = Set2Set(1, set2set_iter)
        else:   
            raise ValueError("Invalid graph pooling type.")
        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.pool_dim, self.num_tasks)
            
        else:
            print('self.mult * self.pool_dim',self.mult * self.pool_dim)
            self.graph_pred_linear = torch.nn.Linear(self.mult *self.pool_dim, self.num_tasks)
            

        
    def estimate(self, emb, emb2mu, emb2std):
        """Estimates mu and std from the given input embeddings."""
        mean = emb2mu(emb)
        std = torch.nn.functional.softplus(emb2std(emb))
        return mean, std

    def kl_div(self, mu_q, std_q, mu_p, std_p):
        """Computes the KL divergence between the two given variational distribution.\
           This computes KL(q||p), which is not symmetric. It quantifies how far is\
           The estimated distribution q from the true distribution of p."""
        k = mu_q.size(1)
        mu_diff = mu_p - mu_q
        mu_diff_sq = torch.mul(mu_diff, mu_diff)
        logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=1)
        logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=1)
        fs = torch.sum(torch.div(std_q ** 2, std_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=1)
        kl_divergence = (fs - k + logdet_std_p - logdet_std_q)*0.5
        return kl_divergence.mean()

    def reparameterize(self, mu, std):
        z = torch.randn(mu.shape[0], mu.shape[1]).to(mu.device)
        return mu + std * z
    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))


    '''def get_logits(self, z, mu, sampling_type):
        if sampling_type == "iid":
            logits = self.classifier(z)
            mean_logits = logits.mean(dim=0)
            logits = logits.permute(1, 2, 0)
        else:
            mean_logits = self.classifier(mu)
            logits = mean_logits
        return logits, mean_logits
'''
    def forward(self, batch):
        
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        node_representation = self.gnn(x, edge_index, edge_attr)
        pooled_output=self.pool(node_representation, batch.batch)
        if self.ib:
            pooled_output = self.mlp(pooled_output)
            batch_size = pooled_output.shape[0]
            mu, std = self.estimate(pooled_output, self.emb2mu, self.emb2std)
            mu_p = self.mu_p.view(1, -1).expand(batch_size, -1)
            std_p = torch.nn.functional.softplus(self.std_p.view(1, -1).expand(batch_size, -1))
            kl_loss = self.kl_div(mu, std, mu_p, std_p)
            z = self.reparameterize(mu, std)
            z=self.compress(z)
            out=self.graph_pred_linear(z)  
        else:
          out=self.graph_pred_linear(self.pool(node_representation, batch.batch))  
          kl_loss=0     
        return out,kl_loss


class enc_dec_sample(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv_mean = GCNConv(hidden_dim, hidden_dim, aggr = "add")
        self.conv_logstddev = GCNConv(hidden_dim, hidden_dim, aggr = "add")
        self.activation= torch.nn.PReLU() 

    def forward(self, x, edge_index, edge_attr):
        hidden = self.activation(x)
        self.mean= self.conv_mean(hidden)
        self.logstd= self.conv_logstddev(hidden)
        gaussian_noise = torch.randn(x.size(0), self.hidden_dim)
        sampled_z= gaussian_noise*torch.exp(self.logstd) + self.mean
        return sampled_z
class cat_linear(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.cat_linear = torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim)
    def forward(self, x_edge_mask, x_attr_mask):
        x_mask = torch.cat((x_edge_mask, x_attr_mask),0)
        z_mask=self.cat_linear(x_mask)
        return z_mask
    