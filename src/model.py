"""model.py"""

import torch
import torch.nn as nn
from torch_scatter import scatter_add

# Multi Layer Perceptron (MLP) class
class MLP(torch.nn.Module):
    def __init__(self, layer_vec, activation = nn.SiLU()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k+1])
            self.layers.append(layer)
            if k != len(layer_vec) - 2: self.layers.append(activation)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Edge model
class EdgeModel(torch.nn.Module):
    def __init__(self, args, dims, activation = nn.SiLU()):
        super(EdgeModel, self).__init__()
        self.n_hidden = args.n_hidden
        self.dim_hidden = args.dim_hidden
        self.edge_mlp = MLP([3*self.dim_hidden] + self.n_hidden*[self.dim_hidden] + [self.dim_hidden],activation=activation)

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        if u is not None:
            out = torch.cat([edge_attr, src, dest, u[batch]], dim=1)
        else: 
            out = torch.cat([edge_attr, src, dest], dim=1)
        out = self.edge_mlp(out)
        return out


# Node model
class NodeModel(torch.nn.Module):
    def __init__(self, args, dims,activation=nn.SiLU()):
        super(NodeModel, self).__init__()
        self.n_hidden = args.n_hidden
        self.dim_hidden = args.dim_hidden
        self.node_mlp = MLP([2*self.dim_hidden + dims['f']] + self.n_hidden*[self.dim_hidden] + [self.dim_hidden],activation=activation)

    def forward(self, x, edge_index, edge_attr, f=None, u=None, batch=None):
        src, dest = edge_index
        out = scatter_add(edge_attr, dest, dim=0, dim_size=x.size(0))
        if f is not None:
            out = torch.cat([x, out, f], dim=1)
        elif u is not None:
            out = torch.cat([x, out, u[batch]], dim=1)
        else:
            out = torch.cat([x, out], dim=1)
        out = self.node_mlp(out)
        return out


# Modification of the original MetaLayer class
class MetaLayer(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def forward(self, x, edge_index, edge_attr, f=None, u=None, batch=None):

        src = edge_index[0]
        dest = edge_index[1]

        edge_attr = self.edge_model(x[src], x[dest], edge_attr, u,
                                    batch if batch is None else batch[src])
        x = self.node_model(x, edge_index, edge_attr, f, u, batch)

        return x, edge_attr


# Thermodyncamics-informed Graph Neural Networks
class TIGNN(torch.nn.Module):
    def __init__(self, args, dims):
        super(TIGNN, self).__init__()
        # Arguments
        passes = args.passes
        n_hidden = args.n_hidden
        dim_hidden = args.dim_hidden
        self.dims = dims
        self.dim_z = self.dims['z']
        # self.dim_q = self.dims['q']
        # dim_node = self.dims['z'] + self.dims['n'] - self.dims['q']
        dim_node = self.dims['z'] + self.dims['n']
        dim_edge = self.dims['q_0'] + 1

        # Encoder MLPs
        self.encoder_node = MLP([dim_node] + n_hidden*[dim_hidden] + [dim_hidden])
        self.encoder_edge = MLP([dim_edge] + n_hidden*[dim_hidden] + [dim_hidden])
        # Processor MLPs
        self.processor = nn.ModuleList()
        for _ in range(passes):
            node_model = NodeModel(args, self.dims)
            edge_model = EdgeModel(args, self.dims)
            GraphNet = MetaLayer(node_model=node_model, edge_model=edge_model)
            self.processor.append(GraphNet)
        # Decoder MLPs
        self.decoder_E = MLP([dim_hidden] + n_hidden*[dim_hidden] + [1])
        self.decoder_S = MLP([dim_hidden] + n_hidden*[dim_hidden] + [1])
        self.decoder_L = MLP([dim_hidden] + n_hidden*[dim_hidden] + [int(self.dim_z*(self.dim_z+1)/2-self.dim_z)])
        self.decoder_M = MLP([dim_hidden] + n_hidden*[dim_hidden] + [int(self.dim_z*(self.dim_z+1)/2)])

        diag = torch.eye(self.dim_z, self.dim_z)
        self.diag = diag[None]
        self.ones = torch.ones(self.dim_z, self.dim_z)

    def forward(self, z, n, edge_index, q_0=None, f=None, g=None, batch=None): 
        '''Pre-process'''
        z.requires_grad = True
        # Node attributes 
        # Eulerian
        if q_0 is not None:
            q = q_0
            v = z
        # Lagrangian
        else:
            q = z[:,:self.dim_q]
            v = z[:,self.dim_q:]
        x = torch.cat((v,n), dim=1)
        # Edge attributes
        src, dest = edge_index
        u = q[src] - q[dest]
        u_norm = torch.norm(u,dim=1).reshape(-1,1)
        edge_attr = torch.cat((u,u_norm), dim=1)

        '''Encode'''
        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)

        '''Process'''
        for GraphNet in self.processor:
            x_res, edge_attr_res = GraphNet(x, edge_index, edge_attr, f=f, u=g, batch=batch)
            x += x_res
            edge_attr += edge_attr_res

        '''Decode'''
        # Gradients
        E = self.decoder_E(x)
        S = self.decoder_S(x)
        dEdz = torch.autograd.grad(E, z, torch.ones(E.shape, device=E.device,dtype=torch.int), create_graph=True)[0]
        dSdz = torch.autograd.grad(S, z, torch.ones(S.shape, device=S.device,dtype=torch.int), create_graph=True)[0]
        # GENERIC flattened matrices
        l = self.decoder_L(x)
        m = self.decoder_M(x)

        '''Reparametrization'''
        L = torch.zeros(x.size(0), self.dim_z, self.dim_z, device=l.device)
        M = torch.zeros(x.size(0), self.dim_z, self.dim_z, device=m.device)
        L[:,torch.tril(self.ones,-1) == 1] = l
        M[:,torch.tril(self.ones) == 1] = m
        # L skew-symmetric
        L = L - torch.transpose(L,1,2)
        # M symmetric and positive semi-definite
        M = torch.bmm(M,torch.transpose(M,1,2))

        return L, M, dEdz.unsqueeze(2), dSdz.unsqueeze(2), E, S



# Standard Mesh Graph Net
class MeshGraphNet(torch.nn.Module):
    def __init__(self, args, dims):
        super(MeshGraphNet, self).__init__()
        # Arguments
        passes = args.passes
        n_hidden = args.n_hidden
        dim_hidden = args.dim_hidden
        self.dims = dims
        self.dim_z = self.dims['z']
        dim_node = self.dims['z'] + self.dims['n']
        dim_edge = self.dims['q_0'] + 1

        # Encoder MLPs
        self.encoder_node = MLP([dim_node] + n_hidden*[dim_hidden] + [dim_hidden])
        self.encoder_edge = MLP([dim_edge] + n_hidden*[dim_hidden] + [dim_hidden])
        # Processor MLPs
        self.processor = nn.ModuleList()
        for _ in range(passes):
            node_model = NodeModel(args, self.dims)
            edge_model = EdgeModel(args, self.dims)
            GraphNet = MetaLayer(node_model=node_model, edge_model=edge_model)
            self.processor.append(GraphNet)

        self.decoder_node = MLP([dim_hidden] + n_hidden*[dim_hidden] + [self.dims['z']])

    def forward(self, z, n, edge_index, q_0): 
        '''Pre-process'''

        x = torch.cat((z,n), dim=1)
        # Edge attributes
        src, dest = edge_index
        u = q_0[src] - q_0[dest]
        u_norm = torch.norm(u,dim=1).reshape(-1,1)
        edge_attr = torch.cat((u,u_norm), dim=1)

        '''Encode'''
        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)

        '''Process'''
        for GraphNet in self.processor:
            x_res, edge_attr_res = GraphNet(x, edge_index, edge_attr)
            x += x_res
            edge_attr += edge_attr_res

        '''Decode'''
        x = self.decoder_node(x)

        return x


# GNN Flow Match Model
class NodeMovement(torch.nn.Module):
    def __init__(self, args, dims):
        super(NodeMovement, self).__init__()
        # Arguments
        passes = args.passes
        n_hidden = args.n_hidden
        dim_hidden = args.dim_hidden
        self.dims = dims
        dim_node = self.dims['n']+1
        dim_edge = self.dims['q_0'] + 1

        # Encoder MLPs
        self.encoder_node = MLP([dim_node] + n_hidden*[dim_hidden] + [dim_hidden], nn.ReLU())
        self.encoder_edge = MLP([dim_edge] + n_hidden*[dim_hidden] + [dim_hidden], nn.ReLU())
        # Processor MLPs
        self.processor = nn.ModuleList()
        for _ in range(passes):
            node_model = NodeModel(args, self.dims, nn.ReLU())
            edge_model = EdgeModel(args, self.dims, nn.ReLU())
            GraphNet = MetaLayer(node_model=node_model, edge_model=edge_model)
            self.processor.append(GraphNet)

        # self.decoder_node = MLP([dim_hidden] + n_hidden*[dim_hidden] + [self.dims['q_0']], nn.ReLU())
        self.decoder_node = MLP([dim_hidden] + n_hidden*[dim_hidden] + [self.dims['q_0']])

    def forward(self, q_0, n, t, edge_index): 
        x = torch.cat((n,t),dim=1)
        # Edge attributes
        src, dest = edge_index
        u = q_0[src] - q_0[dest]
        u_norm = torch.norm(u,dim=1).reshape(-1,1)
        edge_attr = torch.cat((u,u_norm), dim=1)

        '''Encode'''
        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)

        '''Process'''
        for GraphNet in self.processor:
            x_res, edge_attr_res = GraphNet(x, edge_index, edge_attr)
            x += x_res
            edge_attr += edge_attr_res

        '''Decode'''
        x = self.decoder_node(x)

        return x
    


# GNN Flow Match Model with Global Encoding
class NodeMovementGlobal(torch.nn.Module):
    def __init__(self, args, dims):
        super(NodeMovementGlobal, self).__init__()
        # Arguments
        passes = args.passes
        n_hidden = args.n_hidden
        dim_hidden = args.dim_hidden
        self.dims = dims
        dim_node = self.dims['n']+1
        dim_edge = self.dims['q_0'] + 1

        # Encoder MLPs
        self.encoder_node = MLP([dim_node] + n_hidden*[dim_hidden] + [dim_hidden], nn.ReLU())
        self.encoder_edge = MLP([dim_edge] + n_hidden*[dim_hidden] + [dim_hidden], nn.ReLU())
        self.encoder_globals = MLP([dim_node] + n_hidden*[dim_hidden] + [dim_hidden], nn.ReLU())
        # Processor MLPs
        self.processor = nn.ModuleList()
        for _ in range(passes):
            node_model = NodeModel(args, self.dims, nn.ReLU())
            edge_model = EdgeModel(args, self.dims, nn.ReLU())
            GraphNet = MetaLayer(node_model=node_model, edge_model=edge_model)
            self.processor.append(GraphNet)

        self.decoder_node = MLP([dim_hidden] + n_hidden*[dim_hidden] + [self.dims['q_0']])

    # @torch.compile
    # @torch.autocast('mps')
    def forward(self, q_0, n, t, edge_index): 
        x = torch.cat((n,t),dim=1)
        # Edge attributes
        src, dest = edge_index
        u = q_0[src] - q_0[dest]
        u_norm = torch.norm(u,dim=1).reshape(-1,1)
        edge_attr = torch.cat((u,u_norm), dim=1)

        '''Encode'''
        encoded_global = self.encoder_globals(x)
        encoded_global = encoded_global.mean(0)
        encoded_global = encoded_global.repeat(x.shape[0],1)

        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)

        '''Process'''
        for GraphNet in self.processor:
            x_res, edge_attr_res = GraphNet(x, edge_index, edge_attr, f=encoded_global)
            x += x_res
            edge_attr += edge_attr_res

        '''Decode'''
        x = self.decoder_node(x)

        return x
    

class NodeMovementGlobalN(torch.nn.Module):
    def __init__(self, args, dims):
        super(NodeMovementGlobalN, self).__init__()
        # Arguments
        passes = args.passes
        n_hidden = args.n_hidden
        dim_hidden = args.dim_hidden
        self.dims = dims
        dim_node = self.dims['n']+1
        # dim_node = 3+1
        dim_edge = self.dims['q_0'] + 1

        # self.dims['f'] = 64

        # Encoder MLPs
        self.encoder_node = MLP([dim_node] + n_hidden*[dim_hidden] + [dim_hidden])
        self.encoder_edge = MLP([dim_edge] + n_hidden*[dim_hidden] + [dim_hidden])
        # self.encoder_globals = MLP([dim_node] + n_hidden*[dim_hidden] + [self.dims['f']], nn.ReLU())
        self.encoder_globals = MLP([dim_node] + n_hidden*[dim_hidden] + [dim_hidden])
        # Processor MLPs
        self.processor = nn.ModuleList()
        for _ in range(passes):
            node_model = NodeModel(args, self.dims)
            edge_model = EdgeModel(args, self.dims)
            GraphNet = MetaLayer(node_model=node_model, edge_model=edge_model)
            self.processor.append(GraphNet)

        # self.decoder_node = MLP([dim_hidden] + n_hidden*[dim_hidden] + [self.dims['q_0']], nn.ReLU())
        self.decoder_node = MLP([dim_hidden] + n_hidden*[dim_hidden] + [self.dims['q_0']])

    # @torch.compile
    # @torch.autocast('mps')
    def forward(self, q_0, n, t, edge_index): 
        x = torch.cat((n,t),dim=1)
        # Edge attributes
        src, dest = edge_index
        u = q_0[src] - q_0[dest]
        u_norm = torch.norm(u,dim=1).reshape(-1,1)
        edge_attr = torch.cat((u,u_norm), dim=1)

        '''Encode'''
        encoded_global = self.encoder_globals(x)
        encoded_global = encoded_global.mean(0)
        encoded_global = encoded_global.repeat(x.shape[0],1)

        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)

        '''Process'''
        for GraphNet in self.processor:
            x_res, edge_attr_res = GraphNet(x, edge_index, edge_attr, f=encoded_global)
            # x_res, edge_attr_res = GraphNet(x, edge_index, edge_attr)
            x += x_res
            edge_attr += edge_attr_res

        '''Decode'''
        x = self.decoder_node(x)

        return x

# GNN flow matching corrector model
class NodeMovementCorrector(torch.nn.Module):
    def __init__(self, args, dims, fm=False):
        super(NodeMovementCorrector, self).__init__()
        # Arguments
        passes = args.passes
        n_hidden = args.n_hidden
        dim_hidden = args.dim_hidden
        self.dims = dims
        self.dim_z = self.dims['z']
        dim_node = self.dims['n']
        if fm:
            dim_node = self.dims['n'] + 1
        dim_edge = self.dims['q_0'] + 1

        # Encoder MLPs
        # self.encoder_node = MLP([dim_node] + n_hidden*[dim_hidden] + [dim_hidden], nn.ReLU())
        # self.encoder_edge = MLP([dim_edge] + n_hidden*[dim_hidden] + [dim_hidden], nn.ReLU())
        self.encoder_node = MLP([dim_node] + n_hidden*[dim_hidden] + [dim_hidden])
        self.encoder_edge = MLP([dim_edge] + n_hidden*[dim_hidden] + [dim_hidden])
        # Processor MLPs
        self.processor = nn.ModuleList()
        for _ in range(passes):
            # node_model = NodeModel(args, self.dims, nn.ReLU())
            # edge_model = EdgeModel(args, self.dims, nn.ReLU())
            node_model = NodeModel(args, self.dims)
            edge_model = EdgeModel(args, self.dims)
            GraphNet = MetaLayer(node_model=node_model, edge_model=edge_model)
            self.processor.append(GraphNet)

        self.decoder_node = MLP([dim_hidden] + n_hidden*[dim_hidden] + [self.dims['q_0']])


    def forward(self, n, edge_index, q_0, t=None): 
        x = n
        if not(t is None):
            x = torch.cat((n,t),dim=1)

        # Edge attributes
        src, dest = edge_index
        u = q_0[src] - q_0[dest]
        u_norm = torch.norm(u,dim=1).reshape(-1,1)
        edge_attr = torch.cat((u,u_norm), dim=1)

        '''Encode'''
        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)

        '''Process'''
        for GraphNet in self.processor:
            x_res, edge_attr_res = GraphNet(x, edge_index, edge_attr)
            x += x_res
            edge_attr += edge_attr_res

        '''Decode'''
        x = self.decoder_node(x)

        return x


if __name__ == '__main__':
    pass
