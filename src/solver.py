import torch

from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay

from src.interface import Interface
from src.model import TIGNN,MeshGraphNet,NodeMovement,NodeMovementGlobal,NodeMovementGlobalN,NodeMovementCorrector

from matplotlib import pyplot as plt

class model_attributes:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

class Solver(object):
    def __init__(self,
                 solver_inputs,
                 ic_data=None):
        self.solver_inputs = solver_inputs

        self.device = 'cpu'
        if solver_inputs['gpu']:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")

        # Normalization
        self.stats_z = solver_inputs['norm_stats']['stats_z']
        for key in self.stats_z.keys():
            if isinstance(self.stats_z[key],str):
                self.stats_z[key] = torch.load(self.stats_z[key],map_location=self.device)
            else:
                self.stats_z[key] = torch.tensor(self.stats_z[key],device=self.device)
        self.stats_q = solver_inputs['norm_stats']['stats_q']
        for key in self.stats_q.keys():
            if isinstance(self.stats_q[key],str):
                self.stats_q[key] = torch.load(self.stats_q[key],map_location=self.device)
            else:
                self.stats_q[key] = torch.tensor(self.stats_q[key],device=self.device)

        # Net Parameters
        self.dims = solver_inputs['model']['dims']
        self.model_type = solver_inputs['model']['model_type']
        self.model_attributes = model_attributes(solver_inputs['model']['model_attributes'])
        if self.model_type == 'tignn':
            self.net = TIGNN(self.model_attributes, self.dims).to(self.device).float() 
        elif self.model_type == 'mgn':
            self.net = MeshGraphNet(self.model_attributes, self.dims).to(self.device).float() 
        else:
            raise ValueError('Invalid model type %s'%(self.model_type))

        self.fm_dims = solver_inputs['fm_mesh']['dims']
        self.fm_model_type = solver_inputs['fm_mesh']['model_type']
        self.fm_model_attributes = model_attributes(solver_inputs['fm_mesh']['model_attributes'])
        if self.fm_model_type == 'local':
            self.fm_net = NodeMovement(self.fm_model_attributes,self.fm_dims).to(self.device).float()
        if self.fm_model_type == 'global':
            self.fm_net = NodeMovementGlobalN(self.fm_model_attributes,self.fm_dims).to(self.device).float()
        else:
            raise ValueError('Invalid model type %s'%(self.fm_model_type))

        self.net.load_state_dict(torch.load(solver_inputs['model']['model_location'],map_location=self.device))
        self.fm_net.load_state_dict(torch.load(solver_inputs['fm_mesh']['model_location'],map_location=self.device))

        self.corrector =  solver_inputs['fm_mesh']['corrector']
        self.corrector_model_attributes = model_attributes(solver_inputs['fm_mesh']['corrector_model_attributes'])
        if self.corrector:
            self.corrector_model = NodeMovementCorrector(self.corrector_model_attributes,solver_inputs['fm_mesh']['corrector_dims']).to(self.device).float()
            self.corrector_model.load_state_dict(torch.load(solver_inputs['fm_mesh']['corrector_location'],map_location=self.device))

        self.fm_std_scaling = solver_inputs['fm_mesh']['std_scaling']

        self.dt = solver_inputs['dt']

        if not(ic_data is None):
            ic_data = ic_data.detach().to(self.device)
            self.z = self.norm(ic_data.x,self.stats_z)
            self.q_0 = self.norm(ic_data.q_0,self.stats_q)
            self.n = ic_data.n
            self.edge_index = ic_data.edge_index

        self.inlet_vel = (solver_inputs['inlet_vel']-self.stats_z['mean'][0])/self.stats_z['std'][0]
        self.wall_vel = (-self.stats_z['mean'][:2])/self.stats_z['std'][:2]
        self.outlet_p = (solver_inputs['outlet_p']-self.stats_z['mean'][2])/self.stats_z['std'][2]

    def update_state(self,z=None,q_0=None,edge_index=None,n=None):
        if not(z is None):
            self.z = self.norm(z.to(self.device),self.stats_z)
        if not(q_0 is None):
            self.q_0 = self.norm(q_0.to(self.device),self.stats_q)
        if not(n is None):
            self.n = n.to(self.device)
        if not(edge_index is None):
            self.edge_index = edge_index.to(self.device)

    def update(self):
        z = self.z.detach()
        # Net forward pass + Integration
        if self.model_type == 'tignn':
            L_net, M_net, dEdz_net, dSdz_net, self.E_net, self.S_net = self.net(z, self.n, self.edge_index, q_0=self.q_0)
            dzdt_net, _, _ = self.integrator(L_net, M_net, dEdz_net, dSdz_net)
            dzdt_net = torch.clip(dzdt_net,-5,5)
            # self.z += self.dt*dzdt_net
            z = z + self.dt*dzdt_net
            self.net_out = dzdt_net
        elif self.model_type == 'mgn':
            with torch.no_grad():
                net_out = self.net(z, self.n, self.edge_index, q_0=self.q_0)

            if self.dims['d'] == 0:
                z += self.dt*net_out
            else:
                derivitive = z[:,:-self.dims['d']] + self.dt*net_out[:,:-self.dims['d']]
                direct = net_out[:,-self.dims['d']:]
                z = torch.cat([derivitive,direct],dim=1)[:]

            self.net_out = net_out
        self.z = z

    def pull_field(self):
        # Save results
        z_net = self.denorm(self.z.detach(), self.stats_z).cpu()
        # z_net = self.denorm(z.detach(), self.stats_z)
        E = None
        S = None
        if self.model_type == 'tignn':
            # if hasattr(self,'E_net'):
            E = self.E_net.detach().cpu()
            S = self.S_net.detach().cpu()
        
        return z_net,E,S

    def update_mesh(self,n_steps = 100):
        # n_steps = 100

        xt = self.denorm(self.q_0.detach(),self.stats_q)
        x_old = xt.detach().clone()

        # n = self.find_boundaries()
        n = self.find_boundaries(self.q_0,self.n).to(self.device)

        noise = torch.randn(xt.shape,device=self.device)*(xt[:,1].max()-xt[:,1].min())*self.fm_std_scaling
        # noise = torch.randn(xt.shape,device=self.device)*(xt[:,1].max()-xt[:,1].min())/2
        mask = n[:,0] == 1

        xt[mask] = xt[mask] + noise[mask]

        xt = self.redistribute_nodes(xt=xt,n=n,n_steps=n_steps)

        self.q_0 = self.norm(xt.detach(),self.stats_q)

        # update fields to nearest neighbor field
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(x_old.cpu())
        _, indices = nbrs.kneighbors(xt.cpu())
        self.z = self.z[indices.flatten()]

        return xt.cpu()

    def redistribute_nodes(self,xt,n,n_steps=100):
        '''
        Flow matching model to redistribute nodes
        '''
        dt = 1/n_steps
        t_list = torch.linspace(0,1,n_steps)
        mask = n[:,0] == 1
        with torch.no_grad():
            for i,t in enumerate(t_list):
                t = t.expand(n.shape[0]).reshape(n.shape[0],1).to(self.device)
                # construct edges at xt
                _,edges,_ = Interface.build_triangle_edges(xt.cpu().numpy(),skew_threshold=0)
                
                edges = edges.to(self.device)

                v = self.fm_net(xt,n, t, edges)  

                xt[mask] = xt[mask] + v[mask]*dt

                # if i % 10 == 0:
                #     fig = plt.figure()
                #     ax = fig.add_subplot(111)
                #     x_pos = xt.cpu()
                #     x_pos = x_pos.numpy()
                #     vec = v.cpu().numpy()
                #     plot_edges = edges.cpu().numpy()
                #     ax.plot(x_pos[plot_edges,0],x_pos[plot_edges,1],c='k',alpha=0.2)
                #     for j in range(n.shape[1]):
                #         # mask_indiv = data.n[:,j] == 1
                #         mask_indiv = n.cpu()[:,j] == 1
                #         ax.scatter(x_pos[mask_indiv,0],x_pos[mask_indiv,1])
                #         if j == 0:
                #             ax.quiver(x_pos[mask_indiv,0],x_pos[mask_indiv,1],vec[mask_indiv,0],vec[mask_indiv,1])
                #     ax.set_aspect('equal')
                #     ax.set_title('%i of %i'%(i,len(t_list)-1))
                #     plt.show()
        if self.corrector:
            _,edges,_ = Interface.build_triangle_edges(xt.cpu().numpy(),skew_threshold=0)
            with torch.no_grad():
                v_corrector = self.corrector_model(n,edges.to(self.device),xt)
            xt[mask] += v_corrector[mask]
        return xt

    @property
    def pos(self):
        return self.denorm(self.q_0.detach(),self.stats_q) 
    
    @staticmethod
    def find_boundaries(pos,n):
        '''
        creates new one hot tensor which classifies wall nodes as exterior or interior wall nodes
        '''
        new_n = torch.zeros((pos.shape[0],3))
        mask = n[:,0] != 1
        new_n[~mask,0] = 1
        new_n[mask,2] = 1
    
        tri = Delaunay(pos[mask].cpu())
        inds = torch.arange(pos.shape[0])
        mask_inds = inds[mask.cpu()][tri.convex_hull.flatten()]
        new_n[mask_inds,2] = 0
        new_n[mask_inds,1] = 1
        return new_n

    def update_bc(self,push=False):
        # bc handler
        wall_mask = self.n[:,3]==1
        self.z[wall_mask,:2] = self.wall_vel

        inlet_mask = self.n[:,1]==1

        # Provide temporary pressure increase based on dynamic pressure difference to ensure upstream pressures are greater than downstream pressures
        # assume rho to be close to 1 (air kg/m3 @ stp)
        # rho = 1.0
        # if push:
        #     diff = ((self.z[inlet_mask,0]*self.stats_z['std'][0]+self.stats_z['mean'][0])**2 - (self.inlet_vel*self.stats_z['std'][0]+self.stats_z['mean'][0])**2).mean()
        #     self.z[inlet_mask,2] -= (rho*diff)/2

        if self.solver_inputs['inlet_vel_profile'] == 'constant':
            self.z[inlet_mask,0] = self.inlet_vel
            # dt = (self.inlet_vel-self.z[inlet_mask,0])/self.net_out[inlet_mask,0]
            # self.z[inlet_mask,:] = self.z[inlet_mask,:]+self.net_out[inlet_mask,:]*dt.unsqueeze(1).repeat(1,3)
        elif self.solver_inputs['inlet_vel_profile'] == 'parabolic':
            # in vortex case, fully developed couette flow -> velocity inlet profile parabolic
            pos_y = self.q_0[inlet_mask,1]
            span = pos_y.max() - pos_y.min()
            mid_pt = pos_y.min() + span/2
            scale = self.inlet_vel-self.wall_vel[0]
            self.z[inlet_mask,0] = -scale*((pos_y-mid_pt)/span*2)**2+scale+self.wall_vel[0]
            # vel_profile = -scale*((pos_y-mid_pt)/span*2)**2+scale+self.wall_vel[0]
            # dt = (vel_profile-self.z[inlet_mask,0])/self.net_out[inlet_mask,0]
            # self.z[inlet_mask,:] = self.z[inlet_mask,:]+self.net_out[inlet_mask,:]*dt.unsqueeze(1).repeat(1,3)
        else:
            raise ValueError('%s is a invalid inlet velocity profile type'%(self.solver_inputs['inlet_vel_profile']))
        self.z[inlet_mask,1] = self.wall_vel[1]

        outlet_mask = self.n[:,2]==1
        self.z[outlet_mask,2] = self.outlet_p


    # Normalization function
    def norm(self, z, stats):
        return (z - stats['mean']) / stats['std']

    # Denormalization function
    def denorm(self, z, stats):
        return z * stats['std'] + stats['mean']

    # Forward-Euler Integrator for TIGNN
    def integrator(self, L, M, dEdz, dSdz):
        # GENERIC time integration and degeneration
        dzdt = torch.bmm(L,dEdz) + torch.bmm(M,dSdz)
        deg_E = torch.bmm(M,dEdz)
        deg_S = torch.bmm(L,dSdz)

        return dzdt[:,:,0], deg_E[:,:,0], deg_S[:,:,0]

if __name__ == '__main__':
    pass
