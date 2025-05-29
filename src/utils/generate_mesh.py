import torch
import numpy as np
import tqdm

from matplotlib import pyplot as plt

from src.solver import Solver
from src.interface import Interface

def generate_mesh(solver: Solver,external_walls,internal_walls,inlet,outlet,ninternal_nodes,n_iter=10,fm_nsteps=100,fm_mesh_std_scaling=0.5):
    '''
    Creates new mesh from inputs of form
    Inputs:
        solver: Solver class containing flow matching model and methods
        external walls, internal walls, inlet, outlet, ninternal nodes: dictionary containing positions and number of nodes
        n iter: number of flow matching solves to generate mesh. Assumes intially uniform distribution of nodes, with successive
        iterations, should cluster nodes closer to body of interest (near internal walls)
        fm nsteps: number of flow matching steps used to find new distribution. total number of steps is n_iter*fm_nsteps
    '''
    external_walls['pos'] = sorted(external_walls['pos']) 

    x_span = outlet['pos'] - inlet['pos']
    y_span = external_walls['pos'][0] - external_walls['pos'][1]

    # construct boundaries
    dx = x_span/external_walls['npoints']
    x_top_wall = torch.zeros(external_walls['npoints'],2)
    x_top_wall[:,0] = torch.linspace(inlet['pos']+dx,outlet['pos']-dx,x_top_wall.shape[0])
    x_top_wall[:,1] = external_walls['pos'][0]

    n_top_wall = torch.zeros(x_top_wall.shape[0],4,dtype=int)
    n_top_wall[:,3] = 1

    x_bot_wall = torch.zeros(external_walls['npoints'],2)
    x_bot_wall[:,0] = torch.linspace(inlet['pos']+dx,outlet['pos']-dx,x_bot_wall.shape[0])
    x_bot_wall[:,1] = external_walls['pos'][1]

    n_bot_wall = torch.zeros(x_bot_wall.shape[0],4,dtype=int)
    n_bot_wall[:,3] = 1
   
    x_inlet = torch.zeros(inlet['npoints'],2)
    x_inlet[:,0] = inlet['pos']
    x_inlet[:,1] = torch.linspace(external_walls['pos'][1],external_walls['pos'][0],x_inlet.shape[0])
    
    n_inlet = torch.zeros(x_inlet.shape[0],4,dtype=int)
    n_inlet[:,1] = 1
 
    x_outlet = torch.zeros(outlet['npoints'],2)
    x_outlet[:,0] = outlet['pos']
    x_outlet[:,1] = torch.linspace(external_walls['pos'][1],external_walls['pos'][0],x_outlet.shape[0])
 
    n_outlet = torch.zeros(x_outlet.shape[0],4,dtype=int)
    n_outlet[:,2] = 1

    # generate nodes for internal walls

    # internal wall stored as segments that form a closed loop (last point connects to first point)
    # internal_walls['points'] = torch.tensor(internal_walls['points'])
    segment_len = (internal_walls['points']-torch.roll(internal_walls['points'],-1,0)).norm(dim=1)
    cum_len = torch.cumsum(segment_len,0)
    ds = cum_len[-1]/internal_walls['npoints']
    s_locations = torch.linspace(0,cum_len[-1]-ds,internal_walls['npoints'])
    cum_len = torch.cat([torch.zeros(1),cum_len])
    interp_points = torch.cat([internal_walls['points'],internal_walls['points'][0].unsqueeze(0)])

    x_internal_wall = torch.zeros(internal_walls['npoints'],2)
    x_internal_wall[:,0] = torch.tensor(np.interp(s_locations,cum_len,interp_points[:,0]))
    x_internal_wall[:,1] = torch.tensor(np.interp(s_locations,cum_len,interp_points[:,1]))

    n_internal_wall = torch.zeros(x_internal_wall.shape[0],4,dtype=int)
    n_internal_wall[:,3] = 1
 
    # initally assume uniform distributed
    x_internal = torch.rand(ninternal_nodes,2)
    x_internal[:,0] = x_internal[:,0]*x_span + inlet['pos']
    x_internal[:,1] = x_internal[:,1]*y_span + external_walls['pos'][1]

    n_internal = torch.zeros(x_internal.shape[0],4,dtype=int)
    n_internal[:,0] = 1


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(x_internal[:,0].cpu(),x_internal[:,1].cpu())
    # # ax.plot(x_internal[:,0].cpu(),x_internal[:,1].cpu())
    # plt.show()


    # concat all nodes together
    x_all = torch.cat([
        x_top_wall,
        x_bot_wall,
        x_internal_wall,
        x_inlet,
        x_outlet,
        x_internal
    ]).to(solver.device)

    n_all = torch.cat([
        n_top_wall,
        n_bot_wall,
        n_internal_wall,
        n_inlet,
        n_outlet,
        n_internal
    ]).to(solver.device)

    
    # for _ in range(50):
    print('Generating Mesh')
    for _ in tqdm.tqdm(range(n_iter)):
        # add noise for flow matching to redistribute nodes
        noise = torch.randn_like(x_internal)*y_span*fm_mesh_std_scaling

        mask = n_all[:,0] == 1
        x_all[mask] += noise.to(solver.device)

        # build new n tensor for exterior and interior walls
        fm_n = solver.find_boundaries(x_all,n_all).to(solver.device)
        # use solver to generate new node positions
        x_all = solver.redistribute_nodes(x_all,fm_n,fm_nsteps)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # for i in range(fm_n.shape[1]):
        #     mask = (fm_n[:,i] == 1).cpu()
        #     ax.scatter(x_all[mask,0].cpu(),x_all[mask,1].cpu())
        # # ax.plot(x_internal[:,0].cpu(),x_internal[:,1].cpu())
        # plt.show()

    # use interface.build_edges to generate edges
    _,edges,_ = Interface.build_triangle_edges(x_all.cpu().numpy())

    return x_all, edges, n_all
