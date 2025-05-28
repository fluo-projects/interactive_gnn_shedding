from matplotlib import pyplot as plt
# from matplotlib import animation

import os
import warnings
import yaml
import tqdm
import torch
import argparse

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
warnings.filterwarnings('ignore')

# from torch_geometric.data import Data
from scipy.optimize import linear_sum_assignment

from src.interface import Interface
from src.solver import Solver

from src.utils.generate_mesh import generate_mesh
from src.utils.initialization import load_in

import matplotlib.cm as cm
from src.utils.anim_utils import *

def main():
    parser = argparse.ArgumentParser(description='Interactive GNN interface')

    parser.add_argument('--parameter',default='parameters.yaml',type=str,help='YAML parameter file for parameters of simulation and interface')
    parser.add_argument('--anim_parameter',default='anim_parameters.yaml',type=str,help='YAML parameter file for parameters of simulation and interface')

    args = parser.parse_args()

    with open(args.parameter,'r') as f:
        solver_inputs = yaml.load(f,Loader=yaml.FullLoader)
    with open(args.anim_parameter,'r') as f:
        anim_parameters = yaml.load(f,Loader=yaml.FullLoader)
    
    solver = Solver(solver_inputs)

    if anim_parameters['custom']:
        # generate new meshes

        custom_mesh = anim_parameters['custom_parameters']
        if isinstance(custom_mesh['internal_walls']['points'][0][0],list):
            # if nested list of points, generate evolving mesh

            # ensure fully defined points and transition times
            iter_list = custom_mesh['iter_list']
            points_list = custom_mesh['internal_walls']['points']
            assert len(iter_list) >= len(points_list)-1

            pos_history = []

            # generate first mesh

            internal_wall_parameter = {
                'points': torch.tensor(points_list[0]),
                # 'npoints': custom_mesh['internal_walls']['npoints'][0] if isinstance(custom_mesh['internal_walls']['npoints'],list) else custom_mesh['internal_walls']['npoints']
                'npoints': custom_mesh['internal_walls']['npoints']
            }

            print('Generating %i meshes and interpolating between meshes'%(len(points_list)))
            pos_prior,edges_prior,n_prior = generate_mesh(solver,
                                                          custom_mesh['external_walls'],
                                                          internal_wall_parameter,
                                                          custom_mesh['inlet'],
                                                          custom_mesh['outlet'],
                                                          custom_mesh['ninternal_points'],
                                                          fm_mesh_std_scaling=solver_inputs['fm_mesh']['std_scaling']) 

            mask_internal_node = n_prior[:,0] == 1 
            mask_internal_wall = n_prior[:,3] == 1 
            mask_internal_wall[:mask_internal_wall.sum()-custom_mesh['internal_walls']['npoints']] = False

            pos_history = [pos_prior]
            n_history = [n_prior]
            edge_history = [edges_prior]
            for i in range(1,min(len(iter_list),len(points_list))):
                # generate new mesh

                internal_wall_parameter = {
                    'points': torch.tensor(points_list[i]),
                    'npoints': custom_mesh['internal_walls']['npoints'][i] if isinstance(custom_mesh['internal_walls']['npoints'],list) else custom_mesh['internal_walls']['npoints']
                }

                pos,edges,n = generate_mesh(solver,custom_mesh['external_walls'],internal_wall_parameter,custom_mesh['inlet'],custom_mesh['outlet'],custom_mesh['ninternal_points']) 
                assert torch.all(n==n_prior)
 
                # match to old mesh
                inds = torch.arange(pos_prior.shape[0]).to(solver.device)
                l2_norm_internal = torch.cdist(pos_prior[mask_internal_node],pos[mask_internal_node])
                internal_ind_prior,internal_ind_curr = linear_sum_assignment(l2_norm_internal.cpu().numpy()**2)

                pos_internal_prior = pos_prior[mask_internal_node][internal_ind_prior]
                pos_internal_curr = pos[mask_internal_node][internal_ind_curr]
                fwd_internal_inds = inds[mask_internal_node][internal_ind_prior]

                l2_norm_internal_wall = torch.cdist(pos_prior[mask_internal_wall],pos[mask_internal_wall])
                internal_wall_ind_prior,internal_wall_ind_curr = linear_sum_assignment(l2_norm_internal_wall.cpu().numpy()**2)
                fwd_internal_wall_inds = inds[mask_internal_wall][internal_wall_ind_prior]

                pos_internal_wall_prior = pos_prior[mask_internal_wall][internal_wall_ind_prior]
                pos_internal_wall_curr = pos[mask_internal_wall][internal_wall_ind_curr]

                assert iter_list[i-1]>0
                for j in range(iter_list[i-1]):
                    # interpolate between meshes 
                    eta = j/iter_list[i-1]

                    # 
                    pos_curr = pos_prior.detach().clone()
                    # pos_curr[mask_internal_node] = ((1-eta)*pos_internal_prior+eta*pos_internal_curr)[revert_internal_ind]
                    # pos_curr[mask_internal_wall] = ((1-eta)*pos_internal_wall_prior+eta*pos_internal_wall_curr)[revert_internal_wall_ind]
                    # pos_curr[mask_internal_node][internal_ind_prior] = ((1-eta)*pos_internal_prior+eta*pos_internal_curr)
                    # pos_curr[mask_internal_wall][internal_wall_ind_prior] = ((1-eta)*pos_internal_wall_prior+eta*pos_internal_wall_curr)
                    pos_curr[fwd_internal_inds] = ((1-eta)*pos_internal_prior+eta*pos_internal_curr)
                    pos_curr[fwd_internal_wall_inds] = ((1-eta)*pos_internal_wall_prior+eta*pos_internal_wall_curr)

                    # generate new edges
                    _,edges,_ = Interface.build_triangle_edges(pos_curr.cpu().numpy())

                    pos_history.append(pos_curr)
                    edge_history.append(edges)
                    n_history.append(n)

                # match nodes into new mesh
                pos[fwd_internal_inds] = pos_internal_curr
                pos[fwd_internal_wall_inds] = pos_internal_wall_curr
                # _,edges,_ = Interface.build_triangle_edges(pos.cpu().numpy())
                pos_prior, edges_prior, n_prior = pos.detach(), edges.detach(), n.detach()

            # if iter is greater than the sum of iter_list, keep the last state
            continue_iterations = anim_parameters['iter']-(len(pos_history)-1)
            if continue_iterations > 0:
                pos_history += list([pos_prior.detach().clone()]*continue_iterations)
                edge_history += list([edges_prior.detach().clone()]*continue_iterations)
                n_history += list([n_prior.detach().clone()]*continue_iterations)


            
        else:
            custom_mesh['internal_walls']['points'] = torch.tensor(custom_mesh['internal_walls']['points'])

            pos,edges,n = generate_mesh(solver,
                                        custom_mesh['external_walls'],
                                        custom_mesh['internal_walls'],
                                        custom_mesh['inlet'],
                                        custom_mesh['outlet'],
                                        custom_mesh['ninternal_points'],
                                        fm_mesh_std_scaling=solver_inputs['fm_mesh']['std_scaling']) 
            
            pos_history = pos
            n_history = n
            edge_history = edges


        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(pos[edges,0].cpu(),pos[edges,1].cpu())
        # for i in range(n.shape[1]):
        #     mask = n[:,i] == 1
        #     ax.scatter(pos[mask,0].cpu(),pos[mask,1].cpu())
        # ax.set_aspect('equal')
        # plt.show()

        field = torch.zeros(pos.shape[0],3).cpu() 
        field[:,2] = (pos[:,0].max()-pos[:,0])/(pos[:,0].max()-pos[:,0].min())+solver_inputs['outlet_p']

    else:
        pos,field,n,edges = load_in(solver_inputs)

        pos_history = pos
        n_history = n
        edge_history = edges

    field_history = [field]

    solver.update_state(field,pos,edges,n)
    # Generate history

    print('Generating Simulation')
    for i in tqdm.tqdm(range(anim_parameters['iter'])):
        solver.update()
        solver.update_bc()
        # pos_history.append(solver.pos)
        field_history.append(solver.pull_field()[0])
        # n_history.append(solver.n)

        # check if positions are evolving
        if isinstance(pos_history,list):
            solver.update_state(None,pos_history[i+1],edge_history[i+1],n_history[i+1])


    animate_history(anim_parameters,solver_inputs,field_history,pos_history,n_history,edge_history)

if __name__ == '__main__':
    main()