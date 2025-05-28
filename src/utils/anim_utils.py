from matplotlib import pyplot as plt
from matplotlib import animation

import os
# import warnings
# import yaml
import tqdm
import torch
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# warnings.filterwarnings('ignore')

# from torch_geometric.data import Data

# from src.interface import Interface
from src.solver import Solver
import matplotlib.cm as cm


def generate_histories(solver_inputs,iter,ic_data):
    field_history = []
    pos_history = []
    n_history = []
    solver = Solver(solver_inputs,ic_data=ic_data)
    # pos_history.append(solver.pos)
    # field_history.append(solver.pull_field())
    print('Generating Simulation')
    for _ in tqdm.tqdm(range(iter)):
        solver.update()
        solver.update_bc()
        pos_history.append(solver.pos)
        field_history.append(solver.pull_field()[0])
        n_history.append(solver.n)

    return field_history,pos_history,n_history

def animate_history(anim_parameters,solver_inputs,field_history,pos_history,n_history,edge_history):
    save_name = anim_parameters['save_name']
    save_dir = anim_parameters['save_dir']

    title_fmt = 'Time: %06.3f'
    fig = plt.figure(dpi=anim_parameters['dpi'],figsize=anim_parameters['figsize'])
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.set_title(title_fmt%(0))


    # Initial snapshot
    for i in range(len(field_history)):
        z_min_t, z_max_t = field_history[i].min(0), field_history[i].max(0)
        z_min_t, z_max_t = z_min_t[0],z_max_t[0]
        if i == 0:
            z_min = z_min_t
            z_max = z_max_t
        else:
            arr = torch.vstack([z_min,z_min_t])
            z_min = torch.min(arr,0)[0]
            arr = torch.vstack([z_max,z_max_t])
            z_max = torch.max(arr,0)[0]

    # field_tensor = torch.stack(field_history)
    # z_mean = field_tensor.mean([0,1])
    # z_std = field_tensor.std([0,1])
    # z_min = z_mean - 2*z_std
    # z_max = z_mean + 2*z_std

    for i in range(3):
        if anim_parameters['field_min'][i] != 'None':
            z_min[i] = anim_parameters['field_min'][i]
        if anim_parameters['field_max'][i] != 'None':
            z_max[i] = anim_parameters['field_max'][i]

    field = field_history[0].cpu()
    if isinstance(pos_history,list):
        pos = pos_history[0].cpu()
        edges = edge_history[0].cpu()
        n = n_history[0].cpu()
    else:
        pos = pos_history.cpu()
        edges = edge_history.cpu()
        n = n_history.cpu()
    
    levels1 = np.linspace(z_min[0], z_max[0], 5)
    levels2 = np.linspace(z_min[1], z_max[1], 5)    
    levels3 = np.linspace(z_min[2], z_max[2], 5)    
    # levels1 = np.linspace(z_min[0], z_max[0], anim_parameters['contour_levels'])
    # levels2 = np.linspace(z_min[1], z_max[1], anim_parameters['contour_levels'])    
    # levels3 = np.linspace(z_min[2], z_max[2], anim_parameters['contour_levels'])    
    # c1 = ax1.tricontourf(pos[:,0],pos[:,1],field[:,0].cpu(),shading=anim_parameters['contour_shading'],cmap=cm.viridis)
    # c2 = ax2.tricontourf(pos[:,0],pos[:,1],field[:,1].cpu(),shading=anim_parameters['contour_shading'],cmap=cm.viridis)
    # c3 = ax3.tricontourf(pos[:,0],pos[:,1],field[:,2].cpu(),shading=anim_parameters['contour_shading'],cmap=cm.plasma)

    c1 = ax1.tripcolor(pos[:,0],pos[:,1],field[:,0].cpu(),vmax=z_max[0],vmin=z_min[0],shading=anim_parameters['contour_shading'],cmap=cm.viridis)
    c2 = ax2.tripcolor(pos[:,0],pos[:,1],field[:,1].cpu(),vmax=z_max[1],vmin=z_min[1],shading=anim_parameters['contour_shading'],cmap=cm.viridis)
    c3 = ax3.tripcolor(pos[:,0],pos[:,1],field[:,2].cpu(),vmax=z_max[2],vmin=z_min[2],shading=anim_parameters['contour_shading'],cmap=cm.plasma)

    wall_mask = n[:,3] == 1
    ax1.scatter(pos[wall_mask,0],pos[wall_mask,1],c='k',s=anim_parameters['wall_plot_size'])
    ax2.scatter(pos[wall_mask,0],pos[wall_mask,1],c='k',s=anim_parameters['wall_plot_size'])
    ax3.scatter(pos[wall_mask,0],pos[wall_mask,1],c='k',s=anim_parameters['wall_plot_size'])




    # ax1.set_title()
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])

    # ax1.set_xlim(pos[:,0].min(),pos[:,0].max())
    # ax2.set_xlim(pos[:,0].min(),pos[:,0].max())
    # ax3.set_xlim(pos[:,0].min(),pos[:,0].max())
    # ax1.set_ylim(pos[:,1].min(),pos[:,1].max())
    # ax2.set_ylim(pos[:,1].min(),pos[:,1].max())
    # ax3.set_ylim(pos[:,1].min(),pos[:,1].max())

    mask_non_internals = n[:,0] == 0
    ax1.set_xlim(pos[mask_non_internals,0].min(),pos[mask_non_internals,0].max())
    ax2.set_xlim(pos[mask_non_internals,0].min(),pos[mask_non_internals,0].max())
    ax3.set_xlim(pos[mask_non_internals,0].min(),pos[mask_non_internals,0].max())
    ax1.set_ylim(pos[mask_non_internals,1].min(),pos[mask_non_internals,1].max())
    ax2.set_ylim(pos[mask_non_internals,1].min(),pos[mask_non_internals,1].max())
    ax3.set_ylim(pos[mask_non_internals,1].min(),pos[mask_non_internals,1].max())



    # ax1.set_xlim(anim_parameters['inlet']['pos'],anim_parameters['outlet']['pos'])
    # ax2.set_xlim(anim_parameters['inlet']['pos'],anim_parameters['outlet']['pos'])
    # ax3.set_xlim(anim_parameters['inlet']['pos'],anim_parameters['outlet']['pos'])

    # sorted_wall_pos = sorted(anim_parameters['external_wall']['pos'])
    # ax1.set_ylim(sorted_wall_pos[1],sorted_wall_pos[0])
    # ax2.set_ylim(sorted_wall_pos[1],sorted_wall_pos[0])
    # ax3.set_ylim(sorted_wall_pos[1],sorted_wall_pos[0])

    # Plot Boundaries
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')
    # Colorbar
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes('right', size='5%', pad=0.05)
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes('right', size='5%', pad=0.05)
    div3 = make_axes_locatable(ax3)
    cax3 = div3.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(c1, ax=ax1, cax=cax1, ticks=levels1, format='%.2f')
    fig.colorbar(c2, ax=ax2, cax=cax2, ticks=levels2, format='%.2f')
    fig.colorbar(c3, ax=ax3, cax=cax3, ticks=levels3, format='%.2f')


    print('Generating Animation')
    anim_bar = tqdm.tqdm(range(anim_parameters['iter']))
    def anim(iter):
        if iter >= anim_parameters['iter']-1:
            anim_bar.close()
        else:
            anim_bar.update()

        ax1.cla()
        ax2.cla()
        ax3.cla()

        field = field_history[iter+1]
        if isinstance(pos_history,list):
            pos = pos_history[iter+1].cpu()
            edges = edge_history[iter+1].cpu()
            n = n_history[iter+1].cpu()
        else:
            pos = pos_history.cpu()
            edges = edge_history.cpu()
            n = n_history.cpu()

        ax1.tripcolor(pos[:,0],pos[:,1],field[:,0].cpu(),vmax=z_max[0],vmin=z_min[0],shading=anim_parameters['contour_shading'],cmap=cm.viridis)
        ax2.tripcolor(pos[:,0],pos[:,1],field[:,1].cpu(),vmax=z_max[1],vmin=z_min[1],shading=anim_parameters['contour_shading'],cmap=cm.viridis)
        ax3.tripcolor(pos[:,0],pos[:,1],field[:,2].cpu(),vmax=z_max[2],vmin=z_min[2],shading=anim_parameters['contour_shading'],cmap=cm.plasma)

        wall_mask = (n[:,3] == 1).cpu()
        ax1.scatter(pos[wall_mask,0],pos[wall_mask,1],c='k',s=anim_parameters['wall_plot_size'])
        ax2.scatter(pos[wall_mask,0],pos[wall_mask,1],c='k',s=anim_parameters['wall_plot_size'])
        ax3.scatter(pos[wall_mask,0],pos[wall_mask,1],c='k',s=anim_parameters['wall_plot_size'])

        ax1.set_aspect('equal',adjustable='box') 
        ax2.set_aspect('equal',adjustable='box') 
        ax3.set_aspect('equal',adjustable='box') 

        ax1.set_xticks([])
        ax2.set_xticks([])
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])

        ax1.set_title(title_fmt%(iter*solver_inputs['dt']))
        ax1.set_ylabel('X Velocity')
        ax2.set_ylabel('Y Velocity')
        ax3.set_ylabel('Pressure')

        # ax1.set_xlim(pos[:,0].min(),pos[:,0].max())
        # ax2.set_xlim(pos[:,0].min(),pos[:,0].max())
        # ax3.set_xlim(pos[:,0].min(),pos[:,0].max())

        # ax1.set_ylim(pos[:,1].min(),pos[:,1].max())
        # ax2.set_ylim(pos[:,1].min(),pos[:,1].max())
        # ax3.set_ylim(pos[:,1].min(),pos[:,1].max())

        mask_non_internals = n[:,0] == 0
        ax1.set_xlim(pos[mask_non_internals,0].min(),pos[mask_non_internals,0].max())
        ax2.set_xlim(pos[mask_non_internals,0].min(),pos[mask_non_internals,0].max())
        ax3.set_xlim(pos[mask_non_internals,0].min(),pos[mask_non_internals,0].max())
        ax1.set_ylim(pos[mask_non_internals,1].min(),pos[mask_non_internals,1].max())
        ax2.set_ylim(pos[mask_non_internals,1].min(),pos[mask_non_internals,1].max())
        ax3.set_ylim(pos[mask_non_internals,1].min(),pos[mask_non_internals,1].max())
        return fig

    anim = animation.FuncAnimation(fig,anim,anim_parameters['iter'])
    # print('Saving Animation')
    writer = animation.PillowWriter(fps=anim_parameters['fps'])

    anim.save(os.path.join(save_dir,'%s.gif'%(save_name)),writer=writer)

    if anim_parameters['plot_graph']:
        print('Generating Graph Animation')
        graph_figsize = anim_parameters['figsize']
        graph_figsize[1] = int(round(graph_figsize[1]/3))
        fig = plt.figure(dpi=anim_parameters['dpi'],figsize=graph_figsize)
        ax = fig.add_subplot(111)
        ax.plot(pos[edges,0],pos[edges,1],c='k')
        for i in range(n.shape[1]):
            mask = n[:,i] == 1
            ax.scatter(pos[mask,0],pos[mask,1])

        ax1.set_title(title_fmt%(0))

        tot_iter = anim_parameters['iter'] if isinstance(pos_history,list) else 1
        anim_bar = tqdm.tqdm(range(tot_iter))

        def anim_graph(iter):
            if iter >= tot_iter-1:
                anim_bar.close()
            else:
                anim_bar.update()
            ax.cla()

            if isinstance(pos_history,list):
                pos = pos_history[iter+1].cpu()
                edges = edge_history[iter+1].cpu()
                n = n_history[iter+1].cpu()
            else:
                pos = pos_history.cpu()
                edges = edge_history.cpu()
                n = n_history.cpu()

            ax.plot(pos[edges,0],pos[edges,1],c='k',linewidth=anim_parameters['line_width'])
            for i in range(n.shape[1]):
                mask = n[:,i] == 1
                ax.scatter(pos[mask,0],pos[mask,1],s=anim_parameters['scatter_size'])

                ax.set_aspect('equal',adjustable='box') 

            if tot_iter > 1: 
                ax.set_title(title_fmt%(iter*solver_inputs['dt']))
            return fig

        anim = animation.FuncAnimation(fig,anim_graph,tot_iter)
        # print('Saving Animation')
        writer = animation.PillowWriter(fps=anim_parameters['fps'])

        anim.save(os.path.join(save_dir,'%s_graph.gif'%(save_name)),writer=writer)

    print('Done')

def plot_graph(ax: plt.axes,pos,edges,n):
    # plot edges
    ax.plot(pos[edges,0],pos[edges,1],c='k')
    for i in range(n.shape[1]):
        mask = n[:,i] == 1
        ax.scatter(pos[mask,0],pos[mask,1])

