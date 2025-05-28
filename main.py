import os
import warnings
import yaml
import argparse

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
warnings.filterwarnings('ignore')

from torch_geometric.data import Data

from src.interface import Interface
from src.solver import Solver
from src.utils.initialization import load_in

def main():
    parser = argparse.ArgumentParser(description='Interactive GNN interface')
    parser.add_argument('--parameter',default='parameters.yaml',type=str,help='YAML parameter file for parameters of simulation and interface')
    args = parser.parse_args()

    with open(args.parameter,'r') as f:
        solver_inputs = yaml.load(f,Loader=yaml.FullLoader)


    pos,field,n,edges = load_in(solver_inputs)

    data = Data(x=field,n=n,edge_index=edges,q_0=pos).cpu()

    # intialize user interface and solver
    interface = Interface(solver_inputs,data)
    solver = Solver(solver_inputs,ic_data=data)

    t = 0
    while interface.running:
        if interface.update_state:
            # iterates state forward in time
            solver.update()

            # applies bcs, applies pressure increase if recently updated mesh to correct odd pressure behavior
            solver.update_bc(t<50)

            # update fields in interface for display
            interface.update_parameters(*solver.pull_field())
            t += 1

        interface.clock.tick(60)
        interface.handle_events()
        interface.update()
        interface.draw()

        if interface.dirty:
            # if node or edges are changed, update to solver
            solver.update_state(*interface.export_state())
            interface.dirty = False

        if interface.node_update:
            # handels remeshing of body
            interface.update_mesh(solver.update_mesh(),update_edges=True)
            # push updated fields to interface
            interface.update_parameters(*solver.pull_field())
            solver.update_state(*interface.export_state())
            interface.node_update = False
            # reset pressure corrections
            t = 0

    interface.exit()


if __name__ == '__main__':
    main()