import torch
import pickle

def load_in(solver_inputs):
    # load initial conditions 
    if not(solver_inputs['initial_conditions']['pkl_fname']):
        pos = torch.load(solver_inputs['initial_conditions']['pos_fname']).float()
        field = torch.load(solver_inputs['initial_conditions']['field_fname']).float()
        n = torch.load(solver_inputs['initial_conditions']['n_fname']).long()
        edges = torch.load(solver_inputs['initial_conditions']['edge_fname']).long()
    else:
        with open(solver_inputs['initial_conditions']['pkl_fname'], 'rb') as f:
            data = pickle.load(f)
            pos = data['pos']
            field = data['field']
            n = data['n']
            edges = data['edges']

    # if field contains time varying component
    if len(field.shape) == 3:
        field = field[solver_inputs['initial_conditions']['initial_iter']]


    if solver_inputs['initial_conditions']['reset_state']:
        field[:,:2] = 0
        field[:,2] = (pos[:,0].max()-pos[:,0])/(pos[:,0].max()-pos[:,0].min())+solver_inputs['outlet_p']

    return pos,field,n,edges