# Interactive GNN for Vortex Shedding 

This repo contains a Pygame-opengl framework for a interactive gnn simulation of vortex shedding around a cylinder. This work is heavily inspired by Hernández, Badías, Chinesta and Cueto in their Thermodynamics-informed graph neural networks (TIGNN) paper ([link to github](https://github.com/quercushernandez/ThermodynamicsGNN/tree/main)) as well as earlier work in Mesh Graph Nets (MGN) by Pfaff, Fortunato, Sanchez-Gonzalez, and Battaglia ([link to github](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)). For more information about the models, check the credit section for the respective papers.

It should be noted that non of these models are meant to be a actual prediction of flow behavior and is more of a proof of concept and demonstration of what is possible with current GNN models. Morever, the models are somewhat sensitive to the underlying mesh and ocassionally give different results under the same cylinders or other bodies used.

# Components of Repository

## Interactive GNN Window

The interactive 

Run and recorded on m2 macbook air

## Animation Handler

The animation handler allows for a non interactive but time evolving mesh to be simulated. A view of the animations is ebedded below.

![Animation](/anim/animation.gif)

Additionally, the animation handler can also produce a animation of how the mesh evolves as shown below.

![Alt Text](/anim/animation_graph.gif)

# Code Use

## Environment Packages

This codebase requires 

`$ pip install torch torch-scatter matplotlib scikit-learn pygame PyYaml`

## Running the code

For the pure pygame interface run 

`$ python main.py`

This will automatically read in the parameters.yaml file where you can modify the models and simulation parameters. If custom parameters are desired, then run the following.

`$ python main.py --parameter custom_parameters.yaml`

To run the time evolution animations run

`$ python anim_main.py`

Similar to the base pygame interface, the parameters can be modified by the parameters.yaml file (uses this file to select models) and the anim_parameters.yaml file (uses this file to create animations). If custom YAML files are desired, simply run the following.

`$ python anim_main.py --parameter custom_parameters.yaml --anim_parameter custom_anim_parameters.yaml`


## Controls:
w,a,s,d: move screen
scroll: zoom
tab: toggle mode from points to contour fields
g: recenter screen
number keys: change field type
    1: x velocity
    2: y velocity
    3: pressure
    4: learned energy (for tignn only)
    5: learned entropy (for tignn only)

m: toggle auto edge creation
n: remesh all internal nodes
control s: save state (in pickle format)
control l: save state (in pickle format)
control z: undo
control y: redo

in point mode:
    left click: create new node/move existing node
    right click: remove node/edge
    c: change node type

escape: exit

# Information

Using the base models found in the above papers, the models were modified and retrained on the vortex shedding data found in the TIGNN github. Modifications on both the training data in the TIGNN github and the models themselves were conducted, mainly consisting of converting and training the model to triangular based meshes, allowing the models to train on boundary condition cells, and a implementation of a custom flow matching gnn remesher. All together, the retrained models, custom meshing model and pygame wrapper allows for interactive session whhere the GNN models can be modified as desired. These models are relatively stable but do occasionally destabilize to Nans in ill poised meshes (which occasionally occur with the custom meshing model as well). This repo contains 4 models as follows each with their unique behaviors.

1. TIGNN trained on TIGNN dataset
2. MGN trained on TIGNN dataset with direct pressure predictions
3. MGN trained on TIGNN dataset with $\frac{dp}{dt}$ prediction
4. MGN trained on original vortex dataset with $\frac{dp}{dt}$ prediction

#### TIGNN
This model is the original framework for the repo and was the one I was most interested in. However, after modifications and training, the TIGNN model I trained seems to damen out the oscillatory behavior of the vortex shed. This could be due to the boundary condition handler and may benefit from further refinement. Unfortunately, this model also suffers a higher computational cost (due to the autograd functions) over the prior MGN networks. On a m2 macbook air, I was only able to run the interactive session with ~8 fps. As the main focus of this repo is interactive GNN project, this model is not enabled by default (can be enabled in parameters.yaml). However, this model does provide a interesting learned energy and entropy field (Note: this learned field is not necessarily traditional energy/entropy fields in fluid dyanmics but rather a learned field that corresponds to the observed state evolution).

I also had some diffculty training this model (exacerbated to the aforementioned computational cost) as some training sessions had losses that rapidly increased when using the learning rates found within the original paper. If anyone would like to see if they can get a better TIGNN working with lower computational cost and/or better shedding behavior, I would gladly include it into this repo.

#### MGN
There are 3 MGN networks included in this repo, two trained on the modified TIGNN dataset and the other trained on the original data from the MGN paper (but with a pytorch framework). The benefit of the MGN model is that these models do not use a autograd function and are hence much faster. On a m2 macbook air, the interactive section averaged around 25 fps. Additionally, seems to perform adequately in predicting vortex sheds with the current boundary condition handler. However, one thing to note is that the pressure predictions did seem a little unstable and would ocassionally give non real predictions.

Between the 3 MGN networks, the default is the MGN network trained on the TIGNN dataset with $\frac{dp}{dt}$ predictions. This is because the TIGNN dataset is much smaller and thus faster to run over the original vortex shedding dataset. Additionally, the bondary conditions and meshes are harder to create (original MGN vortex shed dataset has a boundary layer mesh which did not train well with the current flow matching mesh model). The direct pressure prediction model was also much more unstable and did not give any meaningful results before diverging to Nans.

I also had difficulty training the MGN model on the original MGN dataset.
Also the orininal MGN paper called for a direct pressure 

It should be once again noted that non of these models are meant to be a actual prediction of flow behavior and is more of a proof of concept and demonstration of what is possible with current GNN models.

mgn has difficulties with pressure, also remeshing has a odd velocity drop across inlet and next set of internal nodes
tignn seems to have a dampened vortex shed compared to training data and MGN model

mgn runs ~4x faster but with larger artifacts (~25 fps on m2 macbook air for 1000 nodes)
tignn provides better results but much slower (~8 fps on m2 macbook air for 1000 nodes)

made interactive with pygame-opengl wrapper, custom flow matching mesher, bc handler, and faster generic mesh graph net model from google deepmind


or their github with the model 

While the base model does not use the thermodyanamics-informed graph neural network for computational speed reasons, this github provided the condensed training data and pytorch framework over earlier meshgraphnet models.

[Original Mesh Graph Net Github](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)


## Modifications

Also includes scripts to generate animations in anim directory

Changed mesh from quad to triangular mesh using delaunay triangles. 
Unstructured the mesh using delaunay traingles

Baseline models created from GITHUB

modification to training to include boundary condition evolution (mainly for pressure changes in wall and inlet nodes and velocity changes in outlet nodes)     
    bad results with some simplistic bc handling (nearest pressure or velocity)
difficulty with derived mgn model in generating stable inlet conditions, model included but devolves quickly

new models trained to 1000 epochs using TIGNN data
TIGNN data remeshed using delaunay triangles to provide unstructured capabilities
tignn: 5e-4 400 milestone exponential decay
mgn: 1e-3 400 milestone exponential decay
proof of concept
added noise to node positions during training for greater robustness while moving nodes



A custom flow matching model was developed for this project. The reason for this was several fold. 
1. As the principal objective of the project was to provide a real-time interactive simulation (rather than physically accurate simulations), a dramatic variation in nodes would cause fluctuations in framerates. Using a more traditional meshing algorithm, it would be more difficult to ensure good mesh creation with a fixed number of nodes without a much more involved algorithm. 
2. I am not entirely sure how the meshes was conducted with what parameters and algorithm. Additionally, the models do seem to be somewhat sensitive to the underlying mesh and thus a different meshing algorithm could possibly cause the model become unstable. Therefore, a unsupervised method to match the target mesh distributions is more desirable to keep the model predicting on a statistically similar mesh as the training data.
3. I wanted to try creating a GNN based flow matching method to see how well it works and practice my flow matching abilities.

The flow matching GNN mesher adds noise to a already formed mesh then denoises it, using the different wall, inlet, and outlet nodes to guide the denoiser to create a new mesh. The GNN also includes a global encoding as that qualitatively seemed to improve the mesh quality

The generate mesh method initally assumes a random uniform distribution across the domain. However, in successive noisers and flow matching steps, the distribution approaches a improved mesh with mesh refinement near bodies of interest.

# Credit

TIGNN and pytorch model framework from

- Hernández, Quercus and Badías, Alberto and Chinesta, Francisco and Cueto, Elías. "[Thermodynamics-informed graph neural networks](https://ieeexplore.ieee.org/document/9787069)." IEEE Transactions on Artificial Intelligence (2022).

Mesh Graph Net paper
Substantiation of extrapolated geometries

After implementation of current flow match GNN, found paper with similar implemenation
Similar GNN Flow Matching paper using different 

FUTURE LINKS:
TRAINING GITHUB


## Notes:
- Seems somewhat sensitive to mesh generation, could maybe add more noise into training to lower propensity of node positions on broader characteristics
- Pressure maps have a tendency to not be correctly predicted yet velocities are still somewhat predicted
    - pressure maps seemed are the most inaccurate, all models seem to suffer from some pressure prediction issues
    - Maybe issue with my methods or a bug somewhere in training
- Evaluate maybe mean scatter even though fundamentally, fvm method of navier stokes tracks summation of fluxes
- Graph is node forming the edges of cells, from what I recall, fvm codes generally use node centers for most of their properties
    - GNN might be trained on interpolated data through cell boundaries rather than cells themselves depending on how the data was formulated in TIGNN github
- No boundary layer cells, unable to capture viscous effects well. However, no slip walls are implemented in training data
- Difficulty training TIGNN with boundary nodes, might be beneficial to separate out the models
- Original MGN model directly predicted pressure, had issues with that model as well