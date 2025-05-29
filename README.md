# Interactive Graph Neural Networks for Vortex Shedding 

This repo contains a Pygame-opengl framework for a interactive graph neural network (GNN) simulation of vortex shedding around a cylinder. This work is heavily inspired by Hernández, Badías, Chinesta and Cueto in their Thermodynamics-informed graph neural networks (TIGNN) paper ([link to github](https://github.com/quercushernandez/ThermodynamicsGNN/tree/main)) as well as earlier work in Mesh Graph Nets (MGN) by Pfaff, Fortunato, Sanchez-Gonzalez, and Battaglia ([link to github](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)). For more information about the models, check the credit section for the respective papers. 

GNN initially captured my attention due to their highly adaptable structure which allows for finer meshes in regions of importance over more traditional image convolutional based approaches. This is of particular usefulness in fluid applications where much of a mesh may have large regions of bulk flow which do not spatially vary signifcantly (which would allow for coarser meshes) but are still nonetheless important to the broader development of local behaviors of interest. Additionally, these types of models naturally lend themselves to adaptive mesh refinement for efficient fluid simulations in more complex fluid simulations (combustion, hypersonics, etc). After reading the papers above, I wanted to create a interactive build of the GNN models with on line simulations to demonstrate this adapativeness and capabilities in a interactive way. However, both of the above papers had gaps in the generation of the meshes and handling of boundary conditions for vortex shedding, so I developed customized models to also generate boundary conditions and new meshes. More information is available in later sections of this readme on the customized models.

It should be noted that non of these models are meant to be a actual prediction of flow behavior and is more of a proof of concept and demonstration of what is possible with current GNN models. Morever, the models are somewhat sensitive to the underlying mesh and ocassionally give different results under the same boundary conditions.

# Components of Repository

## Interactive GNN Application

The interactive pygame wrapper consists of two modes, a graph editing mode and a contour visualization mode. In the graph editing mode, the user can interface with the underlying graph and modify it to a desired state. The new state can also be saved as a pickle file and reloaded as needed. The graph editing mode allows users to move nodes, change the node types, and remesh the field as desired. The contour visualization mode allows the fluid properties to be observed much easier than the graph representation. Below is a recording of the interactive GNN application using the default model.

Run and recorded on m2 macbook air.
![Screen Capture](/anim/interactive_demo.gif)

## Animation Handler

The animation handler allows for a non interactive but time evolving mesh to be simulated. A view of the generated animations is ebedded below.

![Animation](/anim/rotating_plate_anim.gif)
![Animation](/anim/animation.gif)

Additionally, the animation handler can also produce a animation of how the mesh evolves as shown below.

![Alt Text](/anim/animation_graph.gif)

# Code Use

## Requirements

This codebase requires the following packages
- pytorch
- pytorch-scatter
- pytorch-geometric
- matplotlib
- sklearn
- pygame
- pyyaml
- pyopengl

They can be installed using the following commands

`$ pip install torch matplotlib scikit-learn pygame pyyaml PyOpenGL PyOpenGL_accelerate`

Pytorch specific dependencies only installs after pytorch is installed

`$ pip install torch-scatter torch-geometric`

## Running the code

For the pure pygame interface run 

`$ python main.py`

This will automatically read in the parameters.yaml file where you can modify the models and simulation parameters. If custom parameters are desired, then run the following.

`$ python main.py --parameter custom_parameters.yaml`

To run the time evolution animations run

`$ python anim_main.py`

Similar to the base pygame interface, the parameters can be modified by the parameters.yaml file (uses this file to select and setup models) and the anim_parameters.yaml file (uses this file to create animations). If custom YAML files are desired, simply run the following.

`$ python anim_main.py --parameter custom_parameters.yaml --anim_parameter custom_anim_parameters.yaml`


## Controls for Interactive Application
w,a,s,d: move screen  
scroll: zoom  
tab: toggle mode from points to contour fields  
g: recenter screen 
space bar: toggle model simulation  
number keys: change field type  
    1: x velocity
    2: y velocity
    3: pressure
    4: learned energy (for tignn only)
    5: learned entropy (for tignn only)

m: toggle auto edge creation  
n: remesh all internal nodes using flow matching model 
control s: save state (in pickle format)  
control l: load state (in pickle format)  
control z: undo 
control y: redo 

in point mode:
    left click: create new node/move existing node  
    right click: remove node/edge 
    c: change node type 
        black: internal fluid node 
        blue: inlet node 
        orange: outlet node 
        green: wall node (both exterior and of body of interest) . 

escape: exit 

# Information

Using the base models found in the above papers, the models were modified and retrained on the vortex shedding data found in the TIGNN github. Modifications on both the training data in the TIGNN github and the models themselves were conducted, mainly consisting of converting and training the model to triangular based meshes, allowing the models to train on boundary condition nodes, and a implementation of a custom flow matching gnn remesher.  All together, the retrained models, custom meshing model and pygame wrapper allows for interactive session whhere the GNN models can be modified as desired. These models are relatively stable but do occasionally destabilize to Nans in ill poised meshes (which occasionally occur with the custom meshing model as well). This repo contains 4 models as follows each with their unique behaviors.

1. TIGNN trained on TIGNN dataset
2. MGN trained on TIGNN dataset with direct pressure predictions
3. MGN trained on TIGNN dataset with $\frac{dp}{dt}$ prediction
4. MGN trained on original vortex dataset with $\frac{dp}{dt}$ prediction

#### TIGNN
   This model is the original framework for the repo and was the one I was most interested in. However, after modifications and training, the TIGNN model I trained seems to damen out the oscillatory behavior of the vortex shed. This could be due to the boundary condition handler and may benefit from further refinement. Unfortunately, this model also suffers a higher computational cost (due to the autograd functions) over the prior MGN networks. On a m2 macbook air, I was only able to run the interactive session with ~8 fps. As the main focus of this repo is interactive GNN project, this model is not enabled by default (can be enabled in parameters.yaml). However, this model does provide a interesting learned energy and entropy field (Note: this learned field is not necessarily traditional energy/entropy fields in fluid dyanmics but rather a learned field that corresponds to the observed state evolution).

   I also had some diffculty training this model (exacerbated to the aforementioned computational cost) as some training sessions had losses that rapidly increased when using the learning rates found within the original paper. If anyone would like to see if they can get a better TIGNN working with lower computational cost and/or better shedding behavior, I would gladly include it into this repo.

#### MGN
   There are 3 MGN networks included in this repo, two trained on the modified TIGNN dataset and the other trained on the original data from the MGN paper (but with a pytorch framework). The benefit of the MGN model is that these models do not use a autograd function and are hence much faster. On a m2 macbook air, the interactive section averaged around 24 fps. Additionally, seems to perform adequately in predicting vortex sheds with the current boundary condition handler. However, one thing to note is that the pressure predictions did seem a little unstable and would ocassionally give non real predictions.

   Between the 3 MGN networks, the default is the MGN network trained on the TIGNN dataset with $\frac{dp}{dt}$ predictions due to the condensed TIGNN dataset which is faster to run and inference. Additionally, the bondary conditions and meshes are harder to recreate (original MGN vortex shed dataset has a boundary layer mesh which did not train well with the current flow matching mesh model). The $\frac{dp}{dt}$ MGN model was used as opposed to the the direct pressure prediction model since the direct pressure model is more unstable and did not give any meaningful results before diverging to Nans.

   Similar to the TIGNN models, there were some difficulties training on the original MGN dataset (once again exacerbated by the slow training but due to the large amount of data). By monitoring the losses and learning rate, a adquate model was able to be trained.

It should be once again noted that non of these models are meant to be a actual prediction of flow behavior and is more of a proof of concept and demonstration of what is possible with current GNN models.

## Modifications to Original Framework

As mentioned in prior sections, some modifications to the model was performed to create a more flexible framework that allows for greater interactive capabilities. One of the main adjustments was in the way how the models were trained. The original MGN and TIGNN models only trained the models on the internal fluid nodes and prescribed preset boundary conditions. Since these prescribed boundary conditions are unavailable for new modified geometries, a new boundary condition handler was created to generate new boundary conditions on line. Additionally, the GNN models were allowed to train on the evolution of the boundary conditions themselves. This is especially important in creating realistic pressures for the inlet and walls (as these were velocity inlet and no slip wall conditions respectively) as well as creating relistic velocitys at the outlet (pressure outlet condition). Another key modification was the modification of the training data mesh to be based on delaunay triangles rather than quad based mesh found in the training datset. This greatly simplifies the automatic edge creation and mesh creation methods over a quad based meshing algorithm and allows more unstructed geometries to be tested. These modifications in combination with a few more minor modifications (added noise to node positions to make models more invariant to slight changes in position, changed position normalization to normalize the difference between positions rather than the positions themselves) allow the model to create its own boudnary conditions and model fluid like simulation behaviors.

As mentioned before this code also contains a pygame wrapper and a animation handler. The pygame wrapper uses pygame to interface with the user and opengl to display the models. Originally, pygame was used for both but found to be too slow in displaying the thousands of lines in the graph representation so opengl shaders was a better alternative. As for the animation handler, the animations are handled by matplotlib. However, the tedium of creating a new mesh for each geometry was a time consumer processess by moving each node individually. This is exacerbated by a time evolving mesh where multiple meshes are created. Thus, a custom flow matching model for unsupervised mesh creation was created to quickly create new meshes

The reason a custom flow matching mesher was devloped for this project was several fold. 
1. As the principal objective of the project was to provide a real-time interactive simulation (rather than purely physically accurate simulations), a dramatic variation in nodes would cause fluctuations in framerates. Using a more traditional meshing algorithm, it would be more difficult to ensure good mesh creation with a fixed number of nodes without a much more involved algorithm. 
2. I am not entirely sure how the meshes was generated with what parameters and algorithm. Additionally, the models do seem to be somewhat sensitive to the underlying mesh and thus a different meshing algorithm could possibly cause the model to deviate from its trained data and possibly become unstable. Therefore, a unsupervised method to match the target mesh distributions is more desirable to keep the model predicting on a statistically similar mesh as the training data.
3. I wanted to try creating a GNN based flow matching method to see how well it works and practice my flow matching abilities.

The flow matching GNN mesher adds noise to a already formed mesh then denoises it, using the different wall, inlet, and outlet nodes to guide the denoiser to create a new mesh. The GNN also includes a global encoding as that qualitatively seemed to improve the mesh quality. A separate corrector model was also attempted but did not reasonably improve the mesh so it has been disabled in the paramters.yaml file.

The mesh generatation method generates a mesh from scratch from a initally assumed random uniform distribution across the domain. However, in successive noisers and flow matching steps, the distribution approaches a improved mesh with mesh refinement near bodies of interest. Note that no normalization occurs in the flow matching mesh model. This may be something to improve on in a later date to give better scaling capabilities. I have not tried the flow matching mesh model on dramatically different domains and node counts.

# Credit

TIGNN and pytorch model framework from
@article{Hern_ndez_2024,
   title={Thermodynamics-Informed Graph Neural Networks},
   volume={5},
   ISSN={2691-4581},
   url={http://dx.doi.org/10.1109/TAI.2022.3179681},
   DOI={10.1109/tai.2022.3179681},
   number={3},
   journal={IEEE Transactions on Artificial Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Hernández, Quercus and Badías, Alberto and Chinesta, Francisco and Cueto, Elías},
   year={2024},
   month=mar, pages={967–976} }

[Link to Github](https://github.com/quercushernandez/ThermodynamicsGNN/tree/main)

Original Mesh Graph Net Paper (default model)
@inproceedings{pfaff2021learning,
  title={Learning Mesh-Based Simulation with Graph Networks},
  author={Tobias Pfaff and
          Meire Fortunato and
          Alvaro Sanchez-Gonzalez and
          Peter W. Battaglia},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

[Link to Github](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)

Justification and substantiation of generality of mesh graph nets
@misc{schmöcker2024generalizationcapabilitiesmeshgraphnetsunseen,
      title={Generalization capabilities of MeshGraphNets to unseen geometries for fluid dynamics}, 
      author={Robin Schmöcker and Alexander Henkes and Julian Roth and Thomas Wick},
      year={2024},
      eprint={2408.06101},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.06101}, 
}

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
- Possible future implementation of some metric that rejects badly created mesh and tries to create a new one (using similar methods as new n method in solver to find if points within inner boundary or out of outer boundary to reject mesh, and/or using some skew metrics (min,max,std,mean)).

models trained to 1000 epochs using TIGNN data
TIGNN data remeshed using delaunay triangles to provide unstructured capabilities
tignn: learning rate of 5e-4 with 400 milestone exponential decay rate of 0.1
mgn: learning rate of 1e-3 with 400 milestone exponential decay rate of 0.1 (both direct and derivitive models)
mgn on mgn dataset: difficulty training, 25 epochs with 1e-4 learning rate with exponential decay rate of 0.9999991 each epoch (taken from nvidia sugggestion [here](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/examples/cfd/vortex_shedding_mgn/readme.html))