# qTEBD

## qTEBD with MPS
 
![](figure/finite_L10.png)

## qTEBD with circuit

### Single-layer circuit
![](figure/circuit_L10_depth1.png)

### n-layer circuit

* There are different way to optimize n-layer circuit. Here we consider to randomly initialize 2-layer and optimize Niter time per time step.
 
![](figure/circuit_L10_depth2.png)


* With near identity initialization and change Hamiltonian basis.
![](figure/sinit_circuit_L10_depth2.png)


## TODO
* Iterative optimization for each layer. Optimize first layer and fix first layer. Then add second layer and optimize second layer and so on.
* Scaling with Niter
* The effect in the pattern of the layer of gates


