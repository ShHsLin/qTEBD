# qTEBD

qTEBD is the algorithm for real or imaginary time evolution by fidelity maximization. We benchmark such algorithm on MPS for ground state search and quench problem. Then, we show that the algorithm can be generalized to classical simulation of quantum circuit.


### Requirement

numpy, jax, (TeNPy)


## qTEBD for Ground state with MPS and Quantum Circuit

See directory "1 ground state"

## Quantum Quench

See directory "2 time evolution"

### Single-layer circuit

Single-layer circuit should match with the result of qTEBD with MPS and DMRG with MPS of bond dimension 2. Indeed we see that. Optimization for single-layer circuit is benign and could be easily solved with one iteration/sweep per time step.

### n-layer circuit

* There are different way to optimize n-layer circuit. Here we consider to randomly initialize n-layer and optimize Niter time per time step.
* With near identity initialization and change Hamiltonian basis.


## TODO
* Rerun Ground state; Re plot optimization curve; Re plot num para v.s. accuracy
* MPS truncation for gate contraction; logging truncation error in gate contraction; rerun all data;
* Scaling with Niter
* The effect in the pattern of the layer of gates

#### Note
* Does not seem to work : Iterative optimization for each layer. Optimize first layer and fix first layer. Then add second layer and optimize second layer and so on.
* Gradient descent for optimizing fedility of qTEBD-MPS gives very small improvement on top of polar decomposition. Gradient descent itself alone does not work.

* In time evolution, polar sweep along the isometry direction fails.


