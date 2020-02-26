# qTEBD

qTEBD is the algorithm combining imaginary time evolution and fidelity maximization. We benchmark such algorithm on MPS and show that it can be generalized for optimizing quantum circuit.

## qTEBD with MPS

The red line indicate the DMRG result. qTEBD with MPS match with DMRG with same bond dimension exactly when 2nd-order trotter gate is applied.
 
### TFI g = 1.1
L 10 1st order trotter       |  L 10 2nd order trotter
:---------------------------:|:-------------------------:
![](figure/TFI/finite_L10_g1.1_1st.png)   |  ![](figure/TFI/finite_L10_g1.1_2nd.png)

### TFI g = 1.5
L 10 1st order trotter       |  L 10 2nd order trotter
:---------------------------:|:-------------------------:
![](figure/TFI/finite_L10_g1.5_1st.png)   |  ![](figure/TFI/finite_L10_g1.5_2nd.png)


## qTEBD with circuit

We consider quantum circuit of 2-site gates, where the gates is arranged in the ladder form in each layer. Such n-layer circuit forms a subset of MPS of bond dimension 2^n. Contrary to the equivalence between MPS of bond dimension 2^n and quantum circuit of n-qubit gates, layers of 2-site gates could build up entanglement quickly with less parameters comparing to MPS. It is clear by parameter counting that n-layer circuit would have O(n) parameters while MPS of bond dimension 2^n has O(exp(n)) parameters. Thus, it can be thought of as a sparse representation, which is a subset of MPS with unknown variation power. 


### TFI g = 1.0
L 10 1st order trotter       |  L 10 2nd order trotter
:---------------------------:|:-------------------------:
![](figure/TFI/circuit_L10_g1.0_1st.png)   |  ![](figure/TFI/circuit_L10_g1.0_2nd.png)


### TFI g = 1.5
L 10 1st order trotter       |  L 10 2nd order trotter
:---------------------------:|:-------------------------:
![](figure/TFI/circuit_L10_g1.5_1st.png)   |  ![](figure/TFI/circuit_L10_g1.5_2nd.png)

### XXZ Jz = 1.
L 10 1st order trotter       |  L 10 2nd order trotter
:---------------------------:|:-------------------------:
![](figure/XXZ/circuit_L10_g1.0_1st.png)   |  ![]()




### Single-layer circuit
### n-layer circuit


* There are different way to optimize n-layer circuit. Here we consider to randomly initialize 2-layer and optimize Niter time per time step.
* With near identity initialization and change Hamiltonian basis.


## TODO
* plot num para v.s. accuracy
* Scaling with Niter
* The effect in the pattern of the layer of gates

#### Note
* Does not seem to work : Iterative optimization for each layer. Optimize first layer and fix first layer. Then add second layer and optimize second layer and so on.


