# QuantumOperations
This package is developed for the efficient realization of the quantum part of Shor's algorithm, namely, order finding procedure. The package makes use of quantum premitives described in ‘Quantum Networks for Elementary Arithmetic Operations’ - Vedral, Barenco, Ekert, 1995

**NOTE**: There is a relevant paper https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0271462 that contains the realization for pennylane 0.20.0. In the current version 0.25.1, pennylane's developers used slightly different static method for decompositions which leads to errors while executing the code from the paper. This repository is updated to match version 0.25.1.

Files description:

1. **QuantumOperations.py** contains a set of quantum gates and algorithms. Any quantum gate or algorithm is realized as a subclass of the PennyLane's Operation class

2. **ClassicalOperations.py** contains a class of auxiliary classical functions supporting quantum computations

3. **Decomposer.py** contains functions for transpilation in accordance with simplified protocol from ‘Basic circuit compilation techniques for an ion-trap quantum machine’ - Maslov, 2017

4. **Example.ipynb** - Jupyter notebook with an example of usage of the SUM gate from QuantumOperations.py

5. **ExampleOrderFinding.ipynb** - Jupyter notebook with an example of usage of the OrderFinding gate from QuantumOperations.py

6. **Test.ipynb** - Jupyter notebook with various tests of the realized quantum gates and algorithms

7. **ExampleDecomposer.ipynb** - Jupyter notebook with a demo of decomposer

8. **OrderFindingResourceAnalysis.ipynb** - Jupyter notebook with resource analysis for order finding using decomposer

NOTE: library was built using pennylane version 0.20.0
