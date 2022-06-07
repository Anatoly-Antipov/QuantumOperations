# QuantumOperations
This package is developed for the efficient realization of the quantum part of Shor's algorithm, namely, order finding procedure. The package makes use of quantum premitives described in ‘Quantum Networks for Elementary Arithmetic Operations’ - Vedral, Barenco, Ekert, 1995

Files description:

1. **QuantumOperations.py** contains a set of quantum gates and algorithms. Any quantum gate or algorithm is realized as a subclass of the PennyLane's Operation class

2. **ClassicalOperations.py** contains a class of auxiliary classical functions supporting quantum computations

3. **Decomposer.py** contains functions for transpilation in accordance with simplified protocol from ‘Basic circuit compilation techniques for an ion-trap quantum machine’ - Maslov, 2017

4. **Example.ipynb** - Jupyter notebook with an example of usage of the SUM gate from QuantumOperations.py

5. **ExampleOrderFinding.ipynb** - Jupyter notebook with an example of usage of the OrderFinding gate from QuantumOperations.py

6. **Test.ipynb** - Jupyter notebook with various tests of the realized quantum gates and algorithms

7. **ExampleDecomposer.ipynb** - Jupyter notebook with a demo of decomposer

8. **OrderFindingResourceAnalysis.ipynb** - Jupyter notebook with resource analysis for order finding using decomposer