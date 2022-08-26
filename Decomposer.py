import numpy as np
import inspect
import pennylane as qml
from pennylane.wires import Wires
from pennylane.transforms.optimization.optimization_utils import find_next_gate
import QuantumOperations as q

## ------------ Functions for transpilation ------------ ##

# for further improvements:
# 1. possible commutation relations are not exploited
# 2. cumulative block operations that are close to identity are, nevertheless, translated into rotations


# execute several steps of the transpillation
def transpile(op_list):
    return decompose_single_qubit_operations(decompose_CNOT(decompose_Toffoli(decompose_to_qml(op_list))))


# decompose all operations of the QuantumOperations module into qml operations
def decompose_to_qml(list_next_ops):
    
    # list of members of the QuantumOperations module
    members_list = [member[1] for member in inspect.getmembers(q)]
    # check if there are operations from the QuantumOperations module
    q_operations_indicator = any([type(op) in members_list for op in list_next_ops])
    
    while q_operations_indicator:
        
        # go through the operations list and decompose all operations from the QuantumOperations 
        list_next_ops_ = list()
        for op in list_next_ops:

            # only QuantumOperations will be transpiled
            if type(op) in members_list:

                for dec_op in op.decomposition():
                    list_next_ops_.append(dec_op)

            else:
                list_next_ops_.append(op)
        
        # update: list_next_ops
        list_next_ops = list_next_ops_
        # update: check if there are operations from the QuantumOperations module
        q_operations_indicator = any([type(op) in members_list for op in list_next_ops])
    
    return list_next_ops


# translate 3-qubit Toffoli into CNOTs and single-qubit operations
def decompose_Toffoli(list_next_ops):
    
    op_list = list()
    
    for op in list_next_ops:
        if (type(op) == qml.Toffoli) and (len(op.wires) == 3):
            
            # Controlled-V[1,2]
            op_list.append(qml.RY(-np.pi/2,wires=op.wires[1]))
            op_list.append(qml.IsingXX(2*np.pi/8, wires=[op.wires[1], op.wires[2]]))
            op_list.append(qml.RX(-np.pi/4,wires=op.wires[1]))
            op_list.append(qml.RX(np.pi/4,wires=op.wires[2]))
            op_list.append(qml.RY(np.pi/2,wires=op.wires[1]))
            # Controlled-V[0,2]
            op_list.append(qml.RY(-np.pi/2,wires=op.wires[0]))
            op_list.append(qml.IsingXX(2*np.pi/8, wires=[op.wires[0], op.wires[2]]))
            op_list.append(qml.RX(-np.pi/4,wires=op.wires[0]))
            op_list.append(qml.RX(np.pi/4,wires=op.wires[2]))
            op_list.append(qml.RY(np.pi/2,wires=op.wires[0]))
            # CNOT[0,1]
            op_list.append(qml.CNOT(wires=[op.wires[0], op.wires[1]]))
            # Controlled-V_dagger[1,2]
            op_list.append(qml.RY(-np.pi/2,wires=op.wires[1]))
            op_list.append(qml.RX(-np.pi/4,wires=op.wires[2]))
            op_list.append(qml.RX(np.pi/4,wires=op.wires[1]))
            op_list.append(qml.IsingXX(-2*np.pi/8, wires=[op.wires[1], op.wires[2]]))
            op_list.append(qml.RY(np.pi/2,wires=op.wires[1]))
            # CNOT[0,1]
            op_list.append(qml.CNOT(wires=[op.wires[0], op.wires[1]]))
            
        else:
            op_list.append(op)
    
    return op_list


# translate CNOTs into IsingXX and single-qubit rotations
def decompose_CNOT(list_next_ops):
    
    op_list = list()
    
    for op in list_next_ops:
        if type(op) == qml.CNOT:
            op_list.append(qml.RY(np.pi/2,wires=op.wires[0]))
            op_list.append(qml.IsingXX(np.pi/2,wires=op.wires))
            op_list.append(qml.RX(-np.pi/2,wires=op.wires[1]))
            op_list.append(qml.RX(-np.pi/2,wires=op.wires[0]))
            op_list.append(qml.RY(-np.pi/2,wires=op.wires[0]))
        else:
            op_list.append(op)
    
    return op_list


# find all blocks of single-qubit operations and replace them with rotations RZ R RZ
def decompose_single_qubit_operations(list_next_ops):
    
    op_list = list()
    
    while len(list_next_ops) != 0:
        
        # take the first gate from the list_next_ops to further search blocks to merge
        op = list_next_ops[0]
        list_next_ops.pop(0)
        
        # only single-qubit operations will be transpiled
        if len(op.wires) == 1:
            
            # initialize variables before the next cycle
            cumulative_block_operation = op.matrix()
            next_gate_index = find_next_gate(op.wires,list_next_ops)
            
            # while there are single-qubit operations to merge
            while (next_gate_index is not None) and (len(list_next_ops[next_gate_index].wires) == 1):
                
                # update cumulative_block_operation
                cumulative_block_operation = list_next_ops[next_gate_index].matrix().dot(cumulative_block_operation)
                
                # exclude the merged gate from the list and update next_gate_index
                list_next_ops.pop(next_gate_index)
                next_gate_index = find_next_gate(op.wires,list_next_ops)
                
            # when there are no more single-qubit operations to merge, translate the block into rotations
            decomposition_angles = decompose(cumulative_block_operation)
#             op_list.append(q.R(*decomposition_angles, wires=op.wires))
            op_list.append(q.R([decomposition_angles[3]/np.pi, decomposition_angles[2]/np.pi, 0], wires=op.wires))
            op_list.append(q.R([decomposition_angles[1]/np.pi, decomposition_angles[0]/np.pi, 0], wires=op.wires))
            
        else:
            op_list.append(op)
    
    return op_list


# auxiliary function
def theta(a):
    a = complex(a)
    return np.imag(np.log(a/abs(a)))


# analytic decomposition function
def decompose(U):
    
    # exceptions: b = 0 or pi/2
    if complex(U[1][0]) == complex(0):
        b = 0.
        a = theta(U[0][0])
        c = 0.
        return [-np.pi, -c - np.pi/2, 2*b + np.pi, a - c - np.pi/2]
    if complex(U[0][0]) == complex(0):
        b = np.pi/2
        a = 0.
        c = theta(U[1][0])
        return [-np.pi, -c - np.pi/2, 2*b + np.pi, a - c - np.pi/2]
    
    # 'íf' to handle nans in beta due to precision error:
    if abs(U[0][0]) > 1:
        b = np.arccos(1.)
    else:
        b = np.arccos(abs(U[0][0]))
    a = (theta(U[0][0]) - theta(U[1][1]))/2
    c = (theta(U[0][0]) - 2*theta(U[1][0]) + theta(U[1][1]))/2 - np.pi
    
    return [-np.pi, -c - np.pi/2, 2*b + np.pi, a - c - np.pi/2]


# function to find depth of the circuit
def levels(wires, op_list_transpiled):
    
    # remove 1-qubit operations R from the op_list_transpiled
    op_list_transpiled_2q = [op for op in op_list_transpiled if type(op) == qml.IsingXX]
    
    # initialize list of maximum level counts per wire
    wires_max_levels = [0 for i in range(len(wires))]
    
    # every op increases levels in wires_max_levels for operation's wires
    for op in op_list_transpiled_2q:
        lowest_level = max(wires_max_levels[op.wires[0]],wires_max_levels[op.wires[1]])
        wires_max_levels[op.wires[0]] = lowest_level + 1
        wires_max_levels[op.wires[1]] = lowest_level + 1
    
    return max(wires_max_levels)