import numpy as np
import pennylane as q
from sympy import Matrix
import copy


#-------------------------------------#
#--- Classical auxiliary functions ---#
#-------------------------------------#

class ClassicalOperations:
    
    # checks if matrix U is in pennylane.Variable format and makes U proper format if necessary
    def check_matrix(self,U):
        # if U is passed as pennylane.Variable, then use type np.complex128
        if isinstance(U[0][0], q.variable.Variable):
            # no straight-forward't matrix-wise operation .val for ndarray of pennylane.Variables => use cycle
            for i in range(U.shape[0]):
                for j in range(U.shape[0]):
                    U[i][j] = U[i][j].val
            # after the cycle, U has dtype = object. Replace it with dtype = complex128
            U = U.astype(dtype = 'complex128')
        return U
    
    # prints out list of states in a computational basis
    def states_vector(self,wires):
        states_vector = list()
        for i in range(2**len(wires)):
            states_vector.append('|'+'0'*(len(wires)-len(bin(i)[2:]))+bin(i)[2:]+'>')
        return states_vector
    
    
    def get_non_trivial_indices(self,Two_level_U_list):
        # due to algorithmic structure of decomposition, non_trivial_indices are known in advance
        non_trivial_indices_list = list()
        k = 0
        matrix_size = Two_level_U_list[0].shape[0]
        while k < matrix_size:
            for i in range(k+1,matrix_size):
                non_trivial_indices_list.append(list([k,i]))
            k+=1
        # from list of two lists (Two_level_U_list and non_trivial_indices_list) make list of pairs [matrix, non_trivial_indices]
        decomposition_pairs_list = [[Two_level_U_list[i],non_trivial_indices_list[i]] for i in range(len(Two_level_U_list))]
        return decomposition_pairs_list


    # creates list with Gray code and changing bits for every step
    def Gray_code(self,a,b,n):
    
        # add zeros to the left
        a = '0'*(n - len(a)) + a
        b = '0'*(n - len(b)) + b

        # get Gray code
        gray_code = list([a])
        changing_bit = list()
        for i in range(max(len(a),len(a))-1,0-1,-1):
            if a[i] != b[i]:
                # a[i] = b[i]
                a = a[:i]+b[i]+a[i+1:]
                gray_code.append(a)
                changing_bit.append(i)

        return [gray_code, changing_bit]

    
    # 2x2 matrix to the custom power
    def matrix_power(self, U, power=1/2):
        
        if U.shape[0] != 2:
            raise Exception('U should be of size 2x2')
        
        ### Transform U and power
        U = ClassicalOperations.check_matrix(self,U)
        # if power is passed as pennylane.Variable, then use type np.complex128
        if isinstance(power, q.variable.Variable):
            power = power.val
        
        
        # get eigenvectors and eigenvalues
        e = Matrix(U).eigenvects()
        # if algebraic multiplicity of the first eigenvalue is 1, then there are two elements in e with distinct eigenvectors and eigenvalues
        if e[0][1] == 1:
            l0 = np.array(e[0][0]).astype(np.complex128)
            l1 = np.array(e[1][0]).astype(np.complex128)
            h0 = np.array(e[0][2][0].T).astype(np.complex128)
            h0 = h0/np.sqrt(h0.dot(np.conj(h0.T)))
            h1 = np.array(e[1][2][0].T).astype(np.complex128)
            h1 = h1/np.sqrt(h1.dot(np.conj(h1.T)))
        # if algebraic multiplicity of the first eigenvalue is 2 (for instance, for U=I), then there is one element in e with two eigenvectors having similar eigenvalue
        if e[0][1] == 2:
            l0 = np.array(e[0][0]).astype(np.complex128)
            l1 = np.array(e[0][0]).astype(np.complex128)
            h0 = np.array(e[0][2][0].T).astype(np.complex128)
            h0 = h0/np.sqrt(h0.dot(np.conj(h0.T)))
            h1 = np.array(e[0][2][1].T).astype(np.complex128)
            h1 = h1/np.sqrt(h1.dot(np.conj(h1.T)))
        
        return np.power(l0,power)*(h0.T).dot(np.conj(h0)) + np.power(l1,power)*(h1.T).dot(np.conj(h1))
    
    
    def matrix_natural_power(self,U,power):
        
        # check input
        if (int(power) != power) | (power < 1):
            raise Exception('power should be a natural number')
        
        # raise U to the power of 'power'
        if power == 1:
            return U
        else:
            V = U.dot(U)
            for i in range(1,power-1):
                V = V.dot(U)
            return V
    
    
    # Given 2x2 matrix, function returns angles of ZY decomposition
    def ZY_decomposition_angles(self, U):
        
        if U.shape[0] != 2:
            raise Exception('U should be of size 2x2')
        
        ### Transform U
        U = ClassicalOperations.check_matrix(self,U)
        
        ### Computations
        # if U doesn't contain 0's, then apply general method
        if (U[1][1] != 0)&(U[1][0] != 0):
            # alpha
            alpha = np.imag(np.log(U[0][0]*U[1][1] - U[0][1]*U[1][0])) / 2
            # beta and delta
            phi0 = -np.imag(np.log(U[0][0] / U[1][1])) / 2
            phi1 = -np.imag(np.log(-U[0][1] / U[1][0])) / 2
            beta = phi0 + phi1
            delta = phi0 - phi1
            # gamma
            cos_gamma_halved = np.real(U[0][0] / np.exp(1j*(alpha-beta/2-delta/2)))
            sin_gamma_halved = np.real(-U[0][1] / np.exp(1j*(alpha-beta/2+delta/2)))
            cos_sign = cos_gamma_halved >= 0
            sin_sign = sin_gamma_halved >= 0
            if (cos_sign & sin_sign):
                gamma = 2*np.arccos(cos_gamma_halved)
            if ((cos_sign == False) & sin_sign):
                gamma = 2*np.arccos(cos_gamma_halved)
            if (cos_sign & (sin_sign == False)):
                gamma = 2*np.arcsin(sin_gamma_halved)
            if ((cos_sign == False) & (sin_sign == False)):
                gamma = -2*np.arccos(cos_gamma_halved)

        # if U contains 0's, then there are 2 options: U[0][0] = U[1][1] = 0 or U[0][1] = U[1][0] = 0
        # beta and delta are not unique in this case, so we set beta = 2*phi1 and delta = 0
        if (U[1][1] == 0):
            # aplha
            alpha = np.imag(np.log(-U[0][1] + 0j) + np.log(U[1][0] + 0j)) / 2 # '+0j' in order to make argument for np.log COMPLEX
            # beta and delta
            phi1 = -(np.imag(np.log(-U[0][1] + 0j) - np.log(U[1][0] + 0j))) / 2 # '+0j' in order to make argument for np.log COMPLEX
            beta = 2*phi1
            delta = 0
            # gamma
            gamma = np.pi
        # beta and delta are not unique in this case, so we set beta = 2*phi0 and delta = 0
        if (U[1][0] == 0):
            # aplha
            alpha = np.imag(np.log(U[0][0]*U[1][1] + 0j)) / 2 # '+0j' in order to make argument for np.log COMPLEX
            # beta and delta
            phi0 = -np.imag(np.log(U[0][0] / U[1][1] + 0j)) / 2 # '+0j' in order to make argument for np.log COMPLEX
            beta = 2*phi0
            delta = 0
            # gamma
            gamma = 0

        return {'alpha':alpha, 'beta':beta, 'delta':delta, 'gamma':gamma}
    
    # Given angles of ZY decomposition, function returns corresponding 2x2 matrix
    def U_given_ZY_angles(self, alpha, beta, gamma, delta):
        
        # if angles are passed as pennylane.Variable, then use proper format
        if isinstance(alpha, q.variable.Variable):
            alpha = alpha.val
        if isinstance(beta, q.variable.Variable):
            beta = beta.val
        if isinstance(gamma, q.variable.Variable):
            gamma = gamma.val
        if isinstance(delta, q.variable.Variable):
            delta = delta.val
        
        RZ_beta = np.array([[np.exp(-1j*beta/2),0],
                            [0,np.exp(1j*beta/2)]])
        RY_gamma = np.array([[np.cos(gamma/2),-np.sin(gamma/2)],
                            [np.sin(gamma/2),np.cos(gamma/2)]])
        RZ_delta = np.array([[np.exp(-1j*delta/2),0],
                            [0,np.exp(1j*delta/2)]])
        return np.exp(1j*alpha)*RZ_beta.dot(RY_gamma).dot(RZ_delta)
    
    
    # Given U, function returns its two-level unitary decomposition
    # Note that U = decomposition_list[0]*...*decomposition_list[n-1]
    def Two_level_unitary_decomposition(self,U):
        
        ### Transform U
        U = ClassicalOperations.check_matrix(self,U)
        
        if U.shape[0] == 2:
            decomposition_list = list([U])
        
        else:

            V_list = list([U])

            # find matrices V_k such that V_n*...*V_1*U0 = I
            for i in range(1,U.shape[0]):
                if np.isclose(U[i][0], 0, atol=1e-15):
                    V = np.eye(U.shape[0],dtype='complex128')
                    V[0][0] = np.conj(U[0][0])
                else:
                    V = np.eye(U.shape[0],dtype='complex128')
                    V[0][0] = np.conj(U[0][0]) / np.sqrt(U[0][0]*np.conj(U[0][0]) + U[i][0]*np.conj(U[i][0]))
                    V[0][i] = np.conj(U[i][0]) / np.sqrt(U[0][0]*np.conj(U[0][0]) + U[i][0]*np.conj(U[i][0]))
                    V[i][0] = U[i][0] / np.sqrt(U[0][0]*np.conj(U[0][0]) + U[i][0]*np.conj(U[i][0]))
                    V[i][i] = -U[0][0] / np.sqrt(U[0][0]*np.conj(U[0][0]) + U[i][0]*np.conj(U[i][0]))

                V_list.append(V)
                U = V.dot(U)
            V_list.append(np.conj(U).T)
            
            # if True, then decomposed all - no deligation
            if U.shape[0] == 3:
                # rearrange matrices in a way that U0 = np.conj(V_1.T)*...*(np.conj(V_n.T))
                decomposition_list = list()
                for i in range(1,len(V_list)):
                    decomposition_list.append(np.conj(V_list[i].T))

            # recursuive element - if True, then decomposed n-1 and deligated decomposition of the last matrix V_n
            if U.shape[0] > 3:
                
                # rearrange n-1 matrices in a way that U0 = np.conj(V_1.T)*...*(np.conj(V_(n-1).T))
                decomposition_list = list()
                for i in range(1,len(V_list)-1):
                    decomposition_list.append(np.conj(V_list[i].T))
                
                # the last matrix V_n should be further decomposed - append result of the decomposition
                deligated_task = ClassicalOperations.Two_level_unitary_decomposition(self,np.conj(V_list[-1][1:U.shape[0]:1, 1:U.shape[0]:1].T))
                for j in range(len(deligated_task)):
                    deligated_result = np.eye(U.shape[0],dtype='complex128')
                    deligated_result[1:U.shape[0]:1, 1:U.shape[0]:1] = deligated_task[j]
                    decomposition_list.append(deligated_result)
        
        return decomposition_list
        
    
#---------------------------------------------#
#--- Functions impplementing quantum gates ---#
#---------------------------------------------#

class QuantumGates:
    
    # Note that wires should NOT be passed to QNode's function as an argument, since there are 2 problems:
    # 1. If type of wires in device is int and wires are passed as an argument for QNode's function, then wires[i].val type inside the function will be float => error 
    # 2. If type of wires in device is str, then passing str type as an argument for QNode's function will raise error
    
    
    # Implements T_dagger (conjugate transform of T)
    def T_dagger(self,wires):
        q.S(wires=wires)
        q.S(wires=wires)
        q.S(wires=wires)
        q.T(wires=wires)

    
    # Implements standard 2-wires SWAP-gate
    def SWAP(self,wires):
        q.CNOT(wires=[wires[0],wires[1]])
        q.CNOT(wires=[wires[1],wires[0]])
        q.CNOT(wires=[wires[0],wires[1]])
    
    
    # Implements Toffoli gate using Hadamard, phase, controlled-NOT and π/8 gates (Nielsen, Chuang)
    def Toffoli(self,wires):
        q.Hadamard(wires=wires[2])
        q.CNOT(wires=[wires[1],wires[2]])
        QuantumGates.T_dagger(self,wires=wires[2])
        q.CNOT(wires=[wires[0],wires[2]])
        q.T(wires=wires[2])
        q.CNOT(wires=[wires[1],wires[2]])
        QuantumGates.T_dagger(self,wires=wires[2])
        q.CNOT(wires=[wires[0],wires[2]])
        QuantumGates.T_dagger(self,wires=wires[1])
        q.T(wires=wires[2])
        q.Hadamard(wires=wires[2])
        q.CNOT(wires=[wires[0],wires[1]])
        QuantumGates.T_dagger(self,wires=wires[1])
        q.CNOT(wires=[wires[0],wires[1]])
        q.T(wires=wires[0])
        q.S(wires=wires[1])
        
    
    # Implements 4-wires carry operation used for ADDER
    # setup: wires[0] = c_i, wires[1] = a_i, wires[2] = b_i, wires[3] = c_(i+1) = |0>
    # operation carries |1> in wires[3] = c_(i+1) if c_i + a_i + b_i > 1
    # Based on Vedral, Barenco, Ekert - "Quantum Networks for Elementary Arithmetic Operations", 1996
    def CARRY(self,wires):
        QuantumGates.Toffoli(self,wires=wires[1:])
        q.CNOT(wires=[wires[1],wires[2]])
        QuantumGates.Toffoli(self,wires=[wires[0],wires[2],wires[3]])
        
    
    # Implements 3-wires carry operation used for ADDER
    # setup: wires[0] = a, wires[1] = b, wires[2] = |0>
    # operation makes wires[2] = a+b mod 2
    # Based on Vedral, Barenco, Ekert - "Quantum Networks for Elementary Arithmetic Operations", 1996
    def SUM(self,wires):
        q.CNOT(wires=[wires[1],wires[2]])
        q.CNOT(wires=[wires[0],wires[2]])
    
    
    # C_1_U block for 2x2 U to be inserted into the main circuit
    # Implements controlled-U given angles from ZY-decomposition of U
    def Controlled_U_block(self,alpha,beta,gamma,delta,delta_plus_beta,delta_minus_beta,wires):
        
        # if variables are passed as pennylane.Variable, then use proper format
        if isinstance(alpha, q.variable.Variable):
            alpha = alpha.val
        if isinstance(beta, q.variable.Variable):
            beta = beta.val
        if isinstance(gamma, q.variable.Variable):
            gamma = gamma.val
        if isinstance(delta, q.variable.Variable):
            delta = delta.val
        if isinstance(delta_plus_beta, q.variable.Variable):
            delta_plus_beta = delta_plus_beta.val
        if isinstance(delta_minus_beta, q.variable.Variable):
            delta_minus_beta = delta_minus_beta.val
        
        q.RZ((delta_minus_beta)/2,wires=wires[1])
        q.CNOT(wires=[wires[0],wires[1]])
        q.RZ(-(delta_plus_beta)/2,wires=wires[1])
        q.RY(-gamma/2,wires=wires[1])
        q.CNOT(wires=[wires[0],wires[1]])
        q.RY(gamma/2,wires=wires[1])
        q.RZ(beta,wires=wires[1])
        q.PhaseShift(alpha,wires=wires[0])
    
    
    # Implements 2-qubit controlled R_k - phase shift gate which is used in QFT.
    # R_k has matrix form
    # [[1,0                   ],
    #  [0,exp(2*pi*i / (2**k))]]
    # If inverse == True, then the function implements 2-qubit controlled R_k_dagger - phase shift gate which is used in inverse QFT
    # R_k_dagger has matrix form
    # [[1,0                    ],
    #  [0,exp(-2*pi*i / (2**k))]]
    def CR_k(self,k,control_wire,operation_wire,inverse=False):
        
        # for further optimization: note that RZ(pi/(2**(k-1))) and PhaseShift(pi/(2**k)) could be effectively translated into elementary blocks
        
        # if k is passed as pennylane.Variable, then use proper format
        if isinstance(k, q.variable.Variable):
            k = k.val
        # check domain of k
        if (k != int(k)) | (k < 1):
            raise Exception('k should be a natural number')
        
        # circuit
        if inverse:
            q.RZ(-np.pi/(2**k), wires=operation_wire)
            q.CNOT(wires=[control_wire,operation_wire])
            q.RZ(np.pi/(2**k), wires=operation_wire)
            q.CNOT(wires=[control_wire,operation_wire])
            q.PhaseShift(-np.pi/(2**k),wires=control_wire)
        else:
            q.RZ(np.pi/(2**k), wires=operation_wire)
            q.CNOT(wires=[control_wire,operation_wire])
            q.RZ(-np.pi/(2**k), wires=operation_wire)
            q.CNOT(wires=[control_wire,operation_wire])
            q.PhaseShift(np.pi/(2**k),wires=control_wire)
    
    
    # Implements C_n_U given arbitrary U for arbitrary amount of control wires (up to n=5)
    def C_U(self,U,control_wires,operation_wire):
        
        if U.shape[0] != 2:
            raise Exception('U should be of size 2x2')
        
        ### Transform U
        U = ClassicalOperations.check_matrix(self,U)
        
        # C_1_U
        if len(control_wires) == 1:
            
            # get angles to use in Controlled_U_block
            angles = ClassicalOperations.ZY_decomposition_angles(self,U)
            alpha = angles['alpha']
            beta = angles['beta']
            gamma = angles['gamma']
            delta = angles['delta']
            
            # circuit
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[0],operation_wire])
        
        # C_2_U
        if len(control_wires) == 2:
            
            # get angles to use in Controlled_U_block
            angles = ClassicalOperations.ZY_decomposition_angles(self,ClassicalOperations.matrix_power(self,U,1/2))
            alpha = angles['alpha']
            beta = angles['beta']
            gamma = angles['gamma']
            delta = angles['delta']
            
            # circuit
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[0],operation_wire])
            q.CNOT(wires=[control_wires[0],control_wires[1]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[1],operation_wire])
            q.CNOT(wires=[control_wires[0],control_wires[1]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[1],operation_wire]) 
        
        # C_3_U
        if len(control_wires) == 3:
            
            # get angles to use in Controlled_U_block
            angles = ClassicalOperations.ZY_decomposition_angles(self,ClassicalOperations.matrix_power(self,U,1/4))
            alpha = angles['alpha']
            beta = angles['beta']
            gamma = angles['gamma']
            delta = angles['delta']
            
            # circuit
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[0],operation_wire])
            q.CNOT(wires=[control_wires[0],control_wires[1]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[1],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            q.CNOT(wires=[control_wires[0],control_wires[1]])
            q.CNOT(wires=[control_wires[0],control_wires[2]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[0],control_wires[2]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[1],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[2],operation_wire])
        
        # C_4_U
        if len(control_wires) == 4:
            
            # get angles to use in Controlled_U_block
            angles = ClassicalOperations.ZY_decomposition_angles(self,ClassicalOperations.matrix_power(self,U,1/8))
            alpha = angles['alpha']
            beta = angles['beta']
            gamma = angles['gamma']
            delta = angles['delta']
            
            # circuit
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[0],operation_wire])
            q.CNOT(wires=[control_wires[0],control_wires[1]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[1],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            q.CNOT(wires=[control_wires[1],control_wires[3]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[3]])
            q.CNOT(wires=[control_wires[0],control_wires[1]])
            q.CNOT(wires=[control_wires[0],control_wires[2]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            q.CNOT(wires=[control_wires[0],control_wires[2]])
            q.CNOT(wires=[control_wires[0],control_wires[3]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[0],control_wires[3]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[1],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            q.CNOT(wires=[control_wires[1],control_wires[3]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[3]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[3],operation_wire])
        
        # C_5_U
        if len(control_wires) == 5:
            
            # get angles to use in Controlled_U_block
            angles = ClassicalOperations.ZY_decomposition_angles(self,ClassicalOperations.matrix_power(U,1/16))
            alpha = angles['alpha']
            beta = angles['beta']
            gamma = angles['gamma']
            delta = angles['delta']
            
            # circuit
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[0],operation_wire])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[1],operation_wire])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[2],operation_wire])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[3],operation_wire])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[0],control_wires[1]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[1],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            q.CNOT(wires=[control_wires[2],control_wires[4]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[4]])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            q.CNOT(wires=[control_wires[1],control_wires[3]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            q.CNOT(wires=[control_wires[1],control_wires[3]])
            q.CNOT(wires=[control_wires[1],control_wires[4]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[4]])
            q.CNOT(wires=[control_wires[0],control_wires[1]])
            q.CNOT(wires=[control_wires[0],control_wires[2]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            q.CNOT(wires=[control_wires[2],control_wires[4]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[4]])
            q.CNOT(wires=[control_wires[0],control_wires[2]])

            q.CNOT(wires=[control_wires[0],control_wires[3]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            q.CNOT(wires=[control_wires[0],control_wires[3]])
            q.CNOT(wires=[control_wires[0],control_wires[4]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[0],control_wires[4]])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            q.CNOT(wires=[control_wires[2],control_wires[4]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[4]])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            q.CNOT(wires=[control_wires[1],control_wires[3]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            q.CNOT(wires=[control_wires[1],control_wires[3]])
            q.CNOT(wires=[control_wires[1],control_wires[4]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[4]])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            # C_1_U
            QuantumGates.Controlled_U_block(self,alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            q.CNOT(wires=[control_wires[2],control_wires[4]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[4]])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            # C_1_U_dagger
            QuantumGates.Controlled_U_block(self,-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])


    def gray_code_C_X(self,gray_code_element,changing_bit,wires):
        
        # if changing_bit is passed as pennylane.Variable, then use proper format
        if isinstance(changing_bit, q.variable.Variable):
            changing_bit = int(changing_bit.val)
        # wires and gray_code_element cannot be passed properly as an argument to QNode's function
        
        # U = X
        U = np.array([[0,1],
                      [1,0]])

        # flip qubit with PauliX if there is 0 in gray_code_element (but not the changing_bit)
        for i in range(len(wires)):
            if (gray_code_element[i] == '0')&(i != changing_bit):
                q.PauliX(wires=wires[i])

        # define control_wires and operation_wire
        control_wires = copy.deepcopy(wires)
        del control_wires[changing_bit]
        # Controlled operation
        QuantumGates.C_U(self,U,control_wires=control_wires, operation_wire=wires[changing_bit])

        # flip qubit with PauliX if there is 0 in gray_code_element (but not the changing_bit)
        for i in range(len(wires)):
            if (gray_code_element[i] == '0')&(i != changing_bit):
                q.PauliX(wires=wires[i])
    
    
    # Implements Two_level_U given angles from ZY decomposition of U. building block for function U_n
    def Two_level_U(self,U,non_trivial_indices,wires):
        
        ### Transform U and non_trivial_indices
        U = ClassicalOperations.check_matrix(self,U)
        # non_trivial_indices should be list (otherwise, error is raised while executing U[np.ix_...])
        if isinstance(non_trivial_indices, list) == False:
            non_trivial_indices = list(non_trivial_indices)
        for i in range(len(non_trivial_indices)):
            if isinstance(non_trivial_indices[i], q.variable.Variable):
                non_trivial_indices[i] = int(non_trivial_indices[i].val)
        
        print(non_trivial_indices, end="\r")
        U_non_trivial_submatrix = U[np.ix_(non_trivial_indices,non_trivial_indices)]
        
        # get Gray code
        a = bin(int(non_trivial_indices[0]))[2:]
        b = bin(int(non_trivial_indices[1]))[2:]
        gray_code, changing_bit = ClassicalOperations.Gray_code(self,a,b,len(wires))

        # circuit
        # Gray code forward sequence of C_Xs
        for i in range(0,len(gray_code)-2):
            QuantumGates.gray_code_C_X(self,gray_code[i],changing_bit[i],wires)

        # C_U
        # flip qubit with PauliX if there is 0 in gray_code_element[-1] (but not the changing_bit)
        for i in range(len(wires)):
            if (gray_code[-1][i] == '0')&(i != changing_bit[-1]):
                q.PauliX(wires=wires[i])
        # define control_wires and operation_wire
        control_wires = copy.deepcopy(wires)
        del control_wires[changing_bit[-1]]
        QuantumGates.C_U(self,U_non_trivial_submatrix,control_wires=control_wires, operation_wire=wires[changing_bit[-1]])
        # flip qubit with PauliX if there is 0 in gray_code_element[-1] (but not the changing_bit)
        for i in range(len(wires)):
            if (gray_code[-1][i] == '0')&(i != changing_bit[-1]):
                q.PauliX(wires=wires[i])

        # Gray code backward sequence of C_Xs
        for i in range(len(gray_code)-3,-1,-1):
            QuantumGates.gray_code_C_X(self,gray_code[i],changing_bit[i],wires)
    
    
    # Arbitrary U circuit
    def U_n(self, U, wires):
        
        # check dimensionality of U
        if int(np.log2(U.shape[0])) != np.log2(U.shape[0]):
            raise Exception('Wrong shape of U: it should be 2**len(wires)')
        
        ### Transform U
        U = ClassicalOperations.check_matrix(self,U)
        
        # get Two_level_U_list and non_trivial_indices of any matrix in the list
        Two_level_U_list = ClassicalOperations.Two_level_unitary_decomposition(self,U)
        decomposition_pairs_list = ClassicalOperations.get_non_trivial_indices(self,Two_level_U_list)
        
        # consequentially execute Two_level_Us
        # note that circuits should be in reverse order relative to matrix decomposition
        for decomposition_pair in reversed(decomposition_pairs_list):
            # execute Two_level_U
            QuantumGates.Two_level_U(self,wires=wires,U=decomposition_pair[0],non_trivial_indices=decomposition_pair[1])
    
    
    # Implements Two_level_U given angles from ZY decomposition of U. Building block for function C_U_n
    def controlled_Two_level_U(self,U,non_trivial_indices,control_wire,operation_wires):
        
        ### Transform U and non_trivial_indices
        U = ClassicalOperations.check_matrix(self,U)
        # non_trivial_indices should be list (otherwise, error is raised while executing U[np.ix_...])
        if isinstance(non_trivial_indices, list) == False:
            non_trivial_indices = list(non_trivial_indices)
        for i in range(len(non_trivial_indices)):
            if isinstance(non_trivial_indices[i], q.variable.Variable):
                non_trivial_indices[i] = int(non_trivial_indices[i].val)
        
        print(non_trivial_indices, end="\r")
        U_non_trivial_submatrix = U[np.ix_(non_trivial_indices,non_trivial_indices)]
        
        # get Gray code for U_n and then edit it to incorporate control_wire
        a = bin(int(non_trivial_indices[0]))[2:]
        b = bin(int(non_trivial_indices[1]))[2:]
        gray_code, changing_bit = ClassicalOperations.Gray_code(self,a,b,len(operation_wires))
        # create wires = [operation_wires, control_wire] to execute the same operations with refined gray_code elements
        wires = operation_wires + list([control_wire])
        # edit gray_code element to incorporate control_wire =
        # = add control_wire for every gray_code element as '1' in the end of code string
        # note that since control_wire is the last in the list of wires, there is no need to edit changing_bit list
        gray_code = [gray_code[i] + '1' for i in range(len(gray_code))]
        
        # circuit
        # Gray code forward sequence of C_Xs
        for i in range(0,len(gray_code)-2):
            QuantumGates.gray_code_C_X(self,gray_code[i],changing_bit[i],wires)

        # C_U
        # flip qubit with PauliX if there is 0 in gray_code_element[-1] (but not the changing_bit)
        for i in range(len(wires)):
            if (gray_code[-1][i] == '0')&(i != changing_bit[-1]):
                q.PauliX(wires=wires[i])
        # define control_wires and operation_wire
        control_wires = copy.deepcopy(wires)
        del control_wires[changing_bit[-1]]
        QuantumGates.C_U(self,U_non_trivial_submatrix,control_wires=control_wires, operation_wire=wires[changing_bit[-1]])
        # flip qubit with PauliX if there is 0 in gray_code_element[-1] (but not the changing_bit)
        for i in range(len(wires)):
            if (gray_code[-1][i] == '0')&(i != changing_bit[-1]):
                q.PauliX(wires=wires[i])

        # Gray code backward sequence of C_Xs
        for i in range(len(gray_code)-3,-1,-1):
            QuantumGates.gray_code_C_X(self,gray_code[i],changing_bit[i],wires)
    
    
    # controlled n-qubit unitary U (one control wire, n operation wires)
    def C_U_n(self, U, control_wire, operation_wires):
        
        # check dimensionality of U
        if int(np.log2(U.shape[0])) != np.log2(U.shape[0]):
            raise Exception('Wrong shape of U: it should be 2**len(operation_wires)')
        
        ### Transform U
        U = ClassicalOperations.check_matrix(self,U)
        
        # get Two_level_U_list and non_trivial_indices of any matrix in the list
        Two_level_U_list = ClassicalOperations.Two_level_unitary_decomposition(self,U)
        decomposition_pairs_list = ClassicalOperations.get_non_trivial_indices(self,Two_level_U_list)
        
        # consequentially execute controlled_Two_level_Us
        # note that circuits should be in reverse order relative to matrix decomposition
        for decomposition_pair in reversed(decomposition_pairs_list):
            # execute controlled_Two_level_U
            QuantumGates.controlled_Two_level_U(self,U=decomposition_pair[0],non_trivial_indices=decomposition_pair[1],control_wire=control_wire,operation_wires=operation_wires)
    
#-------------------------------------------------#
#--- Functions implementing quantum algorithms ---#
#-------------------------------------------------#

class QuantumAlgorithms:
    
    # Implements |a,b> -> |a,a+b> for binary representation of a and b
    # setup: algorithm uses 3 registers - register with prepared a, register with prepared b and register with auxiliary 0s
    # n wires for the register with a (wires_a)
    # n+1 wires for the register with b (wires_b)
    # n wires for the auxiliary register with c (wires_c)
    # Based on Vedral, Barenco, Ekert - "Quantum Networks for Elementary Arithmetic Operations", 1996
    def ADDER(wires_a,wires_b,wires_c):
        
        # check inputs
        if ( (len(wires_a) == len(wires_c))&(len(wires_a)+1 == len(wires_b))== False ):
            raise Exception('Wrong number of wires: should be n wires for wires_a, n+1 wires for wires_b and n wires for wires_c')
        
        # block of CARRY gates
        for i in range(len(wires_a)-1):
            QuantumGates.CARRY(self,[wires_c[i],wires_a[i],wires_b[i],wires_c[i+1]])
        QuantumGates.CARRY(self,[wires_c[len(wires_a)-1],wires_a[len(wires_a)-1],wires_b[len(wires_a)-1],wires_b[len(wires_a)]])
        
        q.CNOT(wires=[wires_a[len(wires_a)-1],wires_b[len(wires_a)-1]])
        
        # block of CARRY and SUM gates
        QuantumGates.SUM(self,wires=[wires_c[len(wires_a)-1],wires_a[len(wires_a)-1],wires_b[len(wires_a)-1]])
        for i in range(len(wires_a)-2,-1,-1):
            QuantumGates.CARRY(self,[wires_c[i],wires_a[i],wires_b[i],wires_c[i+1]])
            QuantumGates.SUM(self,[wires_c[i],wires_a[i],wires_b[i]])
    
    
    # Implements Quantum Fourier transform. 
    # If inverse == True, then the function implements inverse Quantum Fourier transform
    def QFT(self,wires,inverse=False):
        
        # block of Hadamard and CR_k gates
        for i in range(len(wires)):
            q.Hadamard(wires=wires[i])
            for k in range(2,(len(wires)-i)+1):
                QuantumGates.CR_k(self,k=k,inverse=inverse,control_wire=wires[i+(k-1)],operation_wire=wires[i])
        
        # block of SWAP gates
        # works both if len(wires)%2 == 0 or len(wires)%2 == 1
        for i in range(int(len(wires)/2)):
            QuantumGates.SWAP(self,[wires[i],wires[len(wires)-(i+1)]])
    
    
    def Phase_Estimation(self,U,t,wires):
        
        # 1. Prepare initial state
        # preparation should be performed in QNode environment
        
        # 2. Create superposition with Hadamard and C_U_n gates
        for i in range(t):
            q.Hadamard(wires=wires[i])
        for i in range(t):
            QuantumGates.C_U_n(self,ClassicalOperations.matrix_natural_power(self,U,power=2**i),control_wire=wires[(t-1)-i],operation_wires=wires[t:])
        
        # 3. Apply inverse Quantum Fourier transform to the first register
        QuantumAlgorithms.QFT(self,wires=wires[:t],inverse=True)
        
        # 4. Measure the first register
        # measurements should be performed in QNode environment
        
        
        
