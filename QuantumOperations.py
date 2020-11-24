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
    
    
    # returns np.array with bitwise binary representation of a given number
    # if option reverse=True reverses array so that the first bit is for 2^0 and the last is for 2^(k-1)
    # if option bits is set to a number, then the length of resulting array is bits.
    def int_to_binary_array(self,n,reverse=False,bits=False):
        if bits == False:
            if reverse == False:
                return np.array([int(i) for i in bin(n)[2:]])
            else:
                return np.array([int(i) for i in bin(n)[2:][::-1]])
        else:
            if reverse == False:
                bin_repr = bin(n)[2:]
                bin_repr = '0'*(bits-len(bin_repr)) + bin_repr
                return np.array([int(i) for i in bin_repr])
            else:
                bin_repr = bin(n)[2:][::-1]
                bin_repr = bin_repr + '0'*(bits-len(bin_repr))
                return np.array([int(i) for i in bin_repr])
    
    
    def get_non_trivial_indices(self,Two_level_U_list):
        # due to algorithmic structure of the two-level unitary decomposition, non_trivial_indices are known in advance
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
    
    
    # Euclid's algorithm - finds greater common devider
    # Note: doesn't work correctly for too big numbers (because a%b and int(a/b) do not work)
    def gcd(self,a,b):
        ## check
        # format
        if isinstance(a, q.variable.Variable):
            a = a.val# format
        if isinstance(b, q.variable.Variable):
            b = b.val
        # order
        if a < b:
            a,b = b,a
        
        
        ## algorithm
        r = a%b
        if r == 0:
            return b
        else:
            return ClassicalOperations.gcd(self,b,r)
    
    
    # Auxiliary recursive function for finding alpha and beta in r_n = alpha*a + beta*b, where n: r_n = gcd(a,b)
    # algorithm should be initialized with alpha = 1, beta = -k_(n-2), i = n-2
    # before algorithms' execution, array of k should be defined
    # i denotes level in Euclid's algorithm
    # algorithm should be initialized with alpha = 1, beta = -k_(n-2), i = n-2, where n: r_n = gcd(a,b)
    # Note: doesn't work correctly for too big numbers (because a%b and int(a/b) do not work)
    def diophantine_equation_auxiliary(self,k_list,alpha,beta,i):
        
        if i != 1:
            return ClassicalOperations.diophantine_equation_auxiliary(self,k_list=k_list,alpha=beta,beta=alpha-beta*k_list[i-2],i=i-1)
        else:
            return [alpha,beta]
    
    
    # solves diophantine equation, i.e. given a,b returns x,y such that ax + by = gcd(a,b)
    # Euclid's algorithm produces set of values (k_i,r_i), where r_i = k_i*r_(i+1) + r_(i+2), i goes from 0 to n
    def diophantine_equation(self,a,b):
        
        ## check
        # format
        if isinstance(a, q.variable.Variable):
            a = a.val# format
        if isinstance(b, q.variable.Variable):
            b = b.val
        # order
        flag = 0
        if a < b:
            a,b = b,a
            flag = 1
        
        # if b == gcd(a,b)
        if b == ClassicalOperations.gcd(self,a,b):
            if flag == 0:
                return [0,1]
            if flag == 1:
                return [1,0]
            
        # initialize r and lists of r and k from the equation a = kb + r
        r_list = list([a])
        k_list = list()

        # forward part of the algorithm (Nielsen Chuang p.628)
        while b!=0:
            r_list.append(b)
            k_list.append(int(a/b))
            a,b = b,a%b
        
        # backward part of the algorithm (Nielsen Chuang p. 629)
        if flag == 0:
            return ClassicalOperations.diophantine_equation_auxiliary(self,k_list=k_list,alpha=1,beta=-k_list[-2],i=len(r_list)-2)
        if flag == 1:
            return ClassicalOperations.diophantine_equation_auxiliary(self,k_list=k_list,alpha=1,beta=-k_list[-2],i=len(r_list)-2)[::-1]
    
    
    # finds modular multiplicative inverse using diophantine_equation
    def modular_multiplicative_inverse(self,a,N):
        
        ## check
        # format
        if isinstance(a, q.variable.Variable):
            a = a.val# format
        if isinstance(N, q.variable.Variable):
            N = N.val
        # co-primality
        if ClassicalOperations.gcd(self,a,N) != 1:
            raise Exception('a and N should be co-prime, i.e. gcd(a,N) should be 1')
#         if a >= N:
#             raise Exception('a should be less than N - check that order of arguments of the function is right')
        
        ## algorithm
        # get inverse
        inverse = ClassicalOperations.diophantine_equation(self,a=a,b=N)[0]
        
        # make inverse positive if necessary
        if inverse < 0:
            inverse = inverse%N
        
        return inverse
    
    
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
#--- Functions implementing quantum gates ---#
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
    
    
    # Implements Toffoli gate with 3 controls instead of 2
    # requires 1 work qubit
    def Controlled_Toffoli(self,control_wires,operation_wire,work_wire):
        QuantumGates.Toffoli(self,wires=[control_wires[0],control_wires[1],work_wire])
        QuantumGates.Toffoli(self,wires=[control_wires[2],work_wire,operation_wire])
        QuantumGates.Toffoli(self,wires=[control_wires[0],control_wires[1],work_wire])
    
    
    # Implements standard 2-wires SWAP-gate
    def SWAP(self,wires):
        q.CNOT(wires=[wires[0],wires[1]])
        q.CNOT(wires=[wires[1],wires[0]])
        q.CNOT(wires=[wires[0],wires[1]])
    
    
    # Implements 2-wires SWAP conditional on 1-wire control
    def Controlled_SWAP(self,control_wire,swap_wires):
        QuantumGates.Toffoli(self,wires=[control_wire,swap_wires[0],swap_wires[1]])
        QuantumGates.Toffoli(self,wires=[control_wire,swap_wires[1],swap_wires[0]])
        QuantumGates.Toffoli(self,wires=[control_wire,swap_wires[0],swap_wires[1]])
    
    
    # Implements 2-wires SWAP conditional on 2-wires controls
    # requires 1 work qubit, because it uses controlled_Toffoli
    def Controlled_Controlled_SWAP(self,control_wires,swap_wires,work_wire):
        QuantumGates.Controlled_Toffoli(self,control_wires=[control_wires[0],control_wires[1],swap_wires[0]],operation_wire=swap_wires[1],work_wire=work_wire)
        QuantumGates.Controlled_Toffoli(self,control_wires=[control_wires[0],control_wires[1],swap_wires[1]],operation_wire=swap_wires[0],work_wire=work_wire)
        QuantumGates.Controlled_Toffoli(self,control_wires=[control_wires[0],control_wires[1],swap_wires[0]],operation_wire=swap_wires[1],work_wire=work_wire)
    
    
    # Implements SWAP for 2 n-wires register conditional on 1-wire control
    def Controlled_register_SWAP(self,control_wire,wires_register_1,wires_register_2):
        
        # check inputs
        if ( (isinstance(control_wire, list) == False)&(len(wires_register_1) == len(wires_register_2)) == False):
            raise Exception('Wrong number of wires: should be 1 wire for control_wire, n wires for wires_register_1 and n wires for wires_register_2')
        
        # circuit
        for i in range(len(wires_register_1)):
            QuantumGates.Controlled_SWAP(self,control_wire=control_wire,swap_wires=[wires_register_1[i],wires_register_2[i]])
    
    
    # Implements resetting register with zeros to binary representation of classicaly known number N conditional on 1-wire control
    # if control == 1, then resulting values in the wires_zero_register are [N_0, N_1, ... , N_(n-1)], where N = N_(n-1)*2^(n-1) + ... + N_1*2^1 + N_0*2^0
    def Controlled_reset_zero_register_to_N(self,control_wire,wires_zero_register,N):
        
        # check N
        # format
        if isinstance(N, q.variable.Variable):
            N = N.val
        # check if N does not match the size of wires_zero_register
        if N > 2**(len(wires_zero_register))-1:
            raise Exception('N is too big for the register wires_zero_register')
        # make a string with a binary represenatation of N (in reverse order)
        N = bin(N)[2:][::-1]
        # add zeros to match the register's size
        N = N + '0'*(len(wires_zero_register)-len(N))
        
        # check other inputs
        if (isinstance(control_wire, list)):
            raise Exception('control_wire should not be a list')
        
        # circuit
        # CNOTs with control=control_wire and taget - wire in wires_zero_register, for which N == '1'
        for i in range(len(wires_zero_register)):
            if N[i] == '1':
                q.CNOT(wires=[control_wire,wires_zero_register[i]])
    
    
    # Implements 4-wires carry operation used for ADDER
    # setup: wires[0] = c_i, wires[1] = a_i, wires[2] = b_i, wires[3] = c_(i+1) = |0>
    # operation carries |1> in wires[3] = c_(i+1) if c_i + a_i + b_i > 1
    # Based on Vedral, Barenco, Ekert - "Quantum Networks for Elementary Arithmetic Operations", 1996
    def CARRY(self,wires,inverse=False):
        if inverse == False:
            QuantumGates.Toffoli(self,wires=wires[1:])
            q.CNOT(wires=[wires[1],wires[2]])
            QuantumGates.Toffoli(self,wires=[wires[0],wires[2],wires[3]])
        else:
            QuantumGates.Toffoli(self,wires=[wires[0],wires[2],wires[3]])
            q.CNOT(wires=[wires[1],wires[2]])
            QuantumGates.Toffoli(self,wires=wires[1:])
    
    # Implements 3-wires carry operation used for ADDER
    # setup: wires[0] = a, wires[1] = b, wires[2] = |0>
    # operation makes wires[2] = a+b mod 2
    # Based on Vedral, Barenco, Ekert - "Quantum Networks for Elementary Arithmetic Operations", 1996
    def SUM(self,wires,inverse=False):
        if inverse == False:
            q.CNOT(wires=[wires[1],wires[2]])
            q.CNOT(wires=[wires[0],wires[2]])
        else:
            q.CNOT(wires=[wires[0],wires[2]])
            q.CNOT(wires=[wires[1],wires[2]])
    
    
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
    
    
    # Implements |a,b> -> |a,a+b> for binary representation of a and b
    # setup: algorithm uses 3 registers - register with prepared a, register with prepared b and register with auxiliary 0s
    # n wires for the register with a (wires_a)
    # n+1 wires for the register with b (wires_b)
    # n wires for the auxiliary register with c (wires_c)
    # Based on Vedral, Barenco, Ekert - "Quantum Networks for Elementary Arithmetic Operations", 1996
    def ADDER(self,wires_a,wires_b,wires_c,inverse=False):
        
        # check inputs
        if ( (len(wires_a) == len(wires_c))&(len(wires_a)+1 == len(wires_b))== False ):
            raise Exception('Wrong number of wires: should be n wires for wires_a, n+1 wires for wires_b and n wires for wires_c')
        
        # circuit
        if inverse == False:
            # block of CARRY gates
            for i in range(len(wires_a)-1):
                QuantumGates.CARRY(self,[wires_c[i],wires_a[i],wires_b[i],wires_c[i+1]])
            QuantumGates.CARRY(self,[wires_c[len(wires_a)-1],wires_a[len(wires_a)-1],wires_b[len(wires_a)-1],wires_b[len(wires_a)]])

            q.CNOT(wires=[wires_a[len(wires_a)-1],wires_b[len(wires_a)-1]])

            # block of inverse-CARRY and SUM gates
            QuantumGates.SUM(self,wires=[wires_c[len(wires_a)-1],wires_a[len(wires_a)-1],wires_b[len(wires_a)-1]])
            for i in range(len(wires_a)-2,-1,-1):
                QuantumGates.CARRY(self,[wires_c[i],wires_a[i],wires_b[i],wires_c[i+1]],inverse=True)
                QuantumGates.SUM(self,[wires_c[i],wires_a[i],wires_b[i]])
        else:
            # block of inverse-SUM and CARRY gates
            for i in range(len(wires_a)-1):
                QuantumGates.SUM(self,[wires_c[i],wires_a[i],wires_b[i]],inverse=True)
                QuantumGates.CARRY(self,[wires_c[i],wires_a[i],wires_b[i],wires_c[i+1]])
            QuantumGates.SUM(self,wires=[wires_c[len(wires_a)-1],wires_a[len(wires_a)-1],wires_b[len(wires_a)-1]],inverse=True)
            
            q.CNOT(wires=[wires_a[len(wires_a)-1],wires_b[len(wires_a)-1]])
            
            # block of inverse-CARRY gates
            QuantumGates.CARRY(self,[wires_c[len(wires_a)-1],wires_a[len(wires_a)-1],wires_b[len(wires_a)-1],wires_b[len(wires_a)]],inverse=True)
            for i in range(len(wires_a)-2,-1,-1):
                QuantumGates.CARRY(self,[wires_c[i],wires_a[i],wires_b[i],wires_c[i+1]],inverse=True)
    
    
    # Implements |a,b> -> |a,a+b mod N> for binary representation of a and b
    # N is classicaly known
    # If inverse == True, implements |a,b> -> |a,b-a mod N> for binary representation of a and b
    # setup: algorithm uses 5 registers - register with prepared a, register with prepared b, register with prpared N, register with prepared |0..0> for input c into ADDERs and register t with auxiliary |0>
    # n wires for the register with a (wires_a)
    # n+1 wires for the register with b (wires_b)
    # n wires for the register with |0...0> (wires_c) for ADDERs
    # n wires for the register with N (wires_N)
    # 1 wire for the register with t=|0> (wires_t)
    # Note that it works correctly only for 0 <= a,b < N
    # Note that it seems like register t might be equal to |1> for a,b and N which are not 0 <= a,b < N
    # Based on Vedral, Barenco, Ekert - "Quantum Networks for Elementary Arithmetic Operations", 1996
    def ADDER_MOD(self,wires_a,wires_b,wires_c,wires_N,wires_t,N,inverse=False):
        
        # check inputs
        # is it true that wires_a == wires_N ???
        if ( (len(wires_a)+1 == len(wires_b))&(len(wires_a) == len(wires_c))&(len(wires_a) == len(wires_N))&(isinstance(wires_t, list) == False)== False ):
            raise Exception('Wrong number of wires: should be n wires for wires_a, n+1 wires for wires_b, n wires for wires_c, n wires for wires_N and 1 wire for wires_t')
        
        # circuit
        if inverse == False:
            QuantumAlgorithms.ADDER(self,wires_a,wires_b,wires_c)
            QuantumAlgorithms.ADDER(self,wires_N,wires_b,wires_c,inverse=True)
            
            q.PauliX(wires=wires_b[-1])
            q.CNOT(wires=[wires_b[-1],wires_t])
            q.PauliX(wires=wires_b[-1])
            
            QuantumGates.Controlled_reset_zero_register_to_N(self,control_wire=wires_t,wires_zero_register=wires_N,N=N)
            QuantumAlgorithms.ADDER(self,wires_N,wires_b,wires_c)
            QuantumGates.Controlled_reset_zero_register_to_N(self,control_wire=wires_t,wires_zero_register=wires_N,N=N)
            
            QuantumAlgorithms.ADDER(self,wires_a,wires_b,wires_c,inverse=True)
            q.CNOT(wires=[wires_b[-1],wires_t])
            QuantumAlgorithms.ADDER(self,wires_a,wires_b,wires_c)
        else:
            QuantumAlgorithms.ADDER(self,wires_a,wires_b,wires_c,inverse=True)
            q.CNOT(wires=[wires_b[-1],wires_t])
            QuantumAlgorithms.ADDER(self,wires_a,wires_b,wires_c)
            
            QuantumGates.Controlled_reset_zero_register_to_N(self,control_wire=wires_t,wires_zero_register=wires_N,N=N)
            QuantumAlgorithms.ADDER(self,wires_N,wires_b,wires_c,inverse=True)
            QuantumGates.Controlled_reset_zero_register_to_N(self,control_wire=wires_t,wires_zero_register=wires_N,N=N)
            
            q.PauliX(wires=wires_b[-1])
            q.CNOT(wires=[wires_b[-1],wires_t])
            q.PauliX(wires=wires_b[-1])
            
            QuantumAlgorithms.ADDER(self,wires_N,wires_b,wires_c)
            QuantumAlgorithms.ADDER(self,wires_a,wires_b,wires_c,inverse=True)
    
    
    # Implements |0,z,0> -> |0,z,z> and |1,z,0> -> |1,z,zy mod N> for binary representation of z and y
    # y and N are classically known
    # If inverse == True, implements ???
    # setup: algorithm uses 7 registers - control register, register with prepared z, register with prepared |0..0> for input a into ADDER_MODs, register with prepared |0..0> for input b into ADDER_MODs, register with prepared |0..0> for input c into ADDER_MODs, register with prpared N and register t with auxiliary |0>
    # 1 wire for the control register (control_wire)
    # n wires for the register with z (wires_z)
    # n wires for the register with a (wires_a)
    # n+1 wires for the register with b (wires_b)
    # n wires for the register with |0..0> (wires_c)
    # n wires for the register with N (wires_N)
    # 1 wire for the register with t=|0> (wires_t)
    # Based on Vedral, Barenco, Ekert - "Quantum Networks for Elementary Arithmetic Operations", 1996
    def Controlled_MULT_MOD(self,control_wire,wires_z,wires_a,wires_b,wires_c,wires_N,wires_t,N,m,inverse=False):
        
        n = len(wires_z)
        
        # check N and m
        # format of N and m
        if isinstance(N, q.variable.Variable):
            N = N.val
        if isinstance(m, q.variable.Variable):
            m = m.val
        # check if N does not match the size of wires_N
        if N > 2**(len(wires_N))-1:
            raise Exception('N is too big for the register wires_N')
        
        # check control inputs
        if (isinstance(control_wire, list)):
            raise Exception('control_wire should not be a list')
        if (isinstance(wires_t, list)):
            raise Exception('wires_t should not be a list')
        
        # check wires
        if ( (len(wires_a) == n)&(len(wires_b) == n+1)&(len(wires_c) == n)&(len(wires_N) == n) == False):
            raise Exception('Wrong size of registers - it should be: \n 1 wire for the control register (control_wire) \n n wires for the register with z (wires_z) \n n wires for the register with a (wires_a) \n n+1 wires for the register with b (wires_b) \n n wires for the register with |0..0> (wires_c) \n n wires for the register with N (wires_N) \n 1 wire for the register with t=|0> (wires_t)')
            
        
        # circuit
        if inverse == False:
            ### block with ADDER_MODs
            # cycle to iteratively add ADDER_MODs each surrounded by Toffoli gates with control1=control_wire, control2=wires_z[i] and target - wire in wires_a, for which m*2^i == '1'
            for i in range(len(wires_z)):
                # binary representation of m*2^i mod N
                m_2_to_power_i_mod_N = bin((m*(2**i)) % N)[2:][::-1]
                m_2_to_power_i_mod_N = m_2_to_power_i_mod_N + '0'*( len(wires_a)-len(m_2_to_power_i_mod_N) )
                
                # cycle for Toffoli gates before ADDER_MOD to put m*2^i to wires_a if control_wire == 1
                for j in range(len(wires_a)):
                    if m_2_to_power_i_mod_N[j] == '1':
                        QuantumGates.Toffoli(self,wires=[control_wire,wires_z[i],wires_a[j]])

                # ADDER_MOD[i]
                QuantumAlgorithms.ADDER_MOD(self,wires_a=wires_a,wires_b=wires_b,wires_c=wires_c,wires_N=wires_N,wires_t=wires_t,N=N)

                # cycle for Toffoli gates after ADDER_MOD to make wires_a |0..0> if control_wire == 1
                for j in range(len(wires_a)):
                    if m_2_to_power_i_mod_N[j] == '1':
                        QuantumGates.Toffoli(self,wires=[control_wire,wires_z[i],wires_a[j]])

            ### block for copying z into wires_a conditional on control_wire == |0>. That is, we want |0,z,0> -> |0,z,z>
            q.PauliX(wires=control_wire)
            for i in range(len(wires_z)):
                QuantumGates.Toffoli(self,wires=[control_wire,wires_z[i],wires_a[i]])
            q.PauliX(wires=control_wire)
        
        else:
            ### block for copying z into wires_a conditional on control_wire == |0>
            q.PauliX(wires=control_wire)
            for i in range(len(wires_z)):
                QuantumGates.Toffoli(self,wires=[control_wire,wires_z[i],wires_a[i]])
            q.PauliX(wires=control_wire)
            
            for i in range(len(wires_z)-1,-1,-1):
                # binary representation of m*2^i mod N
                m_2_to_power_i_mod_N = bin((m*(2**i)) % N)[2:][::-1]
                m_2_to_power_i_mod_N = m_2_to_power_i_mod_N + '0'*( len(wires_a)-len(m_2_to_power_i_mod_N) )
                
                # cycle for Toffoli gates after ADDER_MOD to make wires_a |0..0> if control_wire == 1
                for j in range(len(wires_a)):
                    if m_2_to_power_i_mod_N[j] == '1':
                        QuantumGates.Toffoli(self,wires=[control_wire,wires_z[i],wires_a[j]])
                
                # ADDER_MOD[i]
                QuantumAlgorithms.ADDER_MOD(self,wires_a=wires_a,wires_b=wires_b,wires_c=wires_c,wires_N=wires_N,wires_t=wires_t,N=N,inverse=True)
                
                # cycle for Toffoli gates before ADDER_MOD to put m*2^i to wires_a if control_wire == 1
                for j in range(len(wires_a)):
                    if m_2_to_power_i_mod_N[j] == '1':
                        QuantumGates.Toffoli(self,wires=[control_wire,wires_z[i],wires_a[j]])
    
    
    # Implements |x,1> -> |x,y^x mod N>
    # y and N are classically known
    # If inverse == True, implements ???
    # setup: algorithm uses 7 registers - registerwith prepared x, register with prepared 1 for input z into ADDER_MODs, register with prepared |0..0> for input a into ADDER_MODs, register with prepared |0..0> for input b into ADDER_MODs, register with prepared |0..0> for input c into ADDER_MODs, register with prpared N and register t with auxiliary |0>
    # up to 2n wires for the register with x (wires_x)
    # n wires for the register with z (wires_z)
    # n wires for the register with |0..0> (wires_a)
    # n+1 wires for the register with |0..0> (wires_b)
    # n wires for the register with |0..0> (wires_c)
    # n wires for the register with N (wires_N)
    # 1 wire for the register with t=|0> (wires_t)
    # Based on Vedral, Barenco, Ekert - "Quantum Networks for Elementary Arithmetic Operations", 1996
    def MODULAR_EXPONENTIATION(self,wires_x,wires_z,wires_a,wires_b,wires_c,wires_N,wires_t,N,y,inverse=False):
        
        n = len(wires_z)
        
        # check N and m
        # format of N and m
        if isinstance(N, q.variable.Variable):
            N = N.val
        if isinstance(y, q.variable.Variable):
            y = y.val
        # check if N does not match the size of wires_N
        if N > 2**(len(wires_N))-1:
            raise Exception('N is too big for the register wires_N')
        
        # check control inputs
        if (isinstance(wires_t, list)):
            raise Exception('wires_t should not be a list')
        
        # check wires
        if ( (len(wires_z) <= 2*n)&(len(wires_a) == n)&(len(wires_b) == n+1)&(len(wires_c) == n)&(len(wires_N) == n) == False):
            raise Exception('Wrong size of registers - it should be: \n <= 2n wires for the register with x (wires_x) \n n wires for the register with z (wires_z) \n n wires for the register with a (wires_a) \n n+1 wires for the register with b (wires_b) \n n wires for the register with |0..0> (wires_c) \n n wires for the register with N (wires_N) \n 1 wire for the register with t=|0> (wires_t)')
        
        
        # circuit
        
        for i in range(len(wires_x)):
            QuantumAlgorithms.Controlled_MULT_MOD(self,control_wire=wires_x[i],wires_z=wires_z,wires_a=wires_a,wires_b=wires_b,wires_c=wires_c,wires_N=wires_N,wires_t=wires_t,N=N,m=y**(2**i))
            # SWAP register wires_z with wires_b[:-1]
            for j in range(len(wires_z)):
                QuantumGates.Controlled_SWAP(self,control_wire=wires_x[i],swap_wires=[wires_z[j],wires_b[j]])
            # find modular multiplicative inverse of y**(2**i)
            inverse_y_2_i = ClassicalOperations.modular_multiplicative_inverse(self,a=y**(2**i),N=N)
            QuantumAlgorithms.Controlled_MULT_MOD(self,control_wire=wires_x[i],wires_z=wires_z,wires_a=wires_a,wires_b=wires_b,wires_c=wires_c,wires_N=wires_N,wires_t=wires_t,N=N,m=inverse_y_2_i,inverse=True)
    
    
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
        
        
        
