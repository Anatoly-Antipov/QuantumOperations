import numpy as np
import pennylane as q
from sympy import Matrix
import copy


#-------------------------------------#
#--- Classical auxiliary functions ---#
#-------------------------------------#

class NonQuantumOperations:
    
    # prints out list of states in a computational basis
    def states_vector(self,wires):
        states_vector = list()
        for i in range(2**len(wires)):
            states_vector.append('|'+'0'*(len(wires)-len(bin(i)[2:]))+bin(i)[2:]+'>')
        return states_vector


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
        # if U is passed as pennylane.Variable, then use type np.complex128
        if isinstance(U[0][0], q.variable.Variable):
            self.U = np.zeros(dtype=np.complex128,shape=[U.shape[0],U.shape[0]])
            # no straight-forward't matrix-wise operation .val for ndarray of pennylane.Variables => use cycle
            for i in range(U.shape[0]):
                for j in range(U.shape[0]):
                    self.U[i][j] = U[i][j].val
        else:
            self.U = U
        # if power is passed as pennylane.Variable, then use type np.complex128
        if isinstance(power, q.variable.Variable):
            self.power = power.val
        else:
            self.power = power
        
        # get eigenvectors and eigenvalues
        self.e = Matrix(self.U).eigenvects()
        self.l0 = np.array(self.e[0][0]).astype(np.complex128)
        self.l1 = np.array(self.e[1][0]).astype(np.complex128)
        self.h0 = np.array(self.e[0][2][0].T).astype(np.complex128)
        self.h0 = self.h0/np.sqrt(self.h0.dot(np.conj(self.h0.T)))
        self.h1 = np.array(self.e[1][2][0].T).astype(np.complex128)
        self.h1 = self.h1/np.sqrt(self.h1.dot(np.conj(self.h1.T)))
        
        return np.power(self.l0,self.power)*(self.h0.T).dot(np.conj(self.h0)) + np.power(self.l1,self.power)*(self.h1.T).dot(np.conj(self.h1))
    
    
    # Given 2x2 matrix, function returns angles of ZY decomposition
    def ZY_decomposition_angles(self, U):
        
        if U.shape[0] != 2:
            raise Exception('U should be of size 2x2')
        
        ### Transform U
        # if U is passed as pennylane.Variable, then use type np.complex128
        if isinstance(U[0][0], q.variable.Variable):
            self.U = np.zeros(dtype=np.complex128,shape=[U.shape[0],U.shape[0]])
            # no straight-forward't matrix-wise operation .val for ndarray of pennylane.Variables => use cycle
            for i in range(U.shape[0]):
                for j in range(U.shape[0]):
                    self.U[i][j] = U[i][j].val
        else:
            self.U = U
        
        ### Computations
        # if U doesn't contain 0's, then apply general method
        if (self.U[1][1] != 0)&(self.U[1][0] != 0):
            # alpha
            alpha = np.imag(np.log(self.U[0][0]*self.U[1][1] - self.U[0][1]*self.U[1][0])) / 2
            # beta and delta
            phi0 = -np.imag(np.log(self.U[0][0] / self.U[1][1])) / 2
            phi1 = -np.imag(np.log(-self.U[0][1] / self.U[1][0])) / 2
            beta = phi0 + phi1
            delta = phi0 - phi1
            # gamma
            cos_gamma_halved = np.real(self.U[0][0] / np.exp(1j*(alpha-beta/2-delta/2)))
            sin_gamma_halved = np.real(-self.U[0][1] / np.exp(1j*(alpha-beta/2+delta/2)))
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
        if (self.U[1][1] == 0):
            # aplha
            alpha = np.imag(np.log(-self.U[0][1] + 0j) + np.log(self.U[1][0] + 0j)) / 2 # '+0j' in order to make argument for np.log COMPLEX
            # beta and delta
            phi1 = -(np.imag(np.log(-self.U[0][1] + 0j) - np.log(self.U[1][0] + 0j))) / 2 # '+0j' in order to make argument for np.log COMPLEX
            beta = 2*phi1
            delta = 0
            # gamma
            gamma = np.pi
        # beta and delta are not unique in this case, so we set beta = 2*phi0 and delta = 0
        if (self.U[1][0] == 0):
            # aplha
            alpha = np.imag(np.log(self.U[0][0]*self.U[1][1] + 0j)) / 2 # '+0j' in order to make argument for np.log COMPLEX
            # beta and delta
            phi0 = -np.imag(np.log(self.U[0][0] / self.U[1][1] + 0j)) / 2 # '+0j' in order to make argument for np.log COMPLEX
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
        
        self.RZ_beta = np.array([[np.exp(-1j*beta/2),0],
                            [0,np.exp(1j*beta/2)]])
        self.RY_gamma = np.array([[np.cos(gamma/2),-np.sin(gamma/2)],
                            [np.sin(gamma/2),np.cos(gamma/2)]])
        self.RZ_delta = np.array([[np.exp(-1j*delta/2),0],
                            [0,np.exp(1j*delta/2)]])
        return np.exp(1j*alpha)*self.RZ_beta.dot(self.RY_gamma).dot(self.RZ_delta)
    
    
    # Given U, function returns its two-level unitary decomposition
    # Note that U = decomposition_list[0]*...*decomposition_list[n-1]
    def Two_level_unitary_decomposition(self,U):
        
        ### Transform U
        # if U is passed as pennylane.Variable, then use type np.complex128
        if isinstance(U[0][0], q.variable.Variable):
            # no straight-forward't matrix-wise operation .val for ndarray of pennylane.Variables => use cycle
            for i in range(U.shape[0]):
                for j in range(U.shape[0]):
                    U[i][j] = U[i][j].val
        
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
            
            if U.shape[0] == 3:
                # rearrange matrices in a way that U0 = np.conj(V_1.T)*...*(np.conj(V_n.T))
                decomposition_list = list()
                for i in range(1,len(V_list)):
                    decomposition_list.append(np.conj(V_list[i].T))

            # recrsuive element
            if U.shape[0] > 3:
                
                # rearrange n-1 matrices in a way that U0 = np.conj(V_1.T)*...*(np.conj(V_(n-1).T))
                decomposition_list = list()
                for i in range(1,len(V_list)-1):
                    decomposition_list.append(np.conj(V_list[i].T))
                
                # the last matrix V_n should be further decomposed - append result of the decomposition
                deligated_task = NonQuantumOperations.Two_level_unitary_decomposition(self,np.conj(V_list[-1][1:U.shape[0]:1, 1:U.shape[0]:1].T))
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
    
    
    # C_1_U block for 2x2 U to be inserted into the main circuit
    # Implements controlled-U given angles from ZY-decomposition of U
    def Controlled_U_block(self,alpha,beta,gamma,delta,delta_plus_beta,delta_minus_beta,wires=[0,1]):
        
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
    
    
    # Implements C_n_U given angles from ZY decomposition of U for arbitrary amount of control wires (up to n=5)
    def C_U(self,U,control_wires,operation_wire):
        
        if U.shape[0] != 2:
            raise Exception('U should be of size 2x2')
        
        ### Transform U
        # if U is passed as pennylane.Variable, then use type np.complex128
        if isinstance(U[0][0], q.variable.Variable):
            # no straight-forward't matrix-wise operation .val for ndarray of pennylane.Variables => use cycle
            for i in range(U.shape[0]):
                for j in range(U.shape[0]):
                    U[i][j] = U[i][j].val
        
        # C_1_U
        if len(control_wires) == 1:
            
            # get angles to use in Controlled_U_block
            angles = NonQuantumOperations.ZY_decomposition_angles(self,U)
            alpha = angles['alpha']
            beta = angles['beta']
            gamma = angles['gamma']
            delta = angles['delta']
            
            # circuit
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[0],operation_wire])
        
        # C_2_U
        if len(control_wires) == 2:
            
            # get angles to use in Controlled_U_block
            angles = NonQuantumOperations.ZY_decomposition_angles(self,NonQuantumOperations.matrix_power(self,U,1/2))
            alpha = angles['alpha']
            beta = angles['beta']
            gamma = angles['gamma']
            delta = angles['delta']
            
            # circuit
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[0],operation_wire])
            q.CNOT(wires=[control_wires[0],control_wires[1]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[1],operation_wire])
            q.CNOT(wires=[control_wires[0],control_wires[1]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[1],operation_wire]) 
        
        # C_3_U
        if len(control_wires) == 3:
            
            # get angles to use in Controlled_U_block
            angles = NonQuantumOperations.ZY_decomposition_angles(self,NonQuantumOperations.matrix_power(self,U,1/4))
            alpha = angles['alpha']
            beta = angles['beta']
            gamma = angles['gamma']
            delta = angles['delta']
            
            # circuit
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[0],operation_wire])
            q.CNOT(wires=[control_wires[0],control_wires[1]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[1],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            q.CNOT(wires=[control_wires[0],control_wires[1]])
            q.CNOT(wires=[control_wires[0],control_wires[2]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[0],control_wires[2]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[1],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[2],operation_wire])
        
        # C_4_U
        if len(control_wires) == 4:
            
            # get angles to use in Controlled_U_block
            angles = NonQuantumOperations.ZY_decomposition_angles(self,NonQuantumOperations.matrix_power(self,U,1/8))
            alpha = angles['alpha']
            beta = angles['beta']
            gamma = angles['gamma']
            delta = angles['delta']
            
            # circuit
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[0],operation_wire])
            q.CNOT(wires=[control_wires[0],control_wires[1]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[1],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            q.CNOT(wires=[control_wires[1],control_wires[3]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[3]])
            q.CNOT(wires=[control_wires[0],control_wires[1]])
            q.CNOT(wires=[control_wires[0],control_wires[2]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            q.CNOT(wires=[control_wires[0],control_wires[2]])
            q.CNOT(wires=[control_wires[0],control_wires[3]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[0],control_wires[3]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[1],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            q.CNOT(wires=[control_wires[1],control_wires[3]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[3]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[3],operation_wire])
        
        # C_5_U
        if len(control_wires) == 5:
            
            # get angles to use in Controlled_U_block
            angles = NonQuantumOperations.ZY_decomposition_angles(self,NonQuantumOperations.matrix_power(U,1/16))
            alpha = angles['alpha']
            beta = angles['beta']
            gamma = angles['gamma']
            delta = angles['delta']
            
            # circuit
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[0],operation_wire])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[1],operation_wire])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[2],operation_wire])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[3],operation_wire])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[0],control_wires[1]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[1],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            q.CNOT(wires=[control_wires[2],control_wires[4]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[4]])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            q.CNOT(wires=[control_wires[1],control_wires[3]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            q.CNOT(wires=[control_wires[1],control_wires[3]])
            q.CNOT(wires=[control_wires[1],control_wires[4]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[4]])
            q.CNOT(wires=[control_wires[0],control_wires[1]])
            q.CNOT(wires=[control_wires[0],control_wires[2]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            q.CNOT(wires=[control_wires[2],control_wires[4]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[4]])
            q.CNOT(wires=[control_wires[0],control_wires[2]])

            q.CNOT(wires=[control_wires[0],control_wires[3]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            q.CNOT(wires=[control_wires[0],control_wires[3]])
            q.CNOT(wires=[control_wires[0],control_wires[4]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[0],control_wires[4]])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[2],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            q.CNOT(wires=[control_wires[2],control_wires[4]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[4]])
            q.CNOT(wires=[control_wires[1],control_wires[2]])
            q.CNOT(wires=[control_wires[1],control_wires[3]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            q.CNOT(wires=[control_wires[1],control_wires[3]])
            q.CNOT(wires=[control_wires[1],control_wires[4]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[1],control_wires[4]])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[3],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            # C_1_U
            self.Controlled_U_block(alpha,beta,gamma,delta,\
                               delta+beta,delta-beta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            q.CNOT(wires=[control_wires[2],control_wires[3]])
            q.CNOT(wires=[control_wires[2],control_wires[4]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[2],control_wires[4]])
            q.CNOT(wires=[control_wires[3],control_wires[4]])
            # C_1_U_dagger
            self.Controlled_U_block(-alpha,-delta,-gamma,-beta,\
                               -beta-delta,-beta+delta,wires=[control_wires[4],operation_wire])
            q.CNOT(wires=[control_wires[3],control_wires[4]])


    def gray_code_C_X(self,wires,gray_code_element,changing_bit):
        
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
        self.C_U(U,control_wires=control_wires, operation_wire=wires[changing_bit])

        # flip qubit with PauliX if there is 0 in gray_code_element (but not the changing_bit)
        for i in range(len(wires)):
            if (gray_code_element[i] == '0')&(i != changing_bit):
                q.PauliX(wires=wires[i])
    
    
    # Implements Two_level_U given angles from ZY decomposition of U. building block for function U_n
    def Two_level_U(self,wires,U,non_trivial_indices):
        
        ### Transform U and non_trivial_indices
        # if U is passed as pennylane.Variable, then use type np.complex128
        if isinstance(U[0][0], q.variable.Variable):
            # no straight-forward't matrix-wise operation .val for ndarray of pennylane.Variables => use cycle
            for i in range(U.shape[0]):
                for j in range(U.shape[0]):
                    U[i][j] = U[i][j].val
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
        gray_code, changing_bit = NonQuantumOperations.Gray_code(self,a,b,len(wires))

        # circuit
        # Gray code forward sequence of C_Xs
        for i in range(0,len(gray_code)-2):
            self.gray_code_C_X(wires,gray_code[i],changing_bit[i])

        # C_U
        # flip qubit with PauliX if there is 0 in gray_code_element[-1] (but not the changing_bit)
        for i in range(len(wires)):
            if (gray_code[-1][i] == '0')&(i != changing_bit[-1]):
                q.PauliX(wires=wires[i])
        # define control_wires and operation_wire
        control_wires = copy.deepcopy(wires)
        del control_wires[changing_bit[-1]]
        self.C_U(U_non_trivial_submatrix,control_wires=control_wires, operation_wire=wires[changing_bit[-1]])
        # flip qubit with PauliX if there is 0 in gray_code_element[-1] (but not the changing_bit)
        for i in range(len(wires)):
            if (gray_code[-1][i] == '0')&(i != changing_bit[-1]):
                q.PauliX(wires=wires[i])

        # Gray code backward sequence of C_Xs
        for i in range(len(gray_code)-3,-1,-1):
            self.gray_code_C_X(wires,gray_code[i],changing_bit[i])
    
    
    # Arbitrary U circuit
    def U_n(self, wires, U):
        
        # check dimensionality of U
        if int(np.log2(U.shape[0])) != np.log2(U.shape[0]):
            raise Exception('Wrong shape of U: it should be 2**len(wires)')
        
        ### Transform U and non_trivial_indices
        # if U is passed as pennylane.Variable, then use type np.complex128
        if isinstance(U[0][0], q.variable.Variable):
            # no straight-forward't matrix-wise operation .val for ndarray of pennylane.Variables => use cycle
            for i in range(U.shape[0]):
                for j in range(U.shape[0]):
                    U[i][j] = U[i][j].val
        
        Two_level_U_list = NonQuantumOperations.Two_level_unitary_decomposition(self,U)
        
        # consequentially execute Two_level_Us
        # note that circuits should be in reverse order relative to matrix decomposition 
        I = np.eye(U.shape[0],dtype='complex128')
        for V in reversed(Two_level_U_list):
            # get non-trivial indices
            V_mask = np.isclose(V, I,atol=1e-15) == False
            non_trivial_indices = np.where(np.sum(V_mask,axis=1) == 2)[0]
            # execute Two_level_U(V)
            self.Two_level_U(wires,V,non_trivial_indices)
    
    
    # Implements Two_level_U given angles from ZY decomposition of U. Building block for function C_U_n
    def controlled_Two_level_U(self,control_wire,operation_wires,U,non_trivial_indices):
        
        ### Transform U and non_trivial_indices
        # if U is passed as pennylane.Variable, then use type np.complex128
        if isinstance(U[0][0], q.variable.Variable):
            # no straight-forward't matrix-wise operation .val for ndarray of pennylane.Variables => use cycle
            for i in range(U.shape[0]):
                for j in range(U.shape[0]):
                    U[i][j] = U[i][j].val
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
        gray_code, changing_bit = NonQuantumOperations.Gray_code(self,a,b,len(operation_wires))
        # create wires = [operation_wires, control_wire] to execute the same operations with refined gray_code elements
        wires = operation_wires + list([control_wire])
        # edit gray_code element to incorporate control_wire =
        # = add control_wire for every gray_code element as '1' in the end of code string
        # note that since control_wire is the last in the list of wires, there is no need to edit changing_bit list
        gray_code = [gray_code[i] + '1' for i in range(len(gray_code))]
        
        # circuit
        # Gray code forward sequence of C_Xs
        for i in range(0,len(gray_code)-2):
            self.gray_code_C_X(wires,gray_code[i],changing_bit[i])

        # C_U
        # flip qubit with PauliX if there is 0 in gray_code_element[-1] (but not the changing_bit)
        for i in range(len(wires)):
            if (gray_code[-1][i] == '0')&(i != changing_bit[-1]):
                q.PauliX(wires=wires[i])
        # define control_wires and operation_wire
        control_wires = copy.deepcopy(wires)
        del control_wires[changing_bit[-1]]
        self.C_U(U_non_trivial_submatrix,control_wires=control_wires, operation_wire=wires[changing_bit[-1]])
        # flip qubit with PauliX if there is 0 in gray_code_element[-1] (but not the changing_bit)
        for i in range(len(wires)):
            if (gray_code[-1][i] == '0')&(i != changing_bit[-1]):
                q.PauliX(wires=wires[i])

        # Gray code backward sequence of C_Xs
        for i in range(len(gray_code)-3,-1,-1):
            self.gray_code_C_X(wires,gray_code[i],changing_bit[i])
    
    
    # controlled n-qubit unitary U (one control wire, n operation wires)
    def C_U_n(self, control_wire, operation_wires, U):
        
        # check dimensionality of U
        if int(np.log2(U.shape[0])) != np.log2(U.shape[0]):
            raise Exception('Wrong shape of U: it should be 2**len(operation_wires)')
        
        ### Transform U and non_trivial_indices
        # if U is passed as pennylane.Variable, then use type np.complex128
        if isinstance(U[0][0], q.variable.Variable):
            # no straight-forward't matrix-wise operation .val for ndarray of pennylane.Variables => use cycle
            for i in range(U.shape[0]):
                for j in range(U.shape[0]):
                    U[i][j] = U[i][j].val
        
        Two_level_U_list = NonQuantumOperations.Two_level_unitary_decomposition(self,U)
        
        # consequentially execute controlled_Two_level_Us
        # note that circuits should be in reverse order relative to matrix decomposition 
        I = np.eye(U.shape[0],dtype='complex128')
        for V in reversed(Two_level_U_list):
            # get non-trivial indices
            V_mask = np.isclose(V, I,atol=1e-15) == False
            non_trivial_indices = np.where(np.sum(V_mask,axis=1) == 2)[0]
            # execute controlled_Two_level_U(V)
            self.controlled_Two_level_U(control_wire,operation_wires,V,non_trivial_indices)
    
#-------------------------------------------------#
#--- Functions implementing quantum algorithms ---#
#-------------------------------------------------#

class QuantumAlgorithms:
    
    def QFT(self):
        print('function will be here soon')