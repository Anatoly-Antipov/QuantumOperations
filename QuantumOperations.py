import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from ClassicalOperations import ClassicalOperations

c = ClassicalOperations()

class SUM(Operation):

    num_params = 0
    num_wires = 3
    par_domain = None

    def expand(self):
        with qml.tape.QuantumTape() as tape:
            qml.CNOT(wires=[self.wires[1],self.wires[2]])
            qml.CNOT(wires=[self.wires[0],self.wires[2]])
        return tape

class CARRY(Operation):

    num_params = 0
    num_wires = 4
    par_domain = None

    def expand(self):
        with qml.tape.QuantumTape() as tape:
            qml.Toffoli(wires=self.wires[1:])
            qml.CNOT(wires=[self.wires[1],self.wires[2]])
            qml.Toffoli(wires=[self.wires[0],self.wires[2],self.wires[3]])
        return tape

class CARRY_inv(Operation):

    num_params = 0
    num_wires = 4
    par_domain = None

    def expand(self):
        with qml.tape.QuantumTape() as tape:
            qml.Toffoli(wires=[self.wires[0],self.wires[2],self.wires[3]])
            qml.CNOT(wires=[self.wires[1],self.wires[2]])
            qml.Toffoli(wires=self.wires[1:])
        return tape

class ADDER(Operation):

    num_params = 0
    num_wires = AnyWires
    par_domain = None

    def expand(self):
        # check inputs
        if (len(self.wires)-1)%3 != 0 or len(self.wires) < 4:
            raise Exception('Wrong number of wires: should be n wires for wires_a, n+1 wires for wires_b and n wires for wires_c')
        else:
            n = int((len(self.wires)-1)/3)
            wires_a = self.wires[:n]
            wires_b = self.wires[n:2*n+1]
            wires_c = self.wires[2*n+1:]
            
        with qml.tape.QuantumTape() as tape:
        
            # block of CARRY gates
            for i in range(len(wires_a)-1):
                CARRY(wires=[wires_c[i],wires_a[i],wires_b[i],wires_c[i+1]])
            CARRY(wires=[wires_c[len(wires_a)-1],wires_a[len(wires_a)-1],wires_b[len(wires_a)-1],wires_b[len(wires_a)]])

            qml.CNOT(wires=[wires_a[len(wires_a)-1],wires_b[len(wires_a)-1]])

            # block of inverse-CARRY and SUM gates
            SUM(wires=[wires_c[len(wires_a)-1],wires_a[len(wires_a)-1],wires_b[len(wires_a)-1]])
            for i in range(len(wires_a)-2,-1,-1):
                CARRY_inv(wires=[wires_c[i],wires_a[i],wires_b[i],wires_c[i+1]])
                SUM(wires=[wires_c[i],wires_a[i],wires_b[i]])
            
        return tape

class ADDER_inv(Operation):

    num_params = 0
    num_wires = AnyWires
    par_domain = None

    def expand(self):
        # check inputs
        if (len(self.wires)-1)%3 != 0 or len(self.wires) < 4:
            raise Exception('Wrong number of wires: should be n wires for wires_a, n+1 wires for wires_b and n wires for wires_c')
        else:
            n = int((len(self.wires)-1)/3)
            wires_a = self.wires[:n]
            wires_b = self.wires[n:2*n+1]
            wires_c = self.wires[2*n+1:]
            
        with qml.tape.QuantumTape() as tape:
        
            # block of inverse-SUM and CARRY gates
            for i in range(len(wires_a)-1):
                SUM(wires=[wires_c[i],wires_a[i],wires_b[i]])
                CARRY(wires=[wires_c[i],wires_a[i],wires_b[i],wires_c[i+1]])
            SUM(wires=[wires_c[len(wires_a)-1],wires_a[len(wires_a)-1],wires_b[len(wires_a)-1]])
            
            qml.CNOT(wires=[wires_a[len(wires_a)-1],wires_b[len(wires_a)-1]])
            
            # block of inverse-CARRY gates
            CARRY_inv(wires=[wires_c[len(wires_a)-1],wires_a[len(wires_a)-1],wires_b[len(wires_a)-1],wires_b[len(wires_a)]])
            for i in range(len(wires_a)-2,-1,-1):
                CARRY_inv(wires=[wires_c[i],wires_a[i],wires_b[i],wires_c[i+1]])
                
        return tape

class Controlled_reset_zero_register_to_N(Operation):

    num_params = 1
    num_wires = AnyWires
    par_domain = None

    def expand(self):
        # check inputs
        N = int(self.parameters[0])
        control_wire = self.wires[0]
        wires_zero_register = self.wires[1:]
        
        # check if N does not match the size of wires_zero_register
        if N > 2**(len(wires_zero_register))-1:
            raise Exception('N is too big for the register wires_zero_register')
        # make a string with a binary represenatation of N (in reverse order)
        N = bin(N)[2:][::-1]
        # add zeros to match the register's size
        N = N + '0'*(len(wires_zero_register)-len(N))
            
        with qml.tape.QuantumTape() as tape:
        
            # CNOTs with control=control_wire and taget - wire in wires_zero_register, for which N == '1'
            for i in range(len(wires_zero_register)):
                if N[i] == '1':
                    qml.CNOT(wires=[control_wire,wires_zero_register[i]])
            
        return tape

class ADDER_MOD(Operation):

    num_params = 1
    num_wires = AnyWires
    par_domain = None

    def expand(self):
        # check inputs
        # is it true that wires_a == wires_N ???
        if (len(self.wires)-2)%4 != 0 or len(self.wires) < 6:
            raise Exception('Wrong number of wires: should be n wires for wires_a, n+1 wires for wires_b, n wires for wires_c, n wires for wires_N and 1 wire for wires_t')
        else:
            n = int((len(self.wires)-2)/4)
            wires_a = self.wires[:n]
            wires_b = self.wires[n:2*n+1]
            wires_c = self.wires[2*n+1:3*n+1]
            wires_N = self.wires[3*n+1:4*n+1]
            wires_t = self.wires[-1]
            N = int(self.parameters[0])
        
        # check N
        # check if N does not match the size of wires_N
        if N > 2**(len(wires_N))-1:
            raise Exception('N is too big for the register wires_N')
        
        with qml.tape.QuantumTape() as tape:
        
            ADDER(wires=wires_a+wires_b+wires_c)
            ADDER_inv(wires=wires_N+wires_b+wires_c)
            
            qml.PauliX(wires=wires_b[-1])
            qml.CNOT(wires=[wires_b[-1],wires_t])
            qml.PauliX(wires=wires_b[-1])
            
            Controlled_reset_zero_register_to_N(N,wires=[wires_t]+wires_N)
            ADDER(wires=wires_N+wires_b+wires_c)
            Controlled_reset_zero_register_to_N(N,wires=[wires_t]+wires_N)
            
            ADDER_inv(wires=wires_a+wires_b+wires_c)
            qml.CNOT(wires=[wires_b[-1],wires_t])
            ADDER(wires=wires_a+wires_b+wires_c)
            
        return tape

class ADDER_MOD_inv(Operation):

    num_params = 1
    num_wires = AnyWires
    par_domain = None

    def expand(self):
        # check inputs
        # is it true that wires_a == wires_N ???
        if (len(self.wires)-2)%4 != 0 or len(self.wires) < 6:
            raise Exception('Wrong number of wires: should be n wires for wires_a, n+1 wires for wires_b, n wires for wires_c, n wires for wires_N and 1 wire for wires_t')
        else:
            n = int((len(self.wires)-2)/4)
            wires_a = self.wires[:n]
            wires_b = self.wires[n:2*n+1]
            wires_c = self.wires[2*n+1:3*n+1]
            wires_N = self.wires[3*n+1:4*n+1]
            wires_t = self.wires[-1]
            N = int(self.parameters[0])
        
        # check N
        # check if N does not match the size of wires_N
        if N > 2**(len(wires_N))-1:
            raise Exception('N is too big for the register wires_N')
        
        with qml.tape.QuantumTape() as tape:
        
            ADDER_inv(wires=wires_a+wires_b+wires_c)
            qml.CNOT(wires=[wires_b[-1],wires_t])
            ADDER(wires=wires_a+wires_b+wires_c)
            
            Controlled_reset_zero_register_to_N(N,wires=[wires_t]+wires_N)
            ADDER_inv(wires=wires_N+wires_b+wires_c)
            Controlled_reset_zero_register_to_N(N,wires=[wires_t]+wires_N)
            
            qml.PauliX(wires=wires_b[-1])
            qml.CNOT(wires=[wires_b[-1],wires_t])
            qml.PauliX(wires=wires_b[-1])
            
            ADDER(wires=wires_N+wires_b+wires_c)
            ADDER_inv(wires=wires_a+wires_b+wires_c)
            
        return tape

# input parameters: N,m
class Ctrl_MULT_MOD(Operation):

    num_params = 2
    num_wires = AnyWires
    par_domain = None

    def expand(self):
        
        # check wires
        if (len(self.wires)-3)%5 != 0 or len(self.wires) < 8:
            raise Exception('Wrong size of registers - it should be: \n 1 wire for the control register (control_wire) \n n wires for the register with z (wires_z) \n n wires for the register with a (wires_a) \n n+1 wires for the register with b (wires_b) \n n wires for the register with |0..0> (wires_c) \n n wires for the register with N (wires_N) \n 1 wire for the register with t=|0> (wires_t)')
        else:
            n = int((len(self.wires)-3)/5)
            control_wire = self.wires[0]
            wires_z = self.wires[1:n+1]
            wires_a = self.wires[n+1:2*n+1]
            wires_b = self.wires[2*n+1:3*n+2]
            wires_c = self.wires[3*n+2:4*n+2]
            wires_N = self.wires[4*n+2:5*n+2]
            wires_t = self.wires[-1]
            N = int(self.parameters[0])
            m = int(self.parameters[1])
        
        # check N
        # check if N does not match the size of wires_N
        if N > 2**(len(wires_N))-1:
            raise Exception('N is too big for the register wires_N')
        
        with qml.tape.QuantumTape() as tape:
        
            ### block with ADDER_MODs
            # cycle to iteratively add ADDER_MODs each surrounded by Toffoli gates with control1=control_wire, control2=self.wires_z[i] and target - wire in wires_a, for which m*2^i == '1'
            for i in range(len(wires_z)):
                # binary representation of m*2^i mod N
                m_2_to_power_i_mod_N = bin((m*(2**i)) % N)[2:][::-1]
                m_2_to_power_i_mod_N = m_2_to_power_i_mod_N + '0'*( len(wires_a)-len(m_2_to_power_i_mod_N) )
                
                # cycle for Toffoli gates before ADDER_MOD to put m*2^i to wires_a if control_wire == 1
                for j in range(len(wires_a)):
                    if m_2_to_power_i_mod_N[j] == '1':
                        qml.Toffoli(wires=[control_wire,wires_z[i],wires_a[j]])

                # ADDER_MOD[i]
                ADDER_MOD(N,wires=wires_a+wires_b+wires_c+wires_N+[wires_t])

                # cycle for Toffoli gates after ADDER_MOD to make wires_a |0..0> if control_wire == 1
                for j in range(len(wires_a)):
                    if m_2_to_power_i_mod_N[j] == '1':
                        qml.Toffoli(wires=[control_wire,wires_z[i],wires_a[j]])

            ### block for copying z into wires_a conditional on control_wire == |0>. That is, we want |0,z,0> -> |0,z,z>
            qml.PauliX(wires=control_wire)
            for i in range(len(wires_z)):
                qml.Toffoli(wires=[control_wire,wires_z[i],wires_a[i]])
            qml.PauliX(wires=control_wire)
            
        return tape

# input parameters: N,m
class Ctrl_MULT_MOD_inv(Operation):

    num_params = 2
    num_wires = AnyWires
    par_domain = None

    def expand(self):
        
        # check wires
        if (len(self.wires)-3)%5 != 0 or len(self.wires) < 8:
            raise Exception('Wrong size of registers - it should be: \n 1 wire for the control register (control_wire) \n n wires for the register with z (wires_z) \n n wires for the register with a (wires_a) \n n+1 wires for the register with b (wires_b) \n n wires for the register with |0..0> (wires_c) \n n wires for the register with N (wires_N) \n 1 wire for the register with t=|0> (wires_t)')
        else:
            n = int((len(self.wires)-3)/5)
            control_wire = self.wires[0]
            wires_z = self.wires[1:n+1]
            wires_a = self.wires[n+1:2*n+1]
            wires_b = self.wires[2*n+1:3*n+2]
            wires_c = self.wires[3*n+2:4*n+2]
            wires_N = self.wires[4*n+2:5*n+2]
            wires_t = self.wires[-1]
            N = int(self.parameters[0])
            m = int(self.parameters[1])
        
        # check N
        # check if N does not match the size of wires_N
        if N > 2**(len(wires_N))-1:
            raise Exception('N is too big for the register wires_N')
        
        with qml.tape.QuantumTape() as tape:
        
            ### block for copying z into wires_a conditional on control_wire == |0>
            qml.PauliX(wires=control_wire)
            for i in range(len(wires_z)):
                qml.Toffoli(wires=[control_wire,wires_z[i],wires_a[i]])
            qml.PauliX(wires=control_wire)
            
            for i in range(len(wires_z)-1,-1,-1):
                # binary representation of m*2^i mod N
                m_2_to_power_i_mod_N = bin((m*(2**i)) % N)[2:][::-1]
                m_2_to_power_i_mod_N = m_2_to_power_i_mod_N + '0'*( len(wires_a)-len(m_2_to_power_i_mod_N) )
                
                # cycle for Toffoli gates after ADDER_MOD to make wires_a |0..0> if control_wire == 1
                for j in range(len(wires_a)):
                    if m_2_to_power_i_mod_N[j] == '1':
                        qml.Toffoli(wires=[control_wire,wires_z[i],wires_a[j]])
                
                # ADDER_MOD[i]
                ADDER_MOD_inv(N,wires=wires_a+wires_b+wires_c+wires_N+[wires_t])
                
                # cycle for Toffoli gates before ADDER_MOD to put m*2^i to wires_a if control_wire == 1
                for j in range(len(wires_a)):
                    if m_2_to_power_i_mod_N[j] == '1':
                        qml.Toffoli(wires=[control_wire,wires_z[i],wires_a[j]])
            
        return tape
        
# control_wire = self.wire[0], swap_wires = self.wire[1:]
class Ctrl_SWAP(Operation):

    num_params = 0
    num_wires = 3
    par_domain = None

    def expand(self):
        
        control_wire = self.wires[0]
        
        with qml.tape.QuantumTape() as tape:
            qml.Toffoli(wires=[self.wires[0],self.wires[1],self.wires[2]])
            qml.Toffoli(wires=[self.wires[0],self.wires[2],self.wires[1]])
            qml.Toffoli(wires=[self.wires[0],self.wires[1],self.wires[2]])
        return tape

# input parameters: N,y
class MODULAR_EXPONENTIATION(Operation):

    num_params = 2
    num_wires = AnyWires
    par_domain = None

    def expand(self):
        
        # check wires
        if (len(self.wires)-2)%6 !=0 or len(self.wires) < 8:
            raise Exception('Wrong size of registers - it should be: \n n wires for the register with x (wires_x) \n n wires for the register with z (wires_z) \n n wires for the register with a (wires_a) \n n+1 wires for the register with b (wires_b) \n n wires for the register with |0..0> (wires_c) \n n wires for the register with N (wires_N) \n 1 wire for the register with t=|0> (wires_t)')
        else:
            n = int((len(self.wires)-2)/6)
            wires_x = self.wires[0:n]
            wires_z = self.wires[n:2*n]
            wires_a = self.wires[2*n:3*n]
            wires_b = self.wires[3*n:4*n+1]
            wires_c = self.wires[4*n+1:5*n+1]
            wires_N = self.wires[5*n+1:6*n+1]
            wires_t = self.wires[-1]
            N = int(self.parameters[0])
            y = int(self.parameters[1])
        
        # check inputs
        # check N
        # check if N does not match the size of wires_N
        if N > 2**(len(wires_N))-1:
            raise Exception('N is too big for the register wires_N')
        
        with qml.tape.QuantumTape() as tape:
        
            for i in range(len(wires_x)):
                Ctrl_MULT_MOD(N,y**(2**i),wires=[wires_x[i]]+wires_z+wires_a+wires_b+wires_c+wires_N+[wires_t])
                # SWAP register wires_z with wires_b[:-1]
                for j in range(len(wires_z)):
                    Ctrl_SWAP(wires=[wires_x[i],wires_z[j],wires_b[j]])
                # find modular multiplicative inverse of y**(2**i)
                inverse_y_2_i = c.modular_multiplicative_inverse(a=y**(2**i),N=N)
                Ctrl_MULT_MOD_inv(N,inverse_y_2_i,wires=[wires_x[i]]+wires_z+wires_a+wires_b+wires_c+wires_N+[wires_t])
            
        return tape

class CR_k(Operation):

    num_params = 1
    num_wires = 2
    par_domain = None

    def expand(self):
        
        k = int(self.parameters[0])
        
        with qml.tape.QuantumTape() as tape:
            qml.RZ(np.pi/(2**k), wires=self.wires[1])
            qml.CNOT(wires=self.wires)
            qml.RZ(-np.pi/(2**k), wires=self.wires[1])
            qml.CNOT(wires=self.wires)
            qml.PhaseShift(np.pi/(2**k),wires=self.wires[0])
        return tape

class CR_k_inv(Operation):

    num_params = 1
    num_wires = 2
    par_domain = None

    def expand(self):
        
        k = int(self.parameters[0])
        
        with qml.tape.QuantumTape() as tape:
            qml.RZ(-np.pi/(2**k), wires=self.wires[1])
            qml.CNOT(wires=self.wires)
            qml.RZ(np.pi/(2**k), wires=self.wires[1])
            qml.CNOT(wires=self.wires)
            qml.PhaseShift(-np.pi/(2**k),wires=self.wires[0])
        return tape

class QFT_(Operation):

    num_params = 0
    num_wires = AnyWires
    par_domain = None

    def expand(self):
        with qml.tape.QuantumTape() as tape:
            # block of Hadamard and CR_k gates
            for i in range(len(self.wires)):
                qml.Hadamard(wires=self.wires[i])
                for k in range(2,(len(self.wires)-i)+1):
                    CR_k(k,wires=[self.wires[i+(k-1)],self.wires[i]])

            # block of SWAP gates
            # works both if len(wires)%2 == 0 or len(wires)%2 == 1
            for i in range(int(len(self.wires)/2)):
                qml.SWAP(wires=[self.wires[i],self.wires[len(self.wires)-(i+1)]])
        return tape

class QFT_inv(Operation):

    num_params = 0
    num_wires = AnyWires
    par_domain = None

    def expand(self):
        with qml.tape.QuantumTape() as tape:
            # block of Hadamard and CR_k gates
            for i in range(len(self.wires)):
                qml.Hadamard(wires=self.wires[i])
                for k in range(2,(len(self.wires)-i)+1):
                    CR_k_inv(k,wires=[self.wires[i+(k-1)],self.wires[i]])

            # block of SWAP gates
            # works both if len(wires)%2 == 0 or len(wires)%2 == 1
            for i in range(int(len(self.wires)/2)):
                qml.SWAP(wires=[self.wires[i],self.wires[len(self.wires)-(i+1)]])
        return tape

# input parameters: N,y
class Order_Finding(Operation):

    num_params = 2
    num_wires = AnyWires
    par_domain = None

    def expand(self):
        
        # check wires and define registers
        if (len(self.wires)-2)%6 !=0 or len(self.wires) < 8:
            raise Exception('Wrong size of registers - it should be: \n n wires for the register with x (wires_x) \n n wires for the register with z (wires_z) \n n wires for the register with a (wires_a) \n n+1 wires for the register with b (wires_b) \n n wires for the register with |0..0> (wires_c) \n n wires for the register with N (wires_N) \n 1 wire for the register with t=|0> (wires_t)')
        else:
            N = int(self.parameters[0])
            y = int(self.parameters[1])
            n = int((len(self.wires)-2)/6)
            wires_x = self.wires[:n]
            wires_z = self.wires[n:2*n]
            wires_a = self.wires[2*n:3*n]
            wires_b = self.wires[3*n:4*n+1]
            wires_c = self.wires[4*n+1:5*n+1]
            wires_N = self.wires[5*n+1:6*n+1]
            wires_t = self.wires[-1]
        
        # check inputs
        # check N
        # check if N does not match the size of wires_N
        if N > 2**(len(wires_N))-1:
            raise Exception('N is too big for the register wires_N')
        
        with qml.tape.QuantumTape() as tape:

            # Create superposition with Hadamard gates 
            for i in range(n):
                qml.Hadamard(wires=wires_x[i])

            # Apply modular exponentiation
            MODULAR_EXPONENTIATION(N,y,wires=wires_x+wires_z+wires_a+wires_b+wires_c+wires_N+[wires_t])

            # Apply inverse Quantum Fourier transform to the first register
            QFT_inv(wires=wires_x)
            
        return tape