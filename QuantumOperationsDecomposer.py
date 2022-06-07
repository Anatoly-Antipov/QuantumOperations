import numpy as np
import pennylane as qml
from ClassicalOperations import ClassicalOperations

c = ClassicalOperations()

def Test(wires):
    op_list = list()
    
    op_list.append(qml.Hadamard(wires=wires[1]))
    op_list += SUM(wires=wires)
    
    return op_list


def SUM(wires):
    op_list = list()
    
    op_list.append(qml.CNOT(wires=[wires[1],wires[2]]))
    op_list.append(qml.CNOT(wires=[wires[0],wires[2]]))
    
    return op_list


def CARRY(wires):
    op_list = list()
    
    op_list.append(qml.Toffoli(wires=wires[1:]))
    op_list.append(qml.CNOT(wires=[wires[1],wires[2]]))
    op_list.append(qml.Toffoli(wires=[wires[0],wires[2],wires[3]]))
    
    return op_list


def CARRY_inv(wires):
    op_list = list()
    
    op_list.append(qml.Toffoli(wires=[wires[0],wires[2],wires[3]]))
    op_list.append(qml.CNOT(wires=[wires[1],wires[2]]))
    op_list.append(qml.Toffoli(wires=wires[1:]))
    
    return op_list

    
def ADDER(wires):
    op_list = list()
        
    # check inputs
    if (len(wires)-1)%3 != 0 or len(wires) < 4:
        raise Exception('Wrong number of wires: should be n wires for wires_a, n+1 wires for wires_b and n wires for wires_c')
    else:
        n = int((len(wires)-1)/3)
        wires_a = wires[:n]
        wires_b = wires[n:2*n+1]
        wires_c = wires[2*n+1:]
    

    # block of CARRY gates
    for i in range(len(wires_a)-1):
        op_list += CARRY(wires=[wires_c[i],wires_a[i],wires_b[i],wires_c[i+1]])
    op_list += CARRY(wires=[wires_c[len(wires_a)-1],wires_a[len(wires_a)-1],wires_b[len(wires_a)-1],wires_b[len(wires_a)]])

    op_list.append(qml.CNOT(wires=[wires_a[len(wires_a)-1],wires_b[len(wires_a)-1]]))

    # block of inverse-CARRY and SUM gates
    op_list += SUM(wires=[wires_c[len(wires_a)-1],wires_a[len(wires_a)-1],wires_b[len(wires_a)-1]])
    for i in range(len(wires_a)-2,-1,-1):
        op_list += CARRY_inv(wires=[wires_c[i],wires_a[i],wires_b[i],wires_c[i+1]])
        op_list += SUM(wires=[wires_c[i],wires_a[i],wires_b[i]])
    
    return op_list


def ADDER_inv(wires):
    op_list = list()
        
    # check inputs
    if (len(wires)-1)%3 != 0 or len(wires) < 4:
        raise Exception('Wrong number of wires: should be n wires for wires_a, n+1 wires for wires_b and n wires for wires_c')
    else:
        n = int((len(wires)-1)/3)
        wires_a = wires[:n]
        wires_b = wires[n:2*n+1]
        wires_c = wires[2*n+1:]


    # block of inverse-SUM and CARRY gates
    for i in range(len(wires_a)-1):
        op_list += SUM(wires=[wires_c[i],wires_a[i],wires_b[i]])
        op_list += CARRY(wires=[wires_c[i],wires_a[i],wires_b[i],wires_c[i+1]])
    op_list += SUM(wires=[wires_c[len(wires_a)-1],wires_a[len(wires_a)-1],wires_b[len(wires_a)-1]])

    op_list.append(qml.CNOT(wires=[wires_a[len(wires_a)-1],wires_b[len(wires_a)-1]]))

    # block of inverse-CARRY gates
    op_list += CARRY_inv(wires=[wires_c[len(wires_a)-1],wires_a[len(wires_a)-1],wires_b[len(wires_a)-1],wires_b[len(wires_a)]])
    for i in range(len(wires_a)-2,-1,-1):
        op_list += CARRY_inv(wires=[wires_c[i],wires_a[i],wires_b[i],wires_c[i+1]])
    
    return op_list


def Controlled_reset_zero_register_to_N(*parameters, wires):
    op_list = list()
        
    # check inputs
    N = int(parameters[0])
    control_wire = wires[0]
    wires_zero_register = wires[1:]

    # check if N does not match the size of wires_zero_register
    if N > 2**(len(wires_zero_register))-1:
        raise Exception('N is too big for the register wires_zero_register')
    # make a string with a binary represenatation of N (in reverse order)
    N = bin(N)[2:][::-1]
    # add zeros to match the register's size
    N = N + '0'*(len(wires_zero_register)-len(N))


    # CNOTs with control=control_wire and taget - wire in wires_zero_register, for which N == '1'
    for i in range(len(wires_zero_register)):
        if N[i] == '1':
            op_list.append(qml.CNOT(wires=[control_wire,wires_zero_register[i]]))
    
    return op_list


def ADDER_MOD(*parameters, wires):
    op_list = list()
        
    # check inputs
    # is it true that wires_a == wires_N ???
    if (len(wires)-2)%4 != 0 or len(wires) < 6:
        raise Exception('Wrong number of wires: should be n wires for wires_a, n+1 wires for wires_b, n wires for wires_c, n wires for wires_N and 1 wire for wires_t')
    else:
        n = int((len(wires)-2)/4)
        wires_a = wires[:n]
        wires_b = wires[n:2*n+1]
        wires_c = wires[2*n+1:3*n+1]
        wires_N = wires[3*n+1:4*n+1]
        wires_t = wires[-1]
        N = int(parameters[0])

    # check N
    # check if N does not match the size of wires_N
    if N > 2**(len(wires_N))-1:
        raise Exception('N is too big for the register wires_N')


    op_list += ADDER(wires=wires_a+wires_b+wires_c)
    op_list += ADDER_inv(wires=wires_N+wires_b+wires_c)

    op_list.append(qml.PauliX(wires=wires_b[-1]))
    op_list.append(qml.CNOT(wires=[wires_b[-1],wires_t]))
    op_list.append(qml.PauliX(wires=wires_b[-1]))

    op_list += Controlled_reset_zero_register_to_N(N,wires=[wires_t]+wires_N)
    op_list += ADDER(wires=wires_N+wires_b+wires_c)
    op_list += Controlled_reset_zero_register_to_N(N,wires=[wires_t]+wires_N)

    op_list += ADDER_inv(wires=wires_a+wires_b+wires_c)
    op_list.append(qml.CNOT(wires=[wires_b[-1],wires_t]))
    op_list += ADDER(wires=wires_a+wires_b+wires_c)
    
    return op_list


def ADDER_MOD_inv(*parameters, wires):
    op_list = list()
        
    # check inputs
    # is it true that wires_a == wires_N ???
    if (len(wires)-2)%4 != 0 or len(wires) < 6:
        raise Exception('Wrong number of wires: should be n wires for wires_a, n+1 wires for wires_b, n wires for wires_c, n wires for wires_N and 1 wire for wires_t')
    else:
        n = int((len(wires)-2)/4)
        wires_a = wires[:n]
        wires_b = wires[n:2*n+1]
        wires_c = wires[2*n+1:3*n+1]
        wires_N = wires[3*n+1:4*n+1]
        wires_t = wires[-1]
        N = int(parameters[0])

    # check N
    # check if N does not match the size of wires_N
    if N > 2**(len(wires_N))-1:
        raise Exception('N is too big for the register wires_N')


    op_list += ADDER_inv(wires=wires_a+wires_b+wires_c)
    op_list.append(qml.CNOT(wires=[wires_b[-1],wires_t]))
    op_list += ADDER(wires=wires_a+wires_b+wires_c)

    op_list += Controlled_reset_zero_register_to_N(N,wires=[wires_t]+wires_N)
    op_list += ADDER_inv(wires=wires_N+wires_b+wires_c)
    op_list += Controlled_reset_zero_register_to_N(N,wires=[wires_t]+wires_N)

    op_list.append(qml.PauliX(wires=wires_b[-1]))
    op_list.append(qml.CNOT(wires=[wires_b[-1],wires_t]))
    op_list.append(qml.PauliX(wires=wires_b[-1]))

    op_list += ADDER(wires=wires_N+wires_b+wires_c)
    op_list += ADDER_inv(wires=wires_a+wires_b+wires_c)
    
    return op_list


# input parameters: N,m
def Ctrl_MULT_MOD(*parameters, wires):
    op_list = list()

    # check wires
    if (len(wires)-3)%5 != 0 or len(wires) < 8:
        raise Exception('Wrong size of registers - it should be: \n 1 wire for the control register (control_wire) \n n wires for the register with z (wires_z) \n n wires for the register with a (wires_a) \n n+1 wires for the register with b (wires_b) \n n wires for the register with |0..0> (wires_c) \n n wires for the register with N (wires_N) \n 1 wire for the register with t=|0> (wires_t)')
    else:
        n = int((len(wires)-3)/5)
        control_wire = wires[0]
        wires_z = wires[1:n+1]
        wires_a = wires[n+1:2*n+1]
        wires_b = wires[2*n+1:3*n+2]
        wires_c = wires[3*n+2:4*n+2]
        wires_N = wires[4*n+2:5*n+2]
        wires_t = wires[-1]
        N = int(parameters[0])
        m = int(parameters[1])

    # check N
    # check if N does not match the size of wires_N
    if N > 2**(len(wires_N))-1:
        raise Exception('N is too big for the register wires_N')


    ### block with ADDER_MODs
    # cycle to iteratively add ADDER_MODs each surrounded by Toffoli gates with control1=control_wire, control2=wires_z[i] and target - wire in wires_a, for which m*2^i == '1'
    for i in range(len(wires_z)):
        # binary representation of m*2^i mod N
        m_2_to_power_i_mod_N = bin((m*(2**i)) % N)[2:][::-1]
        m_2_to_power_i_mod_N = m_2_to_power_i_mod_N + '0'*( len(wires_a)-len(m_2_to_power_i_mod_N) )

        # cycle for Toffoli gates before ADDER_MOD to put m*2^i to wires_a if control_wire == 1
        for j in range(len(wires_a)):
            if m_2_to_power_i_mod_N[j] == '1':
                op_list.append(qml.Toffoli(wires=[control_wire,wires_z[i],wires_a[j]]))

        # ADDER_MOD[i]
        op_list += ADDER_MOD(N,wires=wires_a+wires_b+wires_c+wires_N+[wires_t])

        # cycle for Toffoli gates after ADDER_MOD to make wires_a |0..0> if control_wire == 1
        for j in range(len(wires_a)):
            if m_2_to_power_i_mod_N[j] == '1':
                op_list.append(qml.Toffoli(wires=[control_wire,wires_z[i],wires_a[j]]))

    ### block for copying z into wires_a conditional on control_wire == |0>. That is, we want |0,z,0> -> |0,z,z>
    op_list.append(qml.PauliX(wires=control_wire))
    for i in range(len(wires_z)):
        op_list.append(qml.Toffoli(wires=[control_wire,wires_z[i],wires_a[i]]))
    op_list.append(qml.PauliX(wires=control_wire))
    
    return op_list


# input parameters: N,m
def Ctrl_MULT_MOD_inv(*parameters, wires):
    op_list = list()
        
    # check wires
    if (len(wires)-3)%5 != 0 or len(wires) < 8:
        raise Exception('Wrong size of registers - it should be: \n 1 wire for the control register (control_wire) \n n wires for the register with z (wires_z) \n n wires for the register with a (wires_a) \n n+1 wires for the register with b (wires_b) \n n wires for the register with |0..0> (wires_c) \n n wires for the register with N (wires_N) \n 1 wire for the register with t=|0> (wires_t)')
    else:
        n = int((len(wires)-3)/5)
        control_wire = wires[0]
        wires_z = wires[1:n+1]
        wires_a = wires[n+1:2*n+1]
        wires_b = wires[2*n+1:3*n+2]
        wires_c = wires[3*n+2:4*n+2]
        wires_N = wires[4*n+2:5*n+2]
        wires_t = wires[-1]
        N = int(parameters[0])
        m = int(parameters[1])

    # check N
    # check if N does not match the size of wires_N
    if N > 2**(len(wires_N))-1:
        raise Exception('N is too big for the register wires_N')


    ### block for copying z into wires_a conditional on control_wire == |0>
    op_list.append(qml.PauliX(wires=control_wire))
    for i in range(len(wires_z)):
        op_list.append(qml.Toffoli(wires=[control_wire,wires_z[i],wires_a[i]]))
    op_list.append(qml.PauliX(wires=control_wire))

    for i in range(len(wires_z)-1,-1,-1):
        # binary representation of m*2^i mod N
        m_2_to_power_i_mod_N = bin((m*(2**i)) % N)[2:][::-1]
        m_2_to_power_i_mod_N = m_2_to_power_i_mod_N + '0'*( len(wires_a)-len(m_2_to_power_i_mod_N) )

        # cycle for Toffoli gates after ADDER_MOD to make wires_a |0..0> if control_wire == 1
        for j in range(len(wires_a)):
            if m_2_to_power_i_mod_N[j] == '1':
                op_list.append(qml.Toffoli(wires=[control_wire,wires_z[i],wires_a[j]]))

        # ADDER_MOD[i]
        op_list += ADDER_MOD_inv(N,wires=wires_a+wires_b+wires_c+wires_N+[wires_t])

        # cycle for Toffoli gates before ADDER_MOD to put m*2^i to wires_a if control_wire == 1
        for j in range(len(wires_a)):
            if m_2_to_power_i_mod_N[j] == '1':
                op_list.append(qml.Toffoli(wires=[control_wire,wires_z[i],wires_a[j]]))
    
    return op_list

        
# control_wire = wire[0], swap_wires = wire[1:]
def Ctrl_SWAP(wires):
    op_list = list()
        
    control_wire = wires[0]

    op_list.append(qml.CNOT(wires=[wires[1],wires[2]]))
    op_list.append(qml.Toffoli(wires=[wires[0],wires[2],wires[1]]))
    op_list.append(qml.CNOT(wires=[wires[1],wires[2]]))
    
    return op_list


# input parameters: N,y
def MODULAR_EXPONENTIATION(*parameters, wires):
    op_list = list()
        
    # check wires
    n_x = int(parameters[2])
    if (len(wires)-2-n_x)%5 != 0:
        raise Exception('Wrong size of registers - it should be: \n n_x wires for the register with x (wires_x) \n n wires for the register with z (wires_z) \n n wires for the register with a (wires_a) \n n+1 wires for the register with b (wires_b) \n n wires for the register with |0..0> (wires_c) \n n wires for the register with N (wires_N) \n 1 wire for the register with t=|0> (wires_t)')
    else:
        n = int((len(wires)-2-n_x)/5)
        N = int(parameters[0])
        y = int(parameters[1])
        wires_x = wires[0:n_x]
        wires_z = wires[n_x:n_x+n]
        wires_a = wires[n_x+n:n_x+2*n]
        wires_b = wires[n_x+2*n:n_x+3*n+1]
        wires_c = wires[n_x+3*n+1:n_x+4*n+1]
        wires_N = wires[n_x+4*n+1:n_x+5*n+1]
        wires_t = wires[-1]

    # check inputs
    # check N
    # check if N does not match the size of wires_N
    if N > 2**(len(wires_N))-1:
        raise Exception('N is too big for the register wires_N')
        
        
    for i in range(len(wires_x)):
        op_list += Ctrl_MULT_MOD(N,y**(2**i),wires=[wires_x[i]]+wires_z+wires_a+wires_b+wires_c+wires_N+[wires_t])
        # SWAP register wires_z with wires_b[:-1]
        for j in range(len(wires_z)):
            op_list += Ctrl_SWAP(wires=[wires_x[i],wires_z[j],wires_b[j]])
        # find modular multiplicative inverse of y**(2**i)
        inverse_y_2_i = c.modular_multiplicative_inverse(a=y**(2**i),N=N)
        op_list += Ctrl_MULT_MOD_inv(N,inverse_y_2_i,wires=[wires_x[i]]+wires_z+wires_a+wires_b+wires_c+wires_N+[wires_t])
    
    return op_list


def CR_k(*parameters, wires):
    op_list = list()
    
    k = int(parameters[0])

    op_list.append(qml.RZ(np.pi/(2**k), wires=wires[1]))
    op_list.append(qml.CNOT(wires=wires))
    op_list.append(qml.RZ(-np.pi/(2**k), wires=wires[1]))
    op_list.append(qml.CNOT(wires=wires))
    op_list.append(qml.PhaseShift(np.pi/(2**k),wires=wires[0]))
    
    return op_list


def CR_k_inv(*parameters, wires):
    op_list = list()

    k = int(parameters[0])


    op_list.append(qml.RZ(-np.pi/(2**k), wires=wires[1]))
    op_list.append(qml.CNOT(wires=wires))
    op_list.append(qml.RZ(np.pi/(2**k), wires=wires[1]))
    op_list.append(qml.CNOT(wires=wires))
    op_list.append(qml.PhaseShift(-np.pi/(2**k),wires=wires[0]))
    
    return op_list


def QFT_(wires):
    op_list = list()


    # block of Hadamard and CR_k gates
    for i in range(len(wires)):
        op_list.append(qml.Hadamard(wires=wires[i]))
        for k in range(2,(len(wires)-i)+1):
            op_list += CR_k(k,wires=[wires[i+(k-1)],wires[i]])

    # block of SWAP gates
    # works both if len(wires)%2 == 0 or len(wires)%2 == 1
    for i in range(int(len(wires)/2)):
        op_list.append(qml.SWAP(wires=[wires[i],wires[len(wires)-(i+1)]]))
    
    return op_list


def QFT_inv(wires):
    op_list = list()


    # block of Hadamard and CR_k gates
    for i in range(len(wires)):
        op_list.append(qml.Hadamard(wires=wires[i]))
        for k in range(2,(len(wires)-i)+1):
            op_list += CR_k_inv(k,wires=[wires[i+(k-1)],wires[i]])

    # block of SWAP gates
    # works both if len(wires)%2 == 0 or len(wires)%2 == 1
    for i in range(int(len(wires)/2)):
        op_list.append(qml.SWAP(wires=[wires[i],wires[len(wires)-(i+1)]]))
    
    return op_list


# input parameters: N,y
def Order_Finding(*parameters, wires):
    op_list = list()

    # check wires and define registers
    n_x = int(parameters[2])
    if (len(wires)-2-n_x)%5 != 0:
        raise Exception('Wrong size of registers - it should be: \n n_x wires for the register with x (wires_x) \n n wires for the register with z (wires_z) \n n wires for the register with a (wires_a) \n n+1 wires for the register with b (wires_b) \n n wires for the register with |0..0> (wires_c) \n n wires for the register with N (wires_N) \n 1 wire for the register with t=|0> (wires_t)')
    else:
        N = int(parameters[0])
        y = int(parameters[1])
        n = int((len(wires)-2-n_x)/5)
        wires_x = wires[0:n_x]
        wires_z = wires[n_x:n_x+n]
        wires_a = wires[n_x+n:n_x+2*n]
        wires_b = wires[n_x+2*n:n_x+3*n+1]
        wires_c = wires[n_x+3*n+1:n_x+4*n+1]
        wires_N = wires[n_x+4*n+1:n_x+5*n+1]
        wires_t = wires[-1]

    # check inputs
    # check N
    # check if N does not match the size of wires_N
    if N > 2**(len(wires_N))-1:
        raise Exception('N is too big for the register wires_N')


    # Create superposition with Hadamard gates 
    for i in range(len(wires_x)):
        op_list.append(qml.Hadamard(wires=wires_x[i]))

    # Apply modular exponentiation
    op_list += MODULAR_EXPONENTIATION(N,y,n_x,wires=wires_x+wires_z+wires_a+wires_b+wires_c+wires_N+[wires_t])

    # Apply inverse Quantum Fourier transform to the first register
    op_list += QFT_inv(wires=wires_x)
    
    return op_list


# realization of the operation R (native for trapped-ion qubit) via qml.Rot
def R(*parameters, wires):
    op_list = list()

    alpha = parameters[0]
    beta = parameters[1]
    phi = parameters[2]


    op_list.append(qml.Rot((phi - alpha + 0.5)*np.pi, beta*np.pi, (phi + alpha - 0.5)*np.pi, wires=wires))
    
    return op_list