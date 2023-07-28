import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from numpy import linalg as LA
import pandas as pd
from scipy import integrate
from scipy.linalg import logm, expm
from time import time



# ===========================================================================================================
# ===================================== Defining few functions for ease ====================================
# ==========================================================================================================

# --------- Conjugate Transpose --------

def conjT(A # Matrix 
         ):
    B = np.conj(A)
    B = np.transpose(B)
    return B

# --------- Partial Trace of bi-partite system ---------

def partial_trace(rho # Density Matrix
                  ,n1 # dimension of first hilbert space
                  ,n2, # dimension of second hilber space
                  x = 1 # partial trace with respect to x space
                 ):
    rho_tensor = rho.reshape([n1,n2,n1,n2])
    if x ==1:    
        Tr_partial = np.trace(rho_tensor,axis1 = 1,axis2 = 3)
    else:
        Tr_partial = np.trace(rho_tensor,axis1 = 0,axis2 = 2)
    return Tr_partial

# --- Vector norm and Vector product

def vec_norm(v # column vector
            ):
    norm = conjT(v)@v
    return np.sqrt(norm[0,0])

def vec_prod(u,v # column vectors
            ):
    norm = conjT(u)@v
    return norm[0,0]




# =====================================================================================
# ============================ Discrete-Time Quantum Walk ==============================
# ======================================================================================


def DQW(θ, # coin angle
        N = 100 # number of steps
       ):
    
    P = 2*N+1    # number of positions

#     print('\n')
#     print('Running quantum walk for ',N, 'steps.')
#     print('\n')


    coin0 = np.array([[1], [0]])  # |0>
    coin1 = np.array([[0], [1]])  # |1>

    C00 = np.outer(coin0, coin0)  # |0><0| 
    C01 = np.outer(coin0, coin1)  # |0><1| 
    C10 = np.outer(coin1, coin0)  # |1><0| 
    C11 = np.outer(coin1, coin1)  # |1><1| 

    # ----------- Localized -----------

    ψx = np.zeros((P,1))
    ψx[N,0] = 1     # array indexing starts from 0, so index N is the central posn

    #------------- Momentum eigenstate ----------

    # k = 0.1

    # posn0 = np.ones(P,dtype=complex)

    # for i in range(len(posn0)):
    #     posn0[i] = np.exp(-1j*k*(-N+i))*posn0[i]

    # posn0 = (1/np.sqrt(P))*posn0

    # ==============================================

    S = np.kron(np.roll(np.eye(P), 1, axis=0),C00) + np.kron(np.roll(np.eye(P), -1, axis=0),C11) # Shift operator
    Toss = np.cos(θ)*C00 + np.sin(θ)*C01 - np.sin(θ)*C10 + np.cos(θ)*C11   # Coin operator
    C = np.kron(np.eye(P),Toss)
    U = S@C # QW unitary evolution


    # Initial state -------------

    χ = (1/np.sqrt(2))*(coin0 + 1j*coin1)
    Ψ = np.kron(ψx,χ)

    Ψ_CP = [] # List of Canonically purified state

    # Evolution ------------------

    n1 = P; n2 = 2 # Dimension of position and coin space

    for i in range(N+1):
        ρ = Ψ@conjT(Ψ)  # Density Matrix
        ρc = partial_trace(ρ,n1,n2,2) # partial trace over position space
        w,v = LA.eig(ρc) 
        Ψ_CP.append(np.sqrt(w[0])*np.kron(v[:,[0]],v[:,[0]]) + np.sqrt(w[1])*np.kron(v[:,[1]],v[:,[1]])) # purified state 
        Ψ = U@Ψ

    return Ψ_CP

# =============================================================================================

fun = lambda x,s: np.exp(1j*x*s/2)*np.sin(x*s/2)/(x/2)

# ==========================================


    
def COMPLEXITY(U, # Target Unitary matrix
               locality, # locality
               μ = 100 # cost function
              ):
    
    α = 2*μ/(1+μ)
    H = logm(U) # Hamiltonian 
    sol = np.array([1j/4*(H[0, 2] + H[1, 3] + H[2, 0] + H[3, 1]), 1/4*(-H[0, 2] - H[1, 3] + H[2, 0] + H[3, 1]), 
    1j/4*(H[0, 1] + H[1, 0] - H[2, 3] - H[3, 2]), 1/4*(-H[0, 1] + H[1, 0] + H[2, 3] - H[3, 2]), -(1j/2)*(H[0, 0] + H[1, 1]), 
    1/4*(-H[0, 3] - H[1, 2] + H[2, 1] + H[3, 0]), -(1j/4)*(H[0, 3] + H[1, 2] + H[2, 1] + H[3, 0]),-1j/4*(H[0, 3] - H[1, 2] - H[2, 1] + H[3, 0]), 
    1/4*(H[0, 3] - H[1, 2] + H[2, 1] - H[3, 0]), -(1j/2)*(H[0, 0] + H[2, 2]), 1j/4*(H[0, 1] + H[1, 0] + H[2, 3] + H[3, 2]), 
    1/4*(-H[0, 1] + H[1, 0] - H[2, 3] + H[3, 2]), 1j/4*(H[0, 2] - H[1, 3] + H[2, 0] - H[3, 1]), 
    1/4*(-H[0, 2] + H[1, 3] + H[2, 0] - H[3, 1]), -(1j/2)*(H[1, 1] + H[2, 2])])
    
    if locality == 1: # 1-local
        f = np.sum(sol[:4]**2)     
        vi = np.array([sol[4],sol[5],sol[6],sol[7],sol[8],sol[9],sol[10],sol[11],sol[12],sol[13],sol[14]])
        
    elif locality == 2: # 2-local
        f = np.sum(sol[:10]**2) 
        vi = np.array([sol[10],sol[11],sol[12],sol[13],sol[14]])
        
    else: # 3-local
        f = np.sum(sol[:14]**2)    
        vi = np.array([sol[14]])
        
    return np.sqrt(f+(1+μ)*np.sum(vi**2))
    
    
    

# =============================================================================================
# ===================== Target Unitary Calculation =======================================
# ============================================================================================


def TARGET_UNITARY(Ψ, # Target state (column vector)
                  ):
    U = np.zeros((4,4),dtype = 'complex')
    U[:,[0]] = Ψ

    u = [Ψ]

    for i in range(3):
        v = np.random.uniform(0,1,size= (4,1)) + 1j*np.random.uniform(0,1,size= (4,1)) # Generate a random-vector   
            # GSP ------------
        for j in range(len(u)):
            v = v - (vec_prod(u[j],v)*u[j])
        v = v/vec_norm(v)
        u.append(v)
        U[:,[i+1]] = v

    det = LA.det(U)
    scaling_factor = det**(1/4)
    U  = U/scaling_factor
    
    return U

# ======================================================================================


def SAMPLE_GEN(Ψ1, # state 1
               Ψ2, # state 2
               sample_size = 500, # sample sime
              ):
    U12 = []
    for k in range(sample_size):

        # Target unitary from ref to step ------------------------------

        U02 = TARGET_UNITARY(Ψ2)

        # Target unitary from ref to (step - 1) ------------------------

        U01 = TARGET_UNITARY(Ψ1)

        # ==============================================================
        # Target unitary from (step - 1) to step

        U12.append(U02@conjT(U01))

        # ==========================================================

        # Code Check ---

#         Ψ  = U12[k]@Ψ_CP[step-1]

#         phase = np.exp(-1j*np.angle(Ψ[0]))
#         Ψ = Ψ*phase
#         if np.allclose(Ψ_CP[step],Ψ) and k == sample -1:
#         print('Looks okay!')
    return U12

# =========================================================

def MIN_COMPLEXITY(U12, # Sample of Unitary operators
                   locality = 2 # locality
                  ):
    niel_com = np.zeros(len(U12))
    
    for i in range(len(niel_com)):
        niel_com[i] = COMPLEXITY(U12[i],locality)
    return np.min(niel_com),np.argmin(niel_com)
    


# ======================================================
