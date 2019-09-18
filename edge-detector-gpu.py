from numba import cuda
#from numba import jit
import math
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
#import time

threadsperblock = 8
blockspergrid = 2
n = threadsperblock*blockspergrid
num_bits = 16




#
#
@cuda.jit
def test(out,out_temp,r,c,mem_state,rng_states,num_bits,prob_dist,input_array,target_tt,delta_array):
    i = cuda.grid(1)
    exp = 2.718281828459045
    max_range = prob_dist.shape[0]
    #produce random design
    for j in range(out.shape[1]):
        for k in range(out.shape[2]):
            indx = int(xoroshiro128p_uniform_float32(rng_states, i)*max_range)
            x = int(xoroshiro128p_uniform_float32(rng_states, i)*out.shape[1])
            y = int(xoroshiro128p_uniform_float32(rng_states, i)*out.shape[2])            
            out[i,x,y] = prob_dist[indx]
    # FUNC1 : D-> mem_states -> truth table --> delta
    # delta(D) -> returns delta
    def cost(D):
        #find mem_states of D for input X
        delta=0
        for X in range(input_array.shape[0]):
            for j in range(r.shape[1]):
                r[i,j] = 0
                c[i,j] = 0
            r[i,0] = 1
#            for j in range(D.shape[0]):
#                for k in range(D.shape[1]):
#                    mem_state[i,j,k] = 0
            for j in range(D.shape[0]):
                for k in range(D.shape[1]):
                    if D[j,k] <= (num_bits+1):
                        if D[j,k] <= 1:
                            mem_state[i,j,k] = D[j,k]
                        else:
                            mem_state[i,j,k] = input_array[X,(D[j,k]-2)]
                    elif D[j,k] > num_bits+1:
                        mem_state[i,j,k] = not(input_array[X,D[j,k]-2-num_bits])
            # find flow for given mem state and store in tt
            for t in range(D.shape[0]*D.shape[1]):
                for j in range(1,D.shape[0]):
                    for k in range(D.shape[1]):
                        r[i,j] = r[i,j] or (mem_state[i,j,k] and c[i,k])
                for j in range(D.shape[1]):
                    for k in range(D.shape[0]):
                        c[i,j] = c[i,j] or (mem_state[i,k,j] and r[i,k])
                if c[i,D.shape[1]-1] == 1:
                    break
            if target_tt[X] == 2:
                delta += 0
            elif target_tt[X] == c[i,D.shape[1]-1]:
                delta += 1
        return delta
                
    # FUNC2 : D --> perturbed D
    #perturb(D) -> returns D_new
    def perturb(D):
        indx = int(xoroshiro128p_uniform_float32(rng_states, i)*max_range)
        x = int(xoroshiro128p_uniform_float32(rng_states, i)*out.shape[1])
        y = int(xoroshiro128p_uniform_float32(rng_states, i)*out.shape[2])
        D[x,y] = prob_dist[indx]
        return D
    
    def copy(A,B):
        for j in range(A.shape[0]):
            for k in range(A.shape[1]):
                B[j,k] = A[j,k]
        return B
    
    # Simluated Annealing
    def simulated_annealing(D,temp,cool_rate):
#        make copies of delta and D
        copy(D,out_temp[i])
        delta_old = cost(D)
        delta_array[i] = delta_old
        for j in range(2000):
            rand_num = xoroshiro128p_uniform_float32(rng_states, i)
            perturb(out_temp[i])
            delta = cost(out_temp[i])
            if delta < delta_old:
                copy(out_temp[i], D)
                delta_array[i] = delta
            if rand_num < exp**(-(delta - delta_old)/temp):
                copy(out_temp[i],D)
                delta_array[i] = delta
                delta_old = delta
                temp = temp*cool_rate
            else:
                delta_old = delta
                temp = temp*cool_rate
          
    simulated_annealing(out[i],5000,0.999)
    
#    copy(out[i],out_temp[i])
#    for j in range(10):
#        perturb(out[i])    
#    delta_array[i] = cost(out[i])

# input arrays :  OUT-array, prob_dist-array, input_bits-array, RNG_states-array, Target-array   
out = np.zeros((n,8,8),dtype=np.int32)
out_temp = np.zeros((n,8,8),dtype=np.int32)
mem_state = np.zeros((n,8,8),dtype='bool')
r = np.zeros((n,8),dtype='bool')
r[0] = 1
c = np.zeros((n,8),dtype='bool')
#computed_tt = np.zeros((n,1),dtype='bool')

def create_prob_pool(num_bits):
    prob_dist = np.zeros((), dtype=np.int32)
    for i in range(int(num_bits/4)):
        prob_dist = np.append(prob_dist,0)
    for i in range(int(num_bits/4)):
        prob_dist = np.append(prob_dist,1)
    for i in range(2,2*num_bits+2):
        prob_dist = np.append(prob_dist,i)
    np.random.shuffle(prob_dist)
    return prob_dist

def trans(x):
    if x == 0: return [0]
    bit = []
    while x:
        bit.append(x % 2)
        x >>= 1
    return bit

def generate_input_matrix(b):
    m = np.zeros((2**b,b),dtype='bool')
    for i in range(2**b):
        bit_array=trans(i)
        for j in range(len(bit_array)):
            m[i,j] = bit_array[j]
    return m

prob_dist = create_prob_pool(16)
input_array = generate_input_matrix(16)
data = np.load('tt_reduced_plus_5.npy') 
target = data[:,:,2].reshape((65536,)) 
target_tt = np.ascontiguousarray(target)
delta = np.zeros((n),dtype=np.int32)    

# generate random states
rng_states = create_xoroshiro128p_states(n, seed=42)

# copy to device memory
prob_dist_d = cuda.to_device(prob_dist)
rng_states_d = cuda.to_device(rng_states) 
mem_state_d = cuda.to_device(mem_state)   
out_d = cuda.to_device(out)
out_temp_d = cuda.to_device(out_temp)
input_array_d = cuda.to_device(input_array)
target_d = cuda.to_device(target_tt)
r_d = cuda.to_device(r)
c_d = cuda.to_device(c)
delta_d = cuda.to_device(delta)
#computed_tt_d = cuda.to_device(computed_tt)

# call kernel
test[blockspergrid,threadsperblock](out_d,out_temp_d,r_d,c_d,mem_state_d,rng_states_d,num_bits,prob_dist_d,input_array_d,target_d,delta_d)

# copy to host
#rng_states_ = delta_d.copy_to_host()
out_h = out_d.copy_to_host()
out_temp_h = out_temp_d.copy_to_host()
delta_h = delta_d.copy_to_host()
#computed_tt_h = computed_tt_d.copy_to_host()
# print answers
for i in range(n):
    print(delta_h[i])
    print(out_h[i,:,:])



