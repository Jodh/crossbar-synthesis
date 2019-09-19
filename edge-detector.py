#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:02:49 2019
@title: edge-detector.py
@author: jodh
@specs: 8 by 8 crossbar
        16 bits (2 pixels of 8 bits each)
"""
import numpy as np
import random
#import time
#import os
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
           
data = np.load('tt_reduced_plus_5.npy') 
TT_target = data[:,:,2].reshape((65536,))

    
#select randomly a design with mems being off with a 28% probability and least significant bit dropped
def generate_random_design(n,l,num_literals,prob_dist):
    m = np.random.choice(np.arange(2*num_literals+2),size=(n,l),p=prob_dist)
    return m   
probability_distribution = np.array([0.2825,0.14,0,0.020625,0.020625,0.020625,0.020625,0.020625,0.020625,0.020625,0,0.020625,0.020625,0.020625,0.020625,0.020625,0.020625,0.020625,0,0.020625,0.020625,0.020625,0.020625,0.020625,0.020625,0.020625,0,0.020625,0.020625,0.020625,0.020625,0.020625,0.020625,0.020625])     
            
#print(D)
# find mem states given design and input
def find_mem_states(D, d_input):
#    t0 = time.time()
    num_literals = 2*len(d_input)+2
    mapping = np.zeros(num_literals, dtype='bool')
    m = np.zeros(np.shape(D))
    for i in range(num_literals):
        if i <=1:
            mapping[i] = i
        elif i <= (len(d_input)+1):
            mapping[i]=d_input[i-2]
            mapping[i+len(d_input)] = not(d_input[i-2])
    for i in range(len(D[:,1])):
        for j in range(len(D[1,:])):
            m[i,j] = mapping[D[i,j]]
#    t1 = time.time()
#    print('finding mem states',t1-t0)
    return m

#m = find_mem_states(D, [1,0,1])
#perturb design radomly
def perturb_design(D, prob_dist):
    n = len(D[:,1])
    l = len(D[1,:])
    i = np.random.randint(0,high=n)
    j = np.random.randint(0,high=l)
#    print(i," ",j,"\n")
#    print(D[i,j],"\n")
    new_literal = np.random.choice(np.arange(len(prob_dist)),p=prob_dist)
    while 1:
        if D[i,j] == new_literal:
            new_literal = np.random.choice(np.arange(len(prob_dist)),p=prob_dist)
        else:
            D[i,j] = new_literal
#            print(new_literal)
            break
    return D   

#D_new = perturb_design(D, probability_distribution)
    
# find flow value at output wire
def flow_at_outwire(mem_states):
#    t0 = time.time()
    n = len(mem_states[:,1])
    l = len(mem_states[1,:])
    r = np.zeros(n,dtype='bool')
    r[0] = 1
    c = np.zeros(l,dtype='bool')
    # iteration through time
    t = 0
    while t < 16:
        for i in range(1,n):
            for j in range(l):
                r[i] = r[i] or (mem_states[i,j] and c[j])
        for j in range(l):
            for i in range(n):
                c[j] = c[j] or (mem_states[i,j] and r[i])
        t = t+1
        if c[l-1] == 1:
            break
#    t1 = time.time()
#    print('flow finding',t1-t0)
    return c[l-1]

# find truth table of given design 
def generate_truth_table(D, num_bits):  
    input_matrix = generate_input_matrix(num_bits)
    tt = np.zeros(len(input_matrix[:,1]), dtype='bool')
    for i in range(len(input_matrix[:,1])):
        m = find_mem_states(D, input_matrix[i,:])
        tt[i] = not(flow_at_outwire(m))
    return tt

# input: design, target truth table, num_bits, num_samples, input matrix; output: sampled truth table of design and sampled target truth table
def sampled_truth_table(D, num_bits, TT_target, num_samples, input_matrix):
    tt = np.zeros(0,dtype='bool')
    tt_target = np.zeros(0,dtype='bool')
#    t0 = time.time()
    sample_array = random.sample(range(2**num_bits), num_samples)
#    t1 = time.time()
#    print(t1-t0)
#    t0 = time.time()
    for i in sample_array:
#        print(i)
        m = find_mem_states(D, input_matrix[i,:])
        tt = np.append(tt,flow_at_outwire(m))
        tt_target = np.append(tt_target,TT_target[i])
#    t1 = time.time()
#    print(' loop time ',t1-t0)
    return np.logical_xor(tt,tt_target)

# input: design, target truth table, num_bits, num_samples, input matrix; output: delta weighted cost function
def reduced_truth_table(D, num_bits, TT_target, input_matrix):
    tt = np.zeros(0,dtype='bool')
    tt_target = np.zeros(0,dtype='bool')
    delta = 0
    for i in range(2**num_bits):
        if TT_target[i] == 2:
            continue
        else:
            m = find_mem_states(D, input_matrix[i,:])
            tt = np.append(tt,not(flow_at_outwire(m)))
            tt_target = np.append(tt_target,TT_target[i])
    for i in range(len(tt)):
        if tt_target[i] != tt[i]:
            if tt_target[i] == 1:
                delta += 3
            else:
                delta += 1
#    t1 = time.time()
#    print(' loop time ',t1-t0)


            
    return delta

    
# the anneling function 
def simulated_annealing(D,num_bits,TT_target,temp,cool_rate,is_sampling, num_samples):
    input_matrix = generate_input_matrix(num_bits)
    if is_sampling == 0:
#        delta = np.logical_xor(tt,TT_target)
        delta = reduced_truth_table(D,num_bits,TT_target,input_matrix)
    else:
        delta = sampled_truth_table(D, num_bits, TT_target, num_samples, input_matrix)
#    delta_old = np.copy(delta)
        delta_old = delta
    D_best = np.copy(D)
    i = 0
    while delta >= 5000:
        if i >= 1800:
            break
        i = i +1
        D_temp = perturb_design(D,probability_distribution)
        if is_sampling == 0:
#            delta = np.logical_xor(tt,TT_target)
            delta = reduced_truth_table(D_temp, num_bits, TT_target, input_matrix)
        else:     
            delta = sampled_truth_table(D_temp, num_bits, TT_target, input_matrix)
        if delta < delta_old:
            D_best = np.copy(D_temp)
        if np.random.rand(1) < np.exp(-(delta-delta_old)/temp):
            D = np.copy(D_temp)
            delta_old = delta
            temp = cool_rate*temp
        else:
            delta_old = delta
            temp = cool_rate*temp
#        if i%1000 == 0:
#            print("\n")
#            print(delta)
#            print(D_best)
#            print("\n")
#            
    print("\n")    
    print(delta)
    print(D_best)
    print("\n")
#    return D_best, D

## JETCAS design PSNR 2.4dB
a = np.zeros(8)
Na = np.zeros(8)
b = np.zeros(8)
Nb = np.zeros(8)

for i in range(8):
    a[i] = int(2+i)
    Na[i] = int(18+i)
    b[i] = int(10+i)
    Nb[i] = int(26+i)
    
D_jetcas = np.array([[Na[7], False, Nb[6], Nb[7], Na[5], False, Nb[7],False],
[Na[6],a[6],False,b[5],Na[5],a[6],b[5],b[5]],
[a[6],False,b[5],Na[1],a[6],b[5],b[7],b[3]],
[Na[5],a[5],a[6],Na[5],b[5],Na[5],b[5],Nb[7]],
[a[0],a[6],Na[7],b[5],a[6],a[6],b[6],a[6]],
[a[6],a[5],Nb[4],Nb[7],Na[5],b[5],False,a[5]],
[b[5],False,False,b[5],b[5],b[5],b[7],b[0]],
[b[7],Nb[6],False,b[7],False,Na[6],a[7],b[7]]],dtype=int) 
    
for i in range(8):
    a[i] = int(18+i)
    Na[i] = int(2+i)
    b[i] = int(26+i)
    Nb[i] = int(20+i)
D_jetcas_inverse = np.array([[Na[7], False, Nb[6], Nb[7], Na[5], False, Nb[7],False],
[Na[6],a[6],False,b[5],Na[5],a[6],b[5],b[5]],
[a[6],False,b[5],Na[1],a[6],b[5],b[7],b[3]],
[Na[5],a[5],a[6],Na[5],b[5],Na[5],b[5],Nb[7]],
[a[0],a[6],Na[7],b[5],a[6],a[6],b[6],a[6]],
[a[6],a[5],Nb[4],Nb[7],Na[5],b[5],False,a[5]],
[b[5],False,False,b[5],b[5],b[5],b[7],b[0]],
[b[7],Nb[6],False,b[7],False,Na[6],a[7],b[7]]],dtype=int)  
    
D = generate_random_design(8,8,16,probability_distribution)  
#D = generate_random_design(6,6,16,probability_distribution) 
#print(D) 
            
simulated_annealing(D=D, num_bits=16, TT_target = TT_target, temp = 5000, cool_rate=0.99, is_sampling=False, num_samples=10000)    


#print(D_best)
#
#
#print(D_jetcas)
#tt_best = generate_truth_table(D_best, 16)
#print(np.sum(np.logical_xor(tt_best, TT_target)))
##
#path_save = '/home/jodh/Documents/python-scripts/'
#def save_table(path, filename,truth_table):
#    filepath = os.path.join(path,filename)
#    np.save(filepath, truth_table)
#tt_jetcas_inverse = tt_jetcas_inverse.reshape(256,256)
#timestamp = str(time.time())
#filename = 'tt_best_'+timestamp   
#save_table(path_save,'tt_jetcas_inverse',tt_jetcas_inverse)

#import seaborn as sns
#
#sns.heatmap(tt_jetcas_inverse)

