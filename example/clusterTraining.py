#reduction of L5_PC using Neuron_Reduce

from __future__ import division
import os
import logging
from neuron import gui,h
from neuron.units import ms, mV
import numpy as np
import time
import matplotlib.pyplot as plt

from threading import Thread

# logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

# Create a L5_PC model (channels in biophys3.hoc and morphology in template.hoc)
h.load_file("import3d.hoc")
h.nrn_load_dll('C:/Users/Windows/Desktop/Drew/neuron_reduce/example/mod/nrnmech.dll')
h.load_file('C:/Users/Windows/Desktop/Drew/neuron_reduce/example/modelFile/L5PCbiophys3.hoc')
h.load_file('C:/Users/Windows/Desktop/Drew/neuron_reduce/example/modelFile/L5PCtemplate.hoc')
complex_cell = h.L5PCtemplate('C:/Users/Windows/Desktop/Drew/neuron_reduce/example/modelFile/cell1.asc')
h.celsius = 37
h.v_init = complex_cell.soma[0].e_pas

# Add synapses to the model
synapses_list, netstims_list, netcons_list, randoms_list = [], [], [] ,[]
seg_for_synapse_list = []

# Add segments of apical and basal sections for synapses (no soma and axon)
all_segments = [i for j in map(list,list(complex_cell.basal)) for i in j] + [i for j in map(list,list(complex_cell.apical)) for i in j] 
len_per_segment = np.array([seg.sec.L/seg.sec.nseg for seg in all_segments])

all_sections_basal = [i for i in map(list,list(complex_cell.basal))] 
all_sections_apical = [i for i in map(list,list(complex_cell.apical))]
all_sections = all_sections_basal + all_sections_apical

from random import sample
from math import floor

rnd = np.random.RandomState(10)
if rnd.uniform() < 0.85:
    e_syn, tau1, tau2, spike_interval, syn_weight = 0, 0.3, 1.8, 1000/2.5, 0.0016
else:
    e_syn, tau1, tau2, spike_interval, syn_weight = -86, 1, 8, 1000/15.0, 0.0008

section_cluster_list = []
Section_cluster_list = []
type_list = []

def synapse_initialization(i):

    Section = rnd.choice(all_sections)
    Section_cluster_list.append(Section)
    
    section = Section[0].sec
    section_cluster_list.append(section)

    loc = section(rnd.uniform()).x
    
    seg_for_synapse = section(loc)
    seg_for_synapse_list.append(seg_for_synapse)
    synapses_list.append(h.Exp2Syn(seg_for_synapse))
    
    type_list.append(rnd.choice(['A','B']))

    synapses_list[i].e, synapses_list[i].tau1, synapses_list[i].tau2 = e_syn, tau1, tau2

    netstims_list.append(h.NetStim())
    netstims_list[i].interval, netstims_list[i].number, netstims_list[i].start, netstims_list[i].noise = spike_interval, 10, 100, 1

    randoms_list.append(h.Random())
    randoms_list[i].Random123(i)
    randoms_list[i].negexp(1)
    netstims_list[i].noiseFromRandom(randoms_list[i])

    netcons_list.append(h.NetCon(netstims_list[i], synapses_list[i])) # need to rewrite with an assign function
    netcons_list[i].delay, netcons_list[i].weight[0] = 0, syn_weight

    # a = "*" * (i // (numSyn//100))
    # b = "." * ((numSyn - i) // (numSyn//100))
    # c = (i / numSyn) * 100
    # dur = time.perf_counter() - start
    # print("\r{:^3.0f}% [{}->{}] {:.2f}s".format(c,a,b,dur),end = "")
    # time.sleep(0.1)

# print("Loop start".center(numSyn // (2*(numSyn//100)),"-"))
# start = time.perf_counter()

## Initialize synapses
numSyn = 1000 #distributed connectivity needs large enough inputs to generate spikes

for i in range(numSyn):
    synapse_initialization(i)

# print("\n" + "Loop end".center(numSyn // (2*(numSyn//100)),'-'))

## Shuffle and train
distance_list = []
distance_limit = 15 #microns
distance_matrix = np.zeros((len(seg_for_synapse_list),len(seg_for_synapse_list)))
for i in range(len(seg_for_synapse_list)):
    for j in range(len(seg_for_synapse_list)):
        if i < j:
            # Use a matrix instead of a list, since the distance between two synapses is fixed 
            # no matter what type they are belonged to
            distance_matrix[i,j] = h.distance(seg_for_synapse_list[i],seg_for_synapse_list[j])
            
            distance_list.append(distance_matrix[i,j]) if distance_matrix[i,j] < distance_limit else None
# cax = plt.imshow(distance_matrix, cmap='viridis')
# plt.colorbar(cax)
# plt.show()

distance_a_a_list = []
distance_b_b_list = []
for i in range(len(type_list)):
    for j in range(len(type_list)):
        if i < j:
            if type_list[i] == type_list[j] == 'A' and distance_matrix[i,j] < distance_limit:
                distance_a_a_list.append(distance_matrix[i,j])
            if type_list[i] == type_list[j] == 'B' and distance_matrix[i,j] < distance_limit:
                distance_b_b_list.append(distance_matrix[i,j])
## Visulization part

import seaborn as sns
plt.figure()
sns.distplot(distance_list,hist=True,label='All')
sns.distplot(distance_a_a_list,hist=True,label='A-A')
sns.distplot(distance_b_b_list,hist=True,label='B-B')
plt.legend()
plt.xlabel('Distance (microns)')
plt.ylabel('Probability')

#Simulate the full neuron for 1 seconds
soma_v = h.Vector().record(complex_cell.soma[0](0.5)._ref_v)
dend_v = h.Vector().record(complex_cell.dend[0](0.5)._ref_v)
apic_v = h.Vector().record(complex_cell.apic[0](0.5)._ref_v)
time_v = h.Vector().record(h._ref_t)

h.tstop = 1000
st = time.time()
h.run()
print('complex cell simulation time {:.4f}'.format(time.time()-st))

#plotting the results

plt.figure()
plt.plot(time_v, soma_v, label='soma')
plt.plot(time_v, dend_v, label='basal')
plt.plot(time_v, apic_v, label='apical')

plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.show()

