#reduction of L5_PC using Neuron_Reduce

from __future__ import division
import os
import logging
from neuron import gui,h
from neuron.units import ms, mV
import numpy as np
import neuron_reduce
import time
import matplotlib.pyplot as plt

from threading import Thread

# logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

#Create a L5_PC model (channels in biophys3.hoc and morphology in template.hoc)
h.nrn_load_dll('C:/Users/Windows/Desktop/Drew/neuron_reduce/example/mod/nrnmech.dll')
h.load_file('L5PCbiophys3.hoc')
h.load_file("import3d.hoc")
h.load_file('L5PCtemplate.hoc')
complex_cell = h.L5PCtemplate('cell1.asc')
h.celsius = 37
h.v_init = complex_cell.soma[0].e_pas


#Add synapses to the model
synapses_list, netstims_list, netcons_list, randoms_list = [], [], [] ,[]

#Add segments of apical and basal sections for synapses (no soma and axon)
all_segments = [i for j in map(list,list(complex_cell.basal)) for i in j] + [i for j in map(list,list(complex_cell.apical)) for i in j] 
len_per_segment = np.array([seg.sec.L/seg.sec.nseg for seg in all_segments])

all_sections_basal = [i for i in map(list,list(complex_cell.basal))] 
all_sections_apical = [i for i in map(list,list(complex_cell.apical))]
all_sections = all_sections_basal + all_sections_apical
from random import sample
from math import floor
# sections_number = 10
# # selected_sections = sample(all_sections_basal,sections_number) + sample(all_sections_apical,sections_number)
# selected_sections = sample(all_sections_basal,sections_number)

# # selected_segments = [i for i in all_sections[0]]
# selected_segments = [i for i in all_sections_basal[5]]
# # selected_segments = [i for j in selected_sections for i in j]
# # selected_segments = all_segments
# len_per_segment = np.array([seg.sec.L/seg.sec.nseg for seg in selected_segments])

rnd = np.random.RandomState(10)
if rnd.uniform() < 0.85:
    e_syn, tau1, tau2, spike_interval, syn_weight = 0, 0.3, 1.8, 1000/2.5, 0.0016
else:
    e_syn, tau1, tau2, spike_interval, syn_weight = -86, 1, 8, 1000/15.0, 0.0008

def Connection(i):

    #Case shown here is the same type of input to all synapses, which are uniformly distributed based on lengths of segments,
    #For more biophysically realistic cases, change the rule of distribution for synapses, 
    #For different types of input, the rule of distribution for synapses can be the same or different

    #Randomly choose a segment and add a synapse, the input to each synapse is the same
    if cluster_flag:
        section_cluster = rnd.choice(section_cluster_list)
        Section_cluster = Section_cluster_list[section_cluster_list.index(section_cluster)]
        loc_lower_bound = loc_lower_bound_list[section_cluster_list.index(section_cluster)]
        loc_upper_bound = loc_upper_bound_list[section_cluster_list.index(section_cluster)]

        loc = rnd.uniform(loc_lower_bound,loc_upper_bound) 
        
        if loc > 1 or loc < 0:
            gap = floor(loc)
            loc = loc - gap
            Section_cluster = all_sections[all_sections.index(Section_cluster) + gap]
            section_cluster = Section_cluster[0].sec
            
        synapses_list.append(h.Exp2Syn(section_cluster(loc)))

    else:
        seg_for_synapse = rnd.choice(all_segments, p=len_per_segment/sum(len_per_segment))
        synapses_list.append(h.Exp2Syn(seg_for_synapse))

    synapses_list[i+k].e, synapses_list[i+k].tau1, synapses_list[i+k].tau2 = e_syn, tau1, tau2

    netstims_list.append(h.NetStim())
    netstims_list[i+k].interval, netstims_list[i+k].number, netstims_list[i+k].start, netstims_list[i+k].noise = spike_interval, 6, 50, 1

    randoms_list.append(h.Random())
    randoms_list[i+k].Random123(i+k)
    randoms_list[i+k].negexp(1)
    netstims_list[i+k].noiseFromRandom(randoms_list[i+k])

    netcons_list.append(h.NetCon(netstims_list[i+k], synapses_list[i+k])) # need to rewrite with an assign function
    netcons_list[i+k].delay, netcons_list[i+k].weight[0] = 0, syn_weight

    a = "*" * (i // (scale//100))
    b = "." * ((scale - i) // (scale//100))
    c = (i / scale) * 100
    dur = time.perf_counter() - start
    print("\r{:^3.0f}% [{}->{}] {:.2f}s".format(c,a,b,dur),end = "")
    time.sleep(0.1)

# Use thread to accelarate the loop
# ths = []
# for i in range(scale):
#     th = Thread(target = Connection, args = (i,))
#     th.start()
#     ths.append(th)
# for th in ths:
#     th.join()


scale = 1000 #distributed connectivity needs large enough inputs to generate spikes
print("Loop start".center(scale // (2*(scale//100)),"-"))
start = time.perf_counter()

cluster_flag = True
#Create first k synapses for cluster: randomly choose k sections and randomly choose a location 
# between 0-1 and create synapses by Exp2Syn, then we can make 5 ranges within 20 microns from 
# these synapses, these should be a list for rnd.choice in func Connection to create further synapses
if cluster_flag:
    k = 100
    scale = scale - k
    cluster_size = 20 #microns

    section_cluster_list = []
    Section_cluster_list = []

    loc_lower_bound_list = []
    loc_upper_bound_list = []

    for i in range(k):
        
        Section = rnd.choice(all_sections)
        Section_cluster_list.append(Section)
        
        section = Section[0].sec
        section_cluster_list.append(section)

        loc = section(rnd.uniform()).x
        loc_lower_bound = loc - cluster_size / section.L
        loc_upper_bound = loc + cluster_size / section.L
        loc_lower_bound_list.append(loc_lower_bound)
        loc_upper_bound_list.append(loc_upper_bound)

        synapses_list.append(h.Exp2Syn(section(loc)))
        synapses_list[i].e, synapses_list[i].tau1, synapses_list[i].tau2 = e_syn, tau1, tau2

        netstims_list.append(h.NetStim())
        netstims_list[i].interval, netstims_list[i].number, netstims_list[i].start, netstims_list[i].noise = spike_interval, 10, 100, 1

        randoms_list.append(h.Random())
        randoms_list[i].Random123(i)
        randoms_list[i].negexp(1)
        netstims_list[i].noiseFromRandom(randoms_list[i])

        netcons_list.append(h.NetCon(netstims_list[i], synapses_list[i])) # need to rewrite with an assign function
        netcons_list[i].delay, netcons_list[i].weight[0] = 0, syn_weight
else:
    k = 0

if cluster_flag:
    title = 'Cluster number: '+str(k)+' cluster size: '+str(cluster_size)
else:
    title = 'Distributed'

for i in range(scale):
    Connection(i)

print("\n" + "Loop end".center(scale // (2*(scale//100)),'-'))

#Simulate the full neuron for 1 seconds
soma_v = h.Vector().record(complex_cell.soma[0](0.5)._ref_v)
dend_v = h.Vector().record(complex_cell.dend[0](0.5)._ref_v)
# dend_v_5 = h.Vector().record(complex_cell.dend[5](0.5)._ref_v)
# dend_v_6 = h.Vector().record(complex_cell.dend[6](0.5)._ref_v)
apic_v = h.Vector().record(complex_cell.apic[0](0.5)._ref_v)
# apic_v_5 = h.Vector().record(complex_cell.apic[5](0.5)._ref_v)
# apic_v_6 = h.Vector().record(complex_cell.apic[6](0.5)._ref_v)
time_v = h.Vector().record(h._ref_t)

h.tstop = 1000
st = time.time()
h.run()
print('complex cell simulation time {:.4f}'.format(time.time()-st))

#plotting the results
plt.figure()
plt.plot(time_v, soma_v, label='soma')
plt.plot(time_v, dend_v, label='basal')
# plt.plot(time_v, dend_v_5, label='basal_5')
# plt.plot(time_v, dend_v_6, label='basal_6')
plt.plot(time_v, apic_v, label='apical')
# plt.plot(time_v, apic_v_5, label='apical_5')
# plt.plot(time_v, apic_v_6, label='apical_6')
plt.legend()
# plt.xlim(140,160)
plt.xlabel('time (ms)')
plt.ylabel('voltage (mV)')
plt.title(title)
plt.show()

