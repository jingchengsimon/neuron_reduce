#reduction of L5_PC using Neuron_Reduce
from __future__ import division
import os
import logging
from neuron import gui, h
from neuron.units import ms, mV
import numpy as np
import time
import matplotlib.pyplot as plt

from threading import Thread

from math import floor
a = -0.7
b = floor(a)
print(a-b)
print(b)
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
sections_number = 5
selected_sections = sample(all_sections_basal,sections_number) + sample(all_sections_apical,sections_number)
index_section_list = []
for selected_section in selected_sections:
    index_section = all_sections.index(selected_section)
    index_section_list.append(index_section)
print(index_section_list)

# selected_segments = [i for i in all_sections[0]]
selected_segments = [i for j in selected_sections for i in j]
len_per_segment = np.array([seg.sec.L/seg.sec.nseg for seg in selected_segments])

rnd = np.random.RandomState(10)

if rnd.uniform() < 0.85:
    e_syn, tau1, tau2, spike_interval, syn_weight = 0, 0.3, 1.8, 1000/2.5, 0.0016
else:
    e_syn, tau1, tau2, spike_interval, syn_weight = -86, 1, 8, 1000/15.0, 0.0008

scale = 1000
print("Loop start".center(scale // (2*(scale//100)),"-"))
start = time.perf_counter()

index_list = []
for i in range(scale):
    seg_for_synapse = rnd.choice(selected_segments, p=len_per_segment/sum(len_per_segment))
    index_seg_for_synapse = all_segments.index(seg_for_synapse)
    index_list.append(index_seg_for_synapse)
    
import seaborn as sns
 
# plt.hist(index_list)
sns.distplot(index_list,hist=False)
plt.xlim(0,639)
plt.show()