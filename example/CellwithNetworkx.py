from __future__ import division
import os
import logging
from neuron import gui, h
from neuron.units import ms, mV
import numpy as np
import time
import pandas as pd
import networkx as nx
from tqdm import tqdm
import warnings
import re
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class CellwithNetworkx:
    def __init__(self, swc_file, numSyn=1000):
        h.load_file("import3d.hoc")
        h.nrn_load_dll('C:/Users/Windows/Desktop/Drew/neuron_reduce/example/mod/nrnmech.dll')
        h.load_file('C:/Users/Windows/Desktop/Drew/neuron_reduce/example/modelFile/L5PCbiophys3.hoc')
        h.load_file('C:/Users/Windows/Desktop/Drew/neuron_reduce/example/modelFile/L5PCtemplate.hoc')
        self.complex_cell = h.L5PCtemplate(swc_file)
        h.celsius = 37
        h.v_init = self.complex_cell.soma[0].e_pas
        self.G = None
        self.sp = None
        self.distance_matrix = None
        self.numSyn = numSyn

        self.parentID_list = None
        self.sectionID_list = None
        self.sectionName_list = None
        self.length_list = None

        self.loc_list = None
        self.type_list = None
        self.sectionID_synapse_list = None
        self.section_synapse_list = None

        self._create_graph()
        self._add_synapses()

    def _create_graph(self):
        all_sections = [i for i in map(list, list(self.complex_cell.soma))] + [i for i in map(list, list(self.complex_cell.basal))] + [i for i in map(list, list(self.complex_cell.apical))]

        # Create DataFrame to store section information
        df = pd.DataFrame(columns=['sectionID', 'parentID', 'sectionName', 'parentName', 'length'])

        parent_list, parentID_list, parent_index_list, sectionID_list, sectionName_list, length_list = [], [], [], [], [], []

        for i, section in enumerate(all_sections):
            Section = section[0].sec
            sectionID = i
            sectionName = Section.psection()['name']
            match = re.search(r'\.(.*?)\[', sectionName)
            sectionType = match.group(1)
            L = Section.psection()['morphology']['L']

            parent_list.append(sectionName)
            parent_index_list.append(sectionID)

            if i == 0:
                parentID = 0
                parentName = 'None'
            else:
                parent = Section.psection()['morphology']['parent'].sec
                parentName = parent.psection()['name']
                parentID = parent_index_list[parent_list.index(parentName)]

            sectionID_list.append(sectionID)
            sectionName_list.append(sectionName)
            parentID_list.append(parentID)
            length_list.append(L)
            
            # create data
            data = {'sectionID': sectionID,
                    'parentID': parentID,
                    'sectionName': sectionName,
                    'parentName': parentName,
                    'length': L,
                    'sectionType': sectionType}

            df = pd.concat([df, pd.DataFrame(data, index=[0])])

        df.to_csv("cell1.csv", encoding='utf-8', index=False)
        Data = open('cell1.csv', "r")
        next(Data, None)  # skip the first line in the input file
        Graphtype = nx.Graph()
        self.G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                                  nodetype=int, data=(('sectionName', str), ('parentName', str),
                                                      ('length', float), ('sectionType', str)))
        self.sp = dict(nx.all_pairs_shortest_path(self.G))

        self.parentID_list, self.sectionID_list, self.sectionName_list, self.length_list = parentID_list, sectionID_list, sectionName_list, length_list
    
    def _add_synapses(self):
        sectionID_list, sectionName_list = self.sectionID_list, self.sectionName_list
        synapses_list, netstims_list, netcons_list, randoms_list = [], [], [], []
        sections_basal_apical = [i for i in map(list, list(self.complex_cell.basal))] + [i for i in map(list, list(self.complex_cell.apical))]
        
        rnd = np.random.RandomState(10)
        if rnd.uniform() < 0.85:
            e_syn, tau1, tau2, spike_interval, syn_weight = 0, 0.3, 1.8, 1000/2.5, 0.0016
        else:
            e_syn, tau1, tau2, spike_interval, syn_weight = -86, 1, 8, 1000/15.0, 0.0008

        loc_list, type_list, sectionID_synapse_list, section_synapse_list = [], [], [], []

        for i in tqdm(range(self.numSyn)):
            Section = rnd.choice(sections_basal_apical)
            section = Section[0].sec
            sectionName = section.psection()['name']
            sectionID_synapse = sectionID_list[sectionName_list.index(sectionName)]
            
            section_synapse_list.append(section)
            sectionID_synapse_list.append(sectionID_synapse)
    
            # Use to differentiate between input type A and B
            type_list.append(rnd.choice(['A', 'B']))

            loc = section(rnd.uniform()).x
            loc_list.append(loc)
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
            
            time.sleep(0.01)
        
        self.loc_list, self.type_list, self.sectionID_synapse_list, self.section_synapse_list = loc_list, type_list, sectionID_synapse_list, section_synapse_list
        self.visualize_synapses()
        
        self.visualize_simulation()

    def get_cell(self):
        return self.complex_cell

    def visualize_synapses(self):
        pass

    def calculate_distance_matrix(self, distance_limit=2000):
        loc_list, sectionID_synapse_list = self.loc_list, self.sectionID_synapse_list
        parentID_list, length_list = self.parentID_list, self.length_list
        
        distance_matrix = np.zeros((self.numSyn, self.numSyn))
        for i in range(self.numSyn):
            for j in range(self.numSyn):
                if i < j:
                    m = sectionID_synapse_list[i]
                    n = sectionID_synapse_list[j]

                    path = self.sp[m][n]

                    distance = 0

                    if len(path) > 1:
                        loc_i = loc_list[i] * (parentID_list[m] == path[1]) + (1-loc_list[i]) * (parentID_list[m] != path[1])
                        loc_j = loc_list[j] * (parentID_list[n] == path[-2]) + (1-loc_list[j]) * (parentID_list[n] != path[-2])
                        for k in path:
                            if k == m:
                                distance = distance + length_list[k]*loc_i
                            if k == n:
                                distance = distance + length_list[k]*loc_j
                            distance = distance + length_list[k]
                    else:
                        distance = length_list[m] * abs(loc_list[i] - loc_list[j])

                    distance_matrix[i, j] = distance_matrix[j, i] = distance

        self.distance_matrix = distance_matrix

        return self.distance_matrix, self.type_list
    
    def visualize_simulation(self):
        soma_v = h.Vector().record(self.complex_cell.soma[0](0.5)._ref_v)
        dend_v = h.Vector().record(self.complex_cell.dend[0](0.5)._ref_v)
        apic_v = h.Vector().record(self.complex_cell.apic[0](0.5)._ref_v)
        time_v = h.Vector().record(h._ref_t)

        h.tstop = 1000
        st = time.time()
        h.run()
        print('complex cell simulation time {:.4f}'.format(time.time()-st))

        # plotting the results

        plt.figure()
        plt.plot(time_v, soma_v, label='soma')
        plt.plot(time_v, dend_v, label='basal')
        plt.plot(time_v, apic_v, label='apical')

        plt.legend()
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        # plt.show()


