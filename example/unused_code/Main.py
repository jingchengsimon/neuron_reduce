# from neuron import gui, h
from CellwithNetworkx import *
from DistanceAnalyzer import *
# import matplotlib.pyplot as plt
# import time

# Example usage:
swc_file_path = 'C:/Users/Windows/Desktop/Drew/neuron_reduce/example/modelFile/cell1.asc'
num_synapses_to_add = 1000

cell1 = CellwithNetworkx(swc_file_path, num_synapses_to_add)
distance_matrix, type_list = cell1.calculate_distance_matrix()
bin_list = [0, 2.7, 4.5, 7.4, 12, 20, 33, 55, 90, 148, 245]
bin_list = [0, 4.5, 12, 33, 90, 245]

num_epochs = 1000
analyzer = DistanceAnalyzer(distance_matrix, type_list, bin_list, num_epochs)
analyzer.cluster_shuffle()

# type_array_clustered = np.array(analyzer.type_list_clustered)
# section_synapse_array = np.array(cell1.section_synapse_list)
# section_synapse_array_A = section_synapse_array[type_array_clustered == 'A']
# section_synapse_array_B = section_synapse_array[type_array_clustered == 'B']

# sl = h.SectionList(section_synapse_array_A)
# ps = h.Shape(sl, True)
# ps.show(0)

# sl = h.SectionList(section_synapse_array_B)
# ps1 = h.Shape(sl, True)
# ps1.show(0)

analyzer.visualize_learning_curve()
analyzer.visualize_results()
plt.title('Cluster Shuffle')

# analyzer.cluster_traditional()
# analyzer.visualize_results()
# plt.title('Cluster Traditional')
# plt.show()

# Display matrix
# plt.matshow(distance_matrix) 
plt.show()

