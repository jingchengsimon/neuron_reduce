import networkx as nx 
import pylab as plt 
from networkx.drawing.nx_agraph import graphviz_layout
    
G = nx.Graph()
G.add_node(1) 
G.add_node(2) 
G.add_node(3) 
# G.add_node(4)
    
G.add_edge(1,2, len=2) 
G.add_edge(2,3, len=2) 
# G.add_edge(3,4, len=1) 
# G.add_edge(4,1, len=1) 
G.add_edge(3,1, len=10) 
# G.add_edge(4,2, len=1) 

pos=graphviz_layout(G) 
nx.draw(G, pos, node_size=1600, node_color=range(len(G)), with_labels=True, cmap=plt.cm.Dark2) 
plt.show()