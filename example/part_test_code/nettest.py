import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("cell1.swc",delimiter=" ",names=['sampleID','type','x','y','z','radius','parentID'])
df = df[['sampleID','parentID','type']]
df.to_csv("cell1.csv", encoding='utf-8', index=False)

Data = open('testcell.csv', "r")
next(Data, None)  # skip the first line in the input file
Graphtype = nx.Graph()

G = nx.parse_edgelist(Data, comments='t', delimiter=',', create_using=Graphtype,
                      nodetype=int, data=(('type', str),('length',float)))

sp = dict(nx.all_pairs_shortest_path(G))

print(sp[1][6])
print(list(G.edges(data=True)))
# print(nx.clustering(G))
plt.figure()
nx.draw(G,with_labels=True, font_weight='bold',node_size=100)
plt.show()
