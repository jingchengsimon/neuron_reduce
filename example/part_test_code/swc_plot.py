import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

swc_file = 'cell1'
segs = pd.read_csv('C:/Users/Windows/Desktop/Drew/neuron_reduce/example/{x}.swc'.format(x=swc_file),comment='#',sep=' ', 
    names=['id','type','x','y','z','r','pid'])

segs = segs.merge(segs,left_on='id',right_on='pid',suffixes=[None, '_p'])
# segs.columns

def seg2coords(s_x,s_y,e_x,e_y,s_r,e_r):
    # s_ prefix for start seg
    # e_ prefix for end seg
    # x is x-coord, y is y-coord, r is radius

    seg_ang = np.angle((e_x-s_x)+(e_y-s_y)*1j) 
    pt11x = s_x+s_r*np.cos(seg_ang-(np.pi/2))
    pt11y = s_y+s_r*np.sin(seg_ang-(np.pi/2))
    
    pt12x = s_x+s_r*np.cos(seg_ang+(np.pi/2))
    pt12y = s_y+s_r*np.sin(seg_ang+(np.pi/2))

    pt21x = e_x+e_r*np.cos(seg_ang-(np.pi/2))
    pt21y = e_y+e_r*np.sin(seg_ang-(np.pi/2))
    
    pt22x = e_x+e_r*np.cos(seg_ang+(np.pi/2))
    pt22y = e_y+e_r*np.sin(seg_ang+(np.pi/2))

    coords = [[pt11x, pt11y], [pt12x, pt12y], [pt22x, pt22y], [pt21x, pt21y]]

    return np.array(coords)

patches = []
for idx, curr_row in segs.iterrows():
    curr_poly = Polygon(seg2coords(curr_row['x'],curr_row['y'], 
        curr_row['x_p'],curr_row['y_p'],curr_row['r'],curr_row['r_p']),True)
    patches.append(curr_poly)

p = PatchCollection(patches)
p.set_facecolor('k')
fig, ax = plt.subplots()
ax.add_collection(p)
ax.axis('equal')
plt.xlim(-300, 300)
plt.ylim(-250, 1500)
plt.show()
# plt.savefig('../Figures/{x}.svg'.format(x=swc_file))