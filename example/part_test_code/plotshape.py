from neuron import h,gui
import matplotlib.pyplot as plt
# h.load_file('cell1.swc')  # a morphology file
h.load_file("import3d.hoc")
h.nrn_load_dll('./mod/nrnmech.dll')
h.load_file('./modelFile/L5PCbiophys3.hoc')
h.load_file('./modelFile/L5PCtemplate.hoc')
complex_cell = h.L5PCtemplate('./modelFile/cell1.asc')


# ps = h.PlotShape(True)
# ps.show(1)

# s = h.PlotShape()
# s.mark(complex_cell.dend[0])
# for i in range(10):
#     s.color(2, sec=complex_cell.dend[i])
#     s.color(2, sec=complex_cell.apic[i])
import numpy as np

sec_list = np.array([sec for sec in h.allsec() if 'dend' in str(sec)])
sl = h.SectionList(sec_list)
ps = h.Shape(sl, True)
ps.show(0)

sec_list = np.array([sec for sec in h.allsec() if 'apic' in str(sec)])
sl = h.SectionList(sec_list)
ps1 = h.Shape(sl, True)
ps1.show(0)

print('end')