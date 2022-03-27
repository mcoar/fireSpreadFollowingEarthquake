# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:48:09 2021

@author: Lab User
"""
#%% start it up and import
import wntr
import numpy as np
import pickle

#%% constants, variables, units, and toggles
# distance
m = 1.          # meters
sec  = 1.       # sec
inch = .0254*m  # inches to meters
ft = 12*inch   # feet to meters
yd = 3*ft      # yards to meters

# pressure
psi = .7031*m

# flow
m3s = 1.
gpm = 1./15850.*m3s
ft3s = (ft**3)*m3s

# time
minute = 60.*sec
hour = 60.*minute
day = 24.*hour

s  = 10*ft       # sprinkler spacing along branch lines
xm = 10*ft      # sprinkler spacing across branch lines (distance between branches)
ground = 50*ft  # ground height of the building
h  = 13*ft      # height of a floor
h1 = 18*ft      # height of the first floor
g  = 9.81*m/sec**2     # gravitational acceleration

meter2inch = 1/inch
end_time = 3600


# unit choice (for output ONLY)
unitp = ft
name_unitp = 'ftH20'
unitf = ft3s
name_unitf = 'cuft/s'
unitl = inch
name_unitl = 'inch'



#%% Set up the network
wn = wntr.network.WaterNetworkModel()
wn.add_pattern('pat1', [1])
wn.add_pattern('pat2', [1,2,3,4,5,6,7,8,9,10])
wn.add_junction('node1', base_demand=0.01, demand_pattern='pat1', elevation=ground, coordinates=(1,2))
wn.add_junction('node2', base_demand=0.02, demand_pattern='pat2', elevation=ground+h1, coordinates=(1,3))
wn.add_junction('node3', base_demand=0.03, demand_pattern='pat2', elevation=ground+h1+h, coordinates=(1,4))
wn.add_reservoir('res', base_head=326*ft, head_pattern='pat1', coordinates=(0,2))
wn.add_pipe('piperes', 'res', 'node1', length=100*ft, diameter=6*inch, roughness=100, minor_loss=0.0, status='OPEN')
wn.add_pipe('pipe1', 'node1', 'node2', length=h1, diameter=4*inch, roughness=100, minor_loss=0.0, status='OPEN')
wn.add_pipe('pipe2', 'node2', 'node3', length=h, diameter=4*inch, roughness=100, minor_loss=0.0, status='OPEN')
ax = wntr.graphics.plot_network(wn)

# leak list
leak_list = ['pipe2']

#%% Pickle the network 
f=open('wn.pickle','wb')
pickle.dump(wn,f)
f.close()

#%% Run the undamaged network with WNTR
sim = wntr.sim.WNTRSimulator(wn)
results_undam = sim.run_sim()
print('------------------------------------------')
print('Undamaged network with WNTR')
print('------------------------------------------')
print('===PRESSURE ({})==='.format(name_unitp))
print(results_undam.node['pressure']/unitp)
print('===FLOWRATE ({})==='.format(name_unitf))
print(results_undam.link['flowrate']/unitf)

#%% Run the damaged network with WNTR

# Take the network out of the pickle jar

f=open('wn.pickle','rb')
wn = pickle.load(f)
f.close()

for leak in leak_list:
    # Add a leak
    ename = leak
    e = wn.get_link(ename)
    theta = np.radians(0.5)             # degrees
    area_modifier = 1.0
    leak_area = 0.5*np.pi*theta*np.power(e.diameter,2)*area_modifier
    
    wn = wntr.morph.split_pipe(wn,ename,ename+'_B',ename+'_leak_node')
    leak_node_name = ename+'_leak_node'
    leak_node = wn.get_node(leak_node_name)
    leak_node.add_leak(wn,leak_area,start_time=0.,end_time=end_time)


# Run the damaged network with WNTR
sim = wntr.sim.WNTRSimulator(wn)
results_dam = sim.run_sim()
print()
print('------------------------------------------')

print('Damaged network with WNTR')
print('------------------------------------------')
print('Leaking node name: ' + leak_node_name)
print('Pipe diam ({0}): {1:0.4f}'.format(name_unitl,e.diameter/unitl))
print('Pipe area ({0}): {1:0.4f}'.format(name_unitl+'2',np.power(e.diameter/unitl/2.,2)*np.pi))
print('Leak size ({0}): {1:0.4f}'.format(name_unitl+'2',leak_area/(unitl**2)))
print('===PRESSURE ({})==='.format(name_unitp))
print(results_dam.node['pressure']/unitp)
print('===FLOWRATE ({})==='.format(name_unitf))
print(results_dam.link['flowrate']/unitf)

#%% Performance Loss due to leak



print()
print('------------------------------------------')
print('PERFORMANCE LOSS')
print('------------------------------------------')
print('==PRESSURE DIFF ({})=='.format(name_unitp))
print(results_dam.node['pressure']/unitp-results_undam.node['pressure']/unitp)
print('==FLOW DIFF ({})=='.format(name_unitf))
print(results_dam.link['flowrate']/unitf-results_undam.link['flowrate']/unitf)


#%% Grab some results
# node_keys = results.node.keys()

# link_keys = results.link.keys()

# print(results.node['pressure'])

# print(results.link['flowrate'])