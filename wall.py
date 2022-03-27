# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 18:58:29 2020

@author: Lab User
"""
from window import Window, Door
import numpy as np
from scipy import stats
import random

class Wall():
    # Constructor
    def __init__(self, name, floor, bay, 
                 wall_type, fire_resistance, 
                 height=156.,length=360., orientation=0.,
                 material='gypsum',thickness=13):
        self.name = name
        self.floor = floor
        self.bay = bay
        # self.compartment = compartment
        self.wall_type = wall_type
        self.height = height      #in
        self.length = length       # in  
        self.material = material    # [str]
        self.thickness = thickness  # mm
        self.get_wall_thermal_properties()
        self.orientation = orientation  # normal dir. outwards from wall, deg CW from N
        self.set_area()
        self.has_window = False
        self.windows = []
        self.num_of_windows = 0
        self.calc_window_area()
        self.has_door = False
        self.doors = []
        self.num_of_doors = 0
        self.is_burned = False
        self.fire_resistance = fire_resistance
        self.ds = 0
        self.calc_burnthrough()
        
    # Setters
    def set_name(self,name):
        self.name = name
    def set_floor(self,floor):
        self.floor = floor
    def set_bay(self,bay):
        self.bay = bay
    def set_height(self,):
        self.height = height
        self.set_area()
    def set_length(self,length):
        self.length = length
        self.set_area()
    # def set_compartment(self,compartment):
    #     self.compartment = compartment
    def set_area(self):
        self.area = self.height*self.length
    def set_is_burned(self, is_burned=True):
        self.is_burned = is_burned
    def set_ds(self,ds):
        self.ds = ds
        
    # Getters
    def get_name(self):
        return self.name
    def get_floor(self):
        return self.floor
    def get_bay(self):
        return self.bay
    def get_type(self):
        return self.wall_type
    def get_height(self):
        return self.height
    def get_height_m(self):
        return self.height/39.4
    def get_length(self):
        return self.length  
    def get_length_m(self):
        return self.length/39.4
    def get_area(self):
        return self.area
    def get_area_m2(self):
        return self.area/1550.
    def get_num_of_windows(self):
        return len(self.windows)
    def get_window_area(self):
        return self.window_area
    def get_window_area_m2(self):
        return self.window_area/1550.
    def get_num_of_doors(self):
        return len(self.doors)
    def get_is_burned(self):
        return self.is_burned
    def get_burnthrough(self):
        return self.t_burnthrough
    def get_ds(self):
        return self.ds
    
    # Descriptors
    def describe(self):
        print(":::Wall " + self.get_name() + " Description:::")
        print(str(self.wall_type) + " Wall")
        print("Floor: " + str(self.floor) + ", Bay: " + str(self.bay) )
        print(str(self.get_length()) + "\" long by " + str(self.get_height()) + "\" high")
        print("Damage State: " + str(self.get_ds()) )
        # print("Compartment: " + self.compartment.get_name())
        if self.get_num_of_windows() == 0:
            print("No windows")
        else:
            print(str(self.get_num_of_windows()) + " Windows: ")
            for win in self.windows:
                print(win.get_name()+ ": " + str(win.get_width()) + "\" x " +
                      str(win.get_height()) + "\"")
        if self.get_num_of_doors() == 0:
            print("No doors")
        else:
            print(str(self.get_num_of_doors()) + " Doors: ")
            for door in self.doors:
                if door.get_is_open() == True:
                    print("Door " + door.get_name() + "is open.")
                else:
                    print("Door " + door.get_name() + "is closed.")
                    
    # Add windows and doors
    def add_window(self, new_window):
        self.windows.append(new_window)
        self.has_window = True
        self.num_of_windows = self.get_num_of_windows()
        self.calc_window_area()
    def rm_windows_all(self):
        self.windows = []
        self.has_window = False
        self.num_of_windows = 0    
    def add_door(self, new_door, is_Open=random.choice([True,False])):
        self.doors.append(new_door)
        self.has_door = True
        self.num_of_doors = self.get_num_of_doors()
    def rm_doors_all(self):
        self.doors = []
        self.has_door = False
        self.num_of_doors = 0
        
    # Check burnthrough times/details
    def calc_burnthrough(self):
        self.t_burnthrough_coef_var = 0.15
        allowable_wall_types = ['exterior','interior','interior non-bearing',
                                'ceiling','roof']
        allowable_fire_resistance = ['fire-resistive','protected','unprotected']
        if self.wall_type.lower() not in allowable_wall_types:
            raise ValueError('Not an allowable wall type. Must be: '+ 
                             '\'exterior\',\'interior\',\'interior non-bearing\',\'ceiling\',\'roof\'')
        if self.fire_resistance.lower() not in allowable_fire_resistance:
            raise ValueError('Not an allowable fire-resistance. Must be: '+
                             '\'fire-resistive\',\'protected\',\'unprotected\'')
        if self.fire_resistance.lower() == 'unprotected':
            self.t_burnthrough_mean = 0.25 # testing. Actual value should be 0.25
        elif self.fire_resistance.lower() == 'protected':
            self.t_burnthrough_mean = 1.
        elif self.fire_resistance.lower() == 'fire-resistive':
            self.t_burnthrough_mean = 2.
            if self.wall_type.lower() == 'roof':
                self.t_burnthrough_mean = 1.5
        if self.wall_type.lower() == 'interior non-bearing':
            self.t_burnthrough_mean = 0.25
        
        self.t_burnthrough_std = self.t_burnthrough_coef_var*self.t_burnthrough_mean
        
        mu, sigma = self.t_burnthrough_mean, self.t_burnthrough_std
        self.t_burnthrough = np.random.lognormal(np.log(mu),sigma,1)
        # print(self.t_burnthrough)
    
    # get total area of windows
    def calc_window_area(self):
        self.window_area = 0.
        if self.get_num_of_windows() > 0:
            for window in self.windows:
                self.window_area += window.get_area()
                
    # get damage state of wall
    def calc_damage_state(self,IDR,typ='all'):
        """
        Calculates the damage state of a steel-framed gypsum wall based on the 
        interstory drift ratio (IDR). 
        
        Reference:
        Pali, T, Macillo, V, Terracciano MT, Bucciero, B, Fiorino, L, and 
        Landolfo, R (2018) In-plane quasi-static cyclic tests of nonstructural
        lightweight steel drywall partitions for seismic performance 
        evaluation. Earthquake Eng. Str. Dyn. 47:1566-1588.

        Parameters
        ----------
        IDR : FLOAT - [0 1]
            drift between nodes of floor n and n+1, divided by height of 
            story n
        typ: str - ['all','t1fix','t1slide','t2fix','t2slide']
            wall typology
            t1: structural elements all 4 sides
            t2: structural elements top and bottom, limited non-structural
                elements left and right
            fix: fixed connections to all structural elements
            slide: top of wall slides along structural elements
            all: uses all samples from the study

        Returns
        -------
        ds : int - [0 3]
            ds0 - no damage
            ds1 - superficial damage - repair with plaster, tape, and paint
            ds2 - local damage to sheathing panels or steel frame components
            ds3 - severe damage - replacement of part or whole wall

        """
        allowed_wall_types = ['all','t1fix','t1slide','t2fix','t2slide']
        if typ.lower() not in allowed_wall_types:
            raise ValueError('Not an allowable wall type. See documentation')
        elif typ.lower() == 'all':
            xm, beta = [0.57, 1.20, 1.71], [.64, .32, .32]
        elif typ.lower() == 't1fix':
            xm, beta = [0.37, 1.05, 1.49], [.51, .29, .29]
        elif typ.lower() == 't1slide':
            xm, beta = [1.27, 1.52, 2.10], [.28, .25, .25]
        elif typ.lower() == 't2fix':
            xm, beta = [0.78, 1.18, 1.44], [.35, .28, .39]
        elif typ.lower() == 't2slide':
            xm, beta = [0.70, 1.01, 1.20], [.39, .25, .32]
        
        p_e = list()
        r = np.random.rand()
        ds = 0
        for i in range(3):
            dist = stats.lognorm(s=beta[i], scale=xm[i], loc=0)
            p = dist.cdf(IDR)
            p_e.append(p)
            if r <= p:
                self.ds = i+1
        return p_e, self.ds, r
    
    def get_wall_thermal_properties(self):
        allowed_wall_materials = ['gypsum','concrete']
        if self.material.lower() not in allowed_wall_materials:
            errortext = 'Not an allowable wall material. Wall {}'.format(self.name)
            raise ValueError(errortext)
        elif self.material.lower() == 'gypsum':
            # From You et al 1986 'Spray cooling in room fires'
            self.volumetric_heat_capacity = 818.    # 'k', kJ/m3C
            self.thermal_conductivity = 3.6e-4      # 'rho*c_p', kW/mC
            self.thermal_diffusivity = self.thermal_conductivity/self.volumetric_heat_capacity  # 'alpha'
        elif self.material.lower() == 'concrete':
            self.volumetric_heat_capacity = 2086.    # 'k', kJ/m3C
            self.thermal_conductivity = 8.0e-4      # 'rho*c_p', kW/mC
            self.thermal_diffusivity = self.thermal_conductivity/self.volumetric_heat_capacity  # 'alpha'
