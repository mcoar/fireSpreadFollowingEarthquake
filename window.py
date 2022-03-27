# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 17:19:09 2020

@author: Lab User
"""

import numpy as np
import random

class Window():
    
    # Constructor
    def __init__(self, name, floor, bay, height=48., width=30., sill_height=36.):
        self.name = name
        self.floor = floor
        self.bay = bay
        # self.compartment = compartment
        self.height = height        # in
        self.width = width          # in
        self.sill_height = sill_height      # in
        self.set_area()
        
    # Setters
    def set_name(self,name):
        self.name = name
    def set_floor(self,floor):
        self.floor = floor
    def set_bay(self,bay):
        self.bay = bay
    def set_height(self,height):
        self.height = height
        self.set_area()
    def set_width(self,width):
        self.width = width
        self.set_area()
    # def set_compartment(self,compartment):
    #     self.compartment = compartment
    def set_sill_height(self,sill_height):
        self.sill_height = sill_height
    def set_area(self):
        self.area = self.width*self.height
        
    
    # Getters
    def get_name(self):
        return self.name
    def get_floor(self):
        return self.floor
    def get_bay(self):
        return self.bay
    def get_height(self):
        return self.height
    def get_height_m(self):
        return self.height/39.4
    def get_width(self):
        return self.width  
    def get_sill_height(self):
        return self.sill_height
    def get_area(self):
        return self.area
    def get_height_m(self):
        return self.height/39.4
    def get_width_m(self):
        return self.width/39.4
    def get_sill_height_m(self):
        return self.sill_height/39.4
    def get_area_m2(self):
        return self.area/1550.
    
    # Descriptors
    def describe(self):
        print(":::Window " + self.get_name() + " Description:::")
        print("Floor: " + str(self.floor) + ", Bay: " + str(self.bay) )
        # print("Compartment: " + str(self.compartment))
    def describe_location(self):
        print("Floor: " + str(self.floor) + ", Bay: " + str(self.bay) )
        
    # Flame Geometry
    # see Lee/Davidson 2010a and Law/O'Brien 1984
    def flame_geometry( self, has_draft, is_top_floor, mdot_r, u, summary=False):
        # INPUTS:
        # has_draft [bool]: True if room has a draft. defined in Lee/Davidson 2010
        # is_top_floor [bool]: True if there is no wall above this window. 
        # mdot_r [float, kg/s]: rate of burning in compartment. see Lee/Davidson 2010
        # u [float, m/s]: wind speed. positive for speed away from facade, negative for speed towards facade.
        # summary: prints a summary
        # OUTPUTS:
        # h_w [float, m]: height of flame tip above top of window.
        # x_w [float, m]: centerline projection of flame plume from window face.
        # w_w [float, m]: width of flame
        # lambda_w [float, m]: thickness of flame plume at top
        # define window geometry
        h_d = self.get_height_m()
        w_d = self.get_width_m()
        A_d = self.get_area_m2()
        # define flame width
        w_w = w_d                   # flame width is always same as window width. 
        if has_draft:               # if there is a draft ... 
            lambda_w = 0.;          # no thickness for draft condition..
            if u <= 0.:             # if this window is upwind (has a negative wind speed)...
                h_w = 0.; x_w = 0.
            else:                   # else if window is downwind (positive wind speed)...
                h_w = 23.9 * u**(-0.43) * mdot_r * A_d**(-0.5) - h_d
                x_w = 0.605 * (u**2/h_d)**0.22 * (h_w + h_d)
        else:                       # no draft condition
            h_w = 12.8 * (mdot_r / w_w)**(2/3) - h_d
            lambda_w = 2./3 * h_d  # flame thickness (m)
            if is_top_floor:
                x_w = 0.6 * h_d * (h_w / h_d)**(1/3)
            else: 
                if h_d < 1.25*w_d:
                    x_w = h_d/3
                else:
                    x_w = 0.3 * h_d * (h_d / w_w)**(0.54)
        if summary:
            print('h_w = {0:0.2f}, x_w = {1:0.2f}, w_w = {2:0.2f}, t_w = {3:0.2f}'.format(h_w,x_w,w_w,lambda_w))                          
        return h_w, x_w, w_w, lambda_w
        
class Door():
    
    # Initiator
    def __init__(self, name, is_open=False, height = 80, width = 36):
        self.name = name        # str
        self.is_open = is_open  # bool
        self.height = height    # in
        self.width = width      # in
        self.set_area()
    
    # Setters
    def set_name(self,name):
        self.name = name
    def set_is_open(self, is_open):
        self.is_open = is_open
    def set_height(self,height):
        self.height = height
        self.set_area()
    def set_width(self,width):
        self.width = width
        self.set_area()
    def set_area(self):
        self.area = self.width*self.height
    
    # Getters    
    def get_name(self):
        return self.name
    def get_is_open(self):
        return self.is_open
    def get_height(self):
        return self.height
    def get_width(self):
        return self.width  
    def get_area(self):
        return self.area    
    def get_area_m2(self):
        return self.area/1550.