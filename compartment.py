# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 10:37:54 2020

@author: Lab User
"""
import numpy as np
import random
from window import Window, Door
from wall import Wall
import sys

mH202psi = 1/1.42233 # meters of water to psi

class Compartment():
    
    temp_pathname = r"C:\\Users\\Lab User\\OneDrive\\Documents\\OpenSees\\TemperatureCurves\\"
    # temp_pathname = r"C:\\Users\\maxwe\\OneDrive\\Documents\\OpenSees\\TemperatureCurves\\"
    temp_filename = 'time-temp.txt'
    time_temp = np.loadtxt(temp_pathname+temp_filename, skiprows = 1)
    
    # %% Constructor
    def __init__(self, name, floor, bay, occupancy='Residential',
                 spr_list = [], pressure = [], min_pressure = [],
                 min_flow=0.9,spr_area=130.,
                 dim1=360.,dim2=360.,height=156.):
        self.name = name
        self.floor = floor
        self.bay = bay
        self.set_occupancy(occupancy)
        self.height = height
        self.dim1 = dim1
        self.dim2 = dim2
        self.sprinklers = tuple(spr_list) # list of strings - names of sprinklers in compartment
        self.pressure = pressure
        self.min_pressure = min_pressure    # psi
        # self.spr_ts = pressure.index[1]-pressure.index[0]
        self.set_area() 
        self.neighbors = {}
        self.is_burning = False
        self.has_draft = False
        self.time_of_ignition = None
        self.is_decayed = False
        self.is_fully_developed = False
        self.time_of_full_development = None
        self.stage = 'not ignited'
        self.time_of_decay = None
        self.time_curve = 0
        self.temp_curve = None
        self.doors_open = {}
        self.walls = {}
        self.check_draft_condition()
        self.T_a = 293.          # Ambient temperature [deg K]
        self.T_r = 293.         # Room temperature, deg K
        self.percent_burned = 0.
        self.rate_of_burning = 0.        # rate of burning, mdot_r [kg/s]
        
    # %% Setters
    def set_name(self, name):
        self.name = name
    def set_floor(self, floor):
        self.floor = floor
    def set_bay(self, bay):
        self.bay = bay
    def set_occupancy(self, occupancy):
        self.occupancy = occupancy
        self.calc_fuel_load_density()
    def set_dim(self,dim1=360.,dim2=360.,height=156.):
        self.dim1 = dim1
        self.dim2 = dim2
        self.height = height
        self.set_area()
    def set_sprinklers(self, sprinklers):
        self.sprinklers = sprinklers
    def set_ignition_location(self, grid_space):
        self.ignition_location = grid_space
    def set_area(self):
        self.area = self.dim1*self.dim2     # in2
        self.area_ft2 = self.area/144.
        self.area_m2 = self.area/1550.
    def set_is_burning(self, is_burning=True):
        if isinstance(is_burning, (bool)):
            self.is_burning = is_burning
        else:
            raise TypeError("is_burning must be [bool]")
    def set_is_fully_developed(self, is_fully_developed=True):
        if isinstance(is_fully_developed, (bool)):
            self.is_fully_developed = is_fully_developed
        else:
            raise TypeError("is_fully_developed must be [bool]") 
    def set_is_decayed(self, is_decayed=True):
        if isinstance(is_decayed,(bool)):
            self.is_decayed = is_decayed
        else:
            raise TypeError('is_decayed must be [bool]')
    def set_has_draft(self, has_draft):
        if isinstance(has_draft, (bool)):
            self.has_draft = has_draft
        else:
            raise TypeError("has_draft must be [bool]")
    def set_time_of_ignition(self, time_of_ignition):
        self.time_of_ignition = time_of_ignition
    def set_time_of_full_development(self,time_of_full_development):
        self.time_of_full_development = float(time_of_full_development)
    def set_time_of_decay(self,time_of_decay):
        self.time_of_decay = time_of_decay
    def set_time_curve(self, time_curve):
        self.time_curve = time_curve
    def set_temp_curve(self, temp_curve):
        self.temp_curve = temp_curve
    
    # %% Getters
    def get_name(self):
        return self.name
    def get_bay(self):
        return self.bay
    def get_floor(self):
        return self.floor
    def get_occupancy(self):
        return self.occupancy
    def get_floor_area_m2(self):
        return self.area_m2
    def get_stage(self):
        return self.stage
    def get_sprinklers(self):
        return self.sprinklers
    def get_num_of_sprinklers(self):
        return len(self.sprinklers)
    def get_num_of_operable_sprinklers(self):
        return self.num_of_operable_sprinklers
    def get_dim(self):
        return self.height, self.dim1, self.dim2
    def get_dim_m(self):
        return self.height/39.4, self.dim1/39.4, self.dim2/39.4
    def get_total_area(self):
        return self.total_area_m2*1550.
    def get_total_area_m2(self):
        return self.total_area_m2
    def get_total_door_area_m2(self):
        return self.total_door_area_m2
    def get_total_window_area(self):
        return self.total_window_area_m2*1550.
    def get_total_window_area_m2(self):
        return self.total_window_area_m2
    def get_total_window_area_m2(self):
        return self.total_door_area_m2
    def get_total_sprinkler_area_ft2(self):
        return self.total_sprinkler_area_ft2
    def get_total_sprinkler_area_m2(self):
        return self.total_sprinkler_area_ft2/10.76
    def get_fire_load_density(self):
        return self.fire_load_density
    def get_neighbors(self):
        return self.neighbors
    def get_num_of_neighbors(self):
        return len(self.neighbors)
    def get_walls(self):
        return self.walls
    def get_num_of_walls(self):
        return len(self.walls)
    def get_is_burning(self):
        return self.is_burning
    def get_is_fully_developed(self):
        return self.is_fully_developed
    def get_is_decayed(self):
        return self.is_decayed
    def get_time_of_full_development(self):
        return self.time_of_full_development
    def get_has_draft(self):
        return self.has_draft
    def get_time_of_ignition(self):
        return self.time_of_ignition
    def get_time_of_decay(self):
        return self.time_of_decay
    def get_time_curve(self):
        return self.time_temp[:,self.time_curve]
    def get_temp_curve(self):
        return self.time_temp[:,self.temp_curve]
    def get_current_temp(self,current_time):
        t_c = self.get_time_curve()
        T_c = self.get_temp_curve()
        t_relative = current_time - self.time_of_ignition
        t_idx = np.searchsorted(t_c, t_relative)
        current_temp = T_c[t_idx-1]+ (T_c[t_idx]-T_c[t_idx-1])/(t_c[t_idx]-t_c[t_idx-1])*(t_relative-t_c[t_idx-1])
        return current_temp
    def get_mdot_r(self):
        return self.rate_of_burning
    def get_left_door_open(self):
        return self.left_door_open
    def get_right_door_open(self):
        return self.right_door_open
    
    # %% Math stuff
    def round_to_nearest(num, base):
        n = num + (base//2)
        return n - (n % base)
        
    # %% Descriptors
    def describe(self):
        print(":::Compartment " + self.get_name() + " Description:::")
        print("Floor: " + str(self.floor) + ", Bay: " + str(self.bay) )
        print("Occupancy: " + self.occupancy + ".")
        print('Floor Area: {:0.1f} sqm'.format(self.get_floor_area_m2()))
        print("Neighbors:")
        if self.get_num_of_neighbors()==0:
            print('   No neighbors currently assigned.')
        else:
            for direction in self.neighbors:    
                compartment = self.neighbors[direction]
                print("   Compartment " + compartment.get_name() + " is " + direction + ".")
        print("Walls:")
        if self.get_num_of_walls()==0:
            print('   No walls currently assigned.')
        else: 
            for name in self.walls:
                wall = self.walls[name]
                num_of_win = wall.get_num_of_windows()
                num_of_door = wall.get_num_of_doors()
                print('    ' + wall.get_name() + ': ' + str(num_of_win) + ' window(s), '
                      + str(num_of_door) + ' door(s).')
        print("Sprinklers:")
        print('    {:d}/{:d} sprinklers are operable (>90% design flow)'.format(self.get_num_of_operable_sprinklers(),self.get_num_of_sprinklers()))
        print('    Total area of coverage: {:0.1f} sqm'.format(self.get_total_sprinkler_area_m2()))
        print("Fire Conditions:")
        print('    {0:0.1f} % burned.'.format(self.percent_burned*100.))
        if self.get_has_draft():
            print('    Compartment has a draft.')
        else: 
            print('    Compartment does not have a draft.')
        print('    Fuel Load of {0:0.1f} kg/m2.'.format(self.fuel_load_density))
        if self.get_is_burning():
            print("    Began burning at time: {0:0.1f}".format(float(self.time_of_ignition)))
            if self.get_is_fully_developed():
                print("    Fully developed at time: {0:0.1f}".format(float(self.time_of_full_development)))
                print('    Full development temperature: {0:0.1f}'.format(self.T_r*100.))
            else:
                print("    Not fully developed")
        else:
            print("    Not currently burning")
    def describe_location(self):
        print("   Floor: " + str(self.floor) + ", Bay: " + str(self.bay) )
        
    # %% Add neighbors
    def add_neighbor(self, new_neighbor, direction):
        self.neighbors[direction] = new_neighbor
        if direction == 'left': 
            opp_dir = 'right'
        elif direction == 'right': 
            opp_dir = 'left'
        else:
            return
        # print('adding door on ' + direction + ' side.')
        if direction not in self.doors_open:
            if opp_dir in new_neighbor.doors_open:
                self.doors_open[direction] = new_neighbor.doors_open[opp_dir]
            else:
                self.doors_open[direction] = random.choice([True,False])
        else:
            return    
    
    def add_wall(self, new_wall, name):
        self.walls[name] = new_wall
        self.check_draft_condition()
        self.calc_total_area()
        
    # %% Fire
    def ignite(self, time_of_ignition, time_curve, temp_curve, ign_loc=0):
        self.set_is_burning(True)
        self.set_time_of_ignition(time_of_ignition)
        self.set_time_curve(time_curve)
        self.set_temp_curve(temp_curve)
        self.ign_loc = ign_loc
        self.stage = 'growth'
    def fully_develop(self, time_of_full_development):
        self.set_is_fully_developed(True)
        self.set_time_of_full_development(time_of_full_development)
        self.stage = 'fully developed'
    def decay(self, time_of_decay):
        self.set_is_fully_developed(False)
        self.set_time_of_decay(time_of_decay)
        self.set_is_decayed()
        self.stage = 'decayed'
        
    # %% Check for draft condition
    def check_draft_condition(self):
        num_of_walls_with_windows = 0
        num_of_walls_with_draft_condition = 0
        for wall in self.walls.values():
            if wall.has_window:
                num_of_walls_with_windows += 1
                continue
            elif wall.get_is_burned():
                num_of_walls_with_draft_condition += 1
                continue
            elif wall.get_ds() >= 2:
                num_of_walls_with_draft_condition += 1
                continue
            elif wall.has_door:
                for door in wall.doors:
                    if door.get_is_open():
                        num_of_walls_with_draft_condition += 1
            if (    num_of_walls_with_windows < 2 
                 and not
                    (     num_of_walls_with_windows >= 1
                      and num_of_walls_with_draft_condition >= 1
                    )
                ):
                self.set_has_draft(False)
            else:
                self.set_has_draft(True)
                break
            
    # %% Calculate occupancy-based fuel load density
    # Source: Modeling post-earthquake fire spread. Diss. Cornell. Lee 2009
    def calc_fuel_load_density_residential(self):
        return np.random.normal(16,4.4)
    def calc_fuel_load_density_hospital(self):
        return np.random.normal(5.4, 1.65)
    def calc_fuel_load_density_government(self):
        return np.random.normal(27.75,31.25)
    def calc_fuel_load_density_office(self):
        return np.random.normal(29,26.75)
    def calc_fuel_load_density_store(self):
        return 46.75
    def calc_fuel_load_density_warehouse(self):
        return 113.5
    def calc_fuel_load_density_school(self):
        return np.random.uniform(31.75,177)

    def calc_fuel_load_density(self):
        occupancy_switcher = {
            'residential':  self.calc_fuel_load_density_residential,
            'hospital':     self.calc_fuel_load_density_hospital,
            'government':   self.calc_fuel_load_density_government,
            'office':       self.calc_fuel_load_density_office,
            'store':        self.calc_fuel_load_density_store,
            'warehouse':    self.calc_fuel_load_density_warehouse,
            'school':       self.calc_fuel_load_density_school
            }
        if self.occupancy not in occupancy_switcher.keys():
            raise ValueError ('Not an allowable occupancy type, must be: ' + ', '.join(occupancy_switcher.keys()))
        func = occupancy_switcher.get(self.occupancy,lambda:'residential')
        self.fuel_load_density = max(func(),0.1)
        
    # %% Update rate of burning, phase, and room temperature
    def update_fire_phase(self,t):
        # t is the time in seconds since ignition
        self.check_draft_condition()
        self.calc_total_area()            
        self.total_room_fire_load = self.area_m2*self.fuel_load_density     # L, kg
        psi = self.total_room_fire_load / np.sqrt(self.get_total_window_area_m2()*self.get_total_area_m2())
        if self.get_has_draft(): # If there's a draft
            # update fuel and burning rate
            self.rate_of_burning = self.total_room_fire_load/1200.   # mdot_r, kg/s
            self.T_r = self.T_a + 1200.*(1.-np.exp(-0.04*psi))        # T_r, Room temperature, degK
        else: # If there's not a draft
            self.rate_of_burning = 0.18 * (1.-np.exp(-0.036*self.eta)) * self.get_total_window_area_m2() * np.sqrt(self.h_d_m / (self.height/self.w_r_m) )
            self.T_r = self.T_a + 6000*(1.-np.exp(-0.1*self.eta))*(1-np.exp(-0.05*psi))*self.eta**(-1/2)
        # Note: might have to update percent burned to take into account a change in draft status.
        self.percent_burned = self.rate_of_burning * t / self.total_room_fire_load  #L_t, --/--
        # update phase
        if self.get_stage() == 'growth':
            t1=0
            try:
                # pressure_at_ign_loc = self.pressure.loc[t1,self.sprinklers[self.ign_loc]]
                pressure_at_ign_loc = max(self.pressure.loc[:,self.sprinklers[self.ign_loc]])
            except:
                breakpoint()
            if (pressure_at_ign_loc*mH202psi)>=self.min_pressure:
                self.decay(t)
                print(self.name + ' was extinguished by sprinkler {}'.format(self.sprinklers[self.ign_loc]))
            elif self.percent_burned >= 0.3 and self.percent_burned <=0.8:
                self.fully_develop(self.get_time_of_ignition()+t)
                print(self.name + ' just fully developed.')
        elif self.get_stage()=='fully developed' and self.percent_burned > 0.8:
            self.decay(t)
            print(self.name + ' just decayed. Total time of FD: {0:0.1f}'.format(t-self.get_time_of_full_development()+self.get_time_of_ignition()))

    def calc_total_area(self):
        # self.total_area = 0.
        self.total_area_m2 = 0.
        # self.total_window_area = 0.
        self.total_window_area_m2 = 0.
        self.total_door_area_m2 = 0.     # area of doors (m2)
        self.h_d_m = 0.       # window height (m)
        self.w_r_m = 0.       # width of wall containing window (m)
        for wall in self.walls.values():
            self.total_area_m2 += wall.get_area_m2() # A_r,T (m2) Total area of floors, walls, and ceiling, minus window area
            # print('total area: {:0.1f}'.format(self.total_area))
            for door in wall.doors:
                self.total_door_area_m2 += door.get_area_m2()
                self.total_area_m2 -= door.get_area_m2()
                self.door_height_m = door.height/39.4
            if wall.get_num_of_windows() > 0:
                self.total_window_area_m2 += wall.get_window_area_m2() # A_d, window area (m2)
                self.w_r_m += wall.get_length_m() #w_r, width of wall containing windows (m)
                # print('  total window area: {:0.1f}'.format(self.total_window_area))
                # print('  total area: {:0.1f}'.format(self.total_area))
                for window in wall.windows:
                    self.h_d_m = max(self.h_d_m,window.get_height_m())
        if self.total_window_area_m2 <= 0.1:
            self.total_window_area_m2 = self.total_door_area_m2 / 2.
        self.eta = self.get_total_area_m2() / ( self.get_total_window_area_m2() * np.sqrt(self.h_d_m))
    
    def calc_sprinkler_area(self,min_flow=0.9,spr_area = 130.):
        # min flow is float between 0 and 1, percent operability compared to design flow
        # spr area is float, area of sprinkler coverage in sqft
        n = self.get_num_of_sprinklers()
        n_oper = 0 # number of operable sprinklers (flow greater than min flow)
        for s in self.sprinklers:
            if self.p.vs.find(s)['flow'] > min_flow:
                n_oper += 1
        self.total_sprinkler_area_ft2 = spr_area*n_oper
        self.num_of_operable_sprinklers = n_oper
            
    def calc_heat_loss(self):
        # You et al 1986 "Spray cooling in room fires"
        Qs = 0.        # conductive heat loss rate through walls and ceiling
        # T_r taken as average room temperature along entire height of wall
        # Temperature at ceiling assumed to be 150% of average room temperature
        for wall in self.walls.values():
            k = wall.thermal_conductivity
            L = wall.thickness
            A = wall.get_area_m2()
            for door in wall.doors:
                A -= door.get_area_m2()
            if wall.wall_type.lower() == 'ceiling':
                dT = 1.5*self.T_r - self.T_a
            else:
                dT = self.T_r - self.T_a
            qi = k*A*dT/L   # 1-D steady state thermal conduction equation
            Qs += qi        # sum across all walls and ceiling
        # Radiative heat loss through the floor and openings is considered to
        # be very small. Can be updated in the future, if necessary.
        Qf = 0.     # radiative heat loss through the floor
        Qr = 0.     # ratiative heat loss through the openings (windows, doors)
        self.Ql = Qs + Qf + Qr 
        
    def get_sprinkler_info(self):
        K = 5.6         # GPM/psi^(1/2)
        K_SI = 80.6     # LPM/bar^(1/2)
        D = 0.5         # in
        D_m = D/39.4  # m
        Dbar = D_m/.0111 # unitless 
        P = self.pressure.iloc[0,:].to_numpy()  # mH2O
        Pbar = P*9.806/17.2 # unitless, but calc includes conversion from mH20 to kPa 
        W = K_SI*np.sqrt(P*9.806/100.) # flow in liters. Have to convert P from mH20 to bar
        return W, Dbar, Pbar
