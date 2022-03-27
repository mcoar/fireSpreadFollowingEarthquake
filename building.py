# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 08:01:44 2020

@author: Lab User
"""

from compartment import Compartment

class Building():
    # Constructor
    def __init__(self,address,nfloors,nbays):
        self.address = address
        self.nfloors = nfloors
        self.nbays = nbays
        self.rooms = {}
        