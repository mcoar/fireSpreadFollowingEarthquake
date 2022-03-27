# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 11:50:53 2020

@author: Lab User
"""

import callOS

pname = 'C:/Users/Lab User/OneDrive/Documents/OpenSees/thermalExamples/Ex-3_Pinned beam with thermal gradient/'
fname = 'pinned.tcl'
cmd = 'openSees_Eurocode ' + fname

callOS.myrun(cmd,pname)