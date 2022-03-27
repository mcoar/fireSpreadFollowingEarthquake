# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:57:27 2020

https://www.semicolonworld.com/question/43537/getting-realtime-output-using-subprocess

@author: Lab User
"""

import subprocess
import shlex

def myrun(cmd,pname):
    """from http://blog.kagesenshi.org/2008/02/teeing-python-subprocesspopen-output.html
    """
    p = subprocess.Popen(cmd, shell=True, cwd=pname, 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.STDOUT, 
                         universal_newlines=True)
    stdout = []
    while True:
        line = p.stdout.readline()
        stdout.append(line)
        print(line, end='')
        if line == '' and p.poll() != None:
            break
    return ''.join(stdout)

# pname = 'C:/Users/Lab User/OneDrive/Documents/OpenSees/thermalExamples/Ex-3_Pinned beam with thermal gradient/'
# fname = 'pinned.tcl'
# cmd = 'openSees_Eurocode ' + fname

# myrun(cmd)




