#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:50:35 2020

@author: Benjamin Schumacher

Example file for correction of sonic anemometer data


"""




from sonic_func import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import os


wd_path = os.getcwd()
data_path = os.path.join(wd_path, "data")
example_file = os.path.join(data_path, "example.csv")


irg2 = pd.read_csv(example_file, header=[0,1], na_values='NAN')

#
#irg2.to_numpy()

timestamp = irg2["TIMESTAMP"].values[3800:]
u = irg2["Ux"].values[3800:]
v = irg2["Uy"].values[3800:]
w = irg2["Uz"].values[3800:]

timestamp_pf, u1_pf, v1_pf, w1_pf = planar_fit(u, v, w, sub_size = 10,timestamp=timestamp)

timestamp_rot3, u1_rot3, v1_rot3, w1_rot3 = triple_rot(u, v, w, sub_size = 10,timestamp=timestamp)



example_pf = pd.DataFrame({'TIMESTAMP': timestamp_pf.flatten(), 'Ux': u1_pf, 'Uy': v1_pf, 'Uz': w1_pf})

example_rot3 = pd.DataFrame({'TIMESTAMP': timestamp_rot3.flatten(), 'Ux': u1_rot3, 'Uy': v1_rot3, 'Uz': w1_rot3})


example_pf.to_csv(data_path+"/example_result_planarfit.csv")

example_rot3.to_csv(data_path+"/example_result_triplerotation.csv")