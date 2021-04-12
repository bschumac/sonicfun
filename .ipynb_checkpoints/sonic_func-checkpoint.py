#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:30:16 2020

@author: Benjamin Schumacher

These  intend to implement the standard functions dealing with measured wind speed data from sonic anemometers.
This is majorly based on:

    Tilt-correction (Planar Fit and triple rotation):
        [1] Wilczak, J. M., Oncley, S. P., & Stage, S. A. (2001). Sonic anemometer tilt correction algorithms.
        Boundary-Layer Meteorology, 99(1), 127-150.
        ... additionally inspired by the MATLAB version to be found on the MATLAB file exchange:
        https://www.mathworks.com/matlabcentral/fileexchange/63655-sonic-anemometer-tilt-correction-algorithm
    
    Friction Velocity, Roughness Length and displacement height estimation:
        [2] Stull R. (199X): Boundary Layer Meteorology
    
        ... additionally inspired by the Earth observation laboratory from UCAR:
            https://www.eol.ucar.edu/content/calculation-roughness-length-and-displacement-height

Overview of Functions:

    remove_outliers1D:
        simple outlier removal based on the mean of the data and its standard deviation
        
    interpolate_nan1D:
        interpolates the nans in the data with a linear interpolation
        
    wrapper_outl_interpolation:
        wrapper function to remove outliers and interpolate them directly

    planar_fit: 
        Sonic Anemometer tilt correction algorithm using the planar fit method see [1]
    
    triple_rot:
        Sonic Anemometer tilt correction algorithm using the triple roatation method see [1]
    
    friction_velo:
        Friction velocity calculation from 1 available sonic anemometer see [2]
    
    displacement_height:
        Displacement height calculation from 2 available sonic anemometers see [2]
    
    roughness_len:
        Roughness length calculation from 2 available sonic anemometers see [2]

"""


import numpy as np
import copy


def remove_outliers1D(array, sigma=5):
    """
    Removes outliers in a 1D array and replaces them with NAN.
    
    Input:
        array: 1D numpy array
            u, v or w velocity
        
        sigma: int
            number of standard deviations which are considered to be "usual"  
    
    Output:
        outarr: 1D numpy array
            
   
    """
    outarr = copy.copy(array)
    arr_mean = np.nanmean(outarr)
    arr_std = np.nanstd(outarr)
    outarr[np.abs(outarr)>arr_mean+sigma*arr_std] = np.nan
    
    return(outarr)    


def interpolate_nan1D(array):
    """
    Interpolates NAN with a linear interpolation.
    
    Input:
        array: 1D numpy array
            u, v or w velocity
            
    Output:
        outarr: 1D numpy array
            
   
    """
    outarr = copy.copy(array)
    ok = ~np.isnan(outarr)
    xp = ok.ravel().nonzero()[0]
    fp = array[~np.isnan(outarr)]
    x  = np.isnan(outarr).ravel().nonzero()[0]

    outarr[np.isnan(outarr)] = np.interp(x, xp, fp)
    return(outarr)


def wrapper_outl_interpolation(array):
    """
    Wrapper function for outlier removal and interpolation
    
    Input:
        array: 1D numpy array
            u, v or w velocity
            
    Output:
        arr_int: 1D numpy array
            
   
    """
    arr_rem = remove_outliers1D(array)
    arr_int = interpolate_nan1D(arr_rem)
    return(arr_int.flatten())




def reshape_data2D(arr, size_y, size_x):
    """
    Reshapes array from 1D structure to 2D structure with sizes provided
    
    Input:
        arr: 1D numpy array
            u, v or w velocity
        
        size_y: int
            size in vertical direction
        
        size_x: int
            size in horizontal direction
    
    Output:
        out_arr: 2D numpy array
            
   
    """
    
    if len(arr)%size_y != 0:
        print("Shortening array to even number by "+str(len(arr)%size_y)+" measurements!")
        arr = arr[0:len(arr)-len(arr)%size_y]
    out_arr = np.swapaxes(np.reshape(arr,(size_y,size_x)),0,1)
    
    return(out_arr)



def findB(meanU, meanV, meanW, M):
    """
    See [1] for more insight. 
    
    """
    
    su = np.nansum(meanU)
    sv = np.nansum(meanV)
    sw = np.nansum(meanW)
    suv = meanU.dot(meanV.flatten())
    suw = meanU.dot(meanW.flatten())
    svw = meanV.dot(meanW.flatten())
    su2 = meanU.dot(meanU.flatten())
    sv2 = meanV.dot(meanV.flatten())
    H = np.array([(M,su,sv),(su, su2, suv),(sv, suv, sv2)])
    g = np.array([sw,suw,svw])
    x = np.linalg.solve(H,g)
    r1=g-(H.dot(x))
    b0 = x[0]
    b1 = x[1]
    b2 = x[2]
    return(b0, b1, b2, r1)


def planar_fit(u, v, w, sub_size = 10, **kwargs):
    """
    Sonic Anemometer tilt correction algorithm using the planar fit method see [1]
    
    Input data:
        u: 1D numpy array
            velocity as measured in u direction
        v: 1D numpy array
            velocity as measured in v direction
        w: 1D numpy array
            velocity as measured in w direction
        
        sub_size: int
            reshaping size used for averaging, default: 10 measurements 
            
        **kwargs: optional 1D array
            will look for timestamp otherwise creates an index as timestamp
            
    Output data:
        u_fit: 1D numpy array
            tilt corrected u-velocity
        
        v_fit: 1D numpy array
            tilt corrected v-velocity
        
        w_fit: 1D numpy array
            tilt corrected w-velocity
            
        timestamp: 1D numpy array
            (shortened) timestamp when one is provided, otherwise an index number
            
    
    """
    
    
    # Firstly finding the optionally provided timestamp
    try:
        timestamp = kwargs.get("timestamp")
    except:
        pass
    try:
        Ts = kwargs.get("Ts")
    except:
        pass
    try:
        CO2 = kwargs.get("CO2")
    except:
        pass
    try:
        H2O = kwargs.get("H2O")
    except:
        pass
    
    if timestamp is None:
        print("No timestamp provided! Creating artificial timestamp...")
        timestamp = np.arange(0,len(u))
    elif len(timestamp) != len(u):
        print("Timestamp does not match measurements! Using artificial timestamp...")
        timestamp = np.arange(0,len(u))
    
    
    round_len = int(len(u)/sub_size)
    
    # Cutting the timestamp to the length of the data
    
    if len(timestamp)%round_len != 0:
        timestamp = timestamp[0:len(timestamp)-len(timestamp)%round_len]
        try:
            Ts = Ts[0:len(Ts)-len(Ts)%round_len]
        except:
            pass
        try:
            CO2 = Ts[0:len(CO2)-len(CO2)%round_len]
        except:
            pass
        try:
            H2O = H2O[0:len(H2O)-len(H2O)%round_len]
        except:
            pass
        
    
    
    # reshaping data to optimize calculation
    
    u1 = reshape_data2D(u, round_len, sub_size)
    v1 = reshape_data2D(v, round_len, sub_size)
    w1 = reshape_data2D(w, round_len, sub_size)
    
    
    
    u_fit = np.empty(u1.shape)
    v_fit = np.empty(v1.shape)
    w_fit = np.empty(w1.shape)
    
    # calculation starts here
    
    meanU = np.nanmean(u1, 1)
    meanV = np.nanmean(v1, 1)
    meanW = np.nanmean(w1, 1)
    M = meanU.size
    b0, b1, b2, r1 = findB(meanU, meanV, meanW, M)
    
    Deno = np.sqrt(1+b1**2+b2**2)
    p31 = -b1/Deno
    p32 = -b2/Deno
    p33 = 1.00/Deno
    
    cosGamma = p33/np.sqrt(p32**2+p33**2)
    sinGamma = -p32/np.sqrt(p32**2+p33**2)
    cosBeta = np.sqrt(p32**2+p33**2);
    sinBeta = p31
    
    R2 = np.array([(1,0,0), (0,cosGamma, -sinGamma), (0, sinGamma, cosGamma)])
    R3 = np.array([(cosBeta,0,sinBeta),(0,1,0),(-sinBeta,0,cosBeta)])
    R2t = R2.conj().transpose()
    R3t = R3.conj().transpose()
    
    UVW = np.array([meanU,meanV,meanW]).conj()
    
    A0 = np.dot(np.dot(R3t,R2t),UVW) 
    Alpha = np.arctan2(A0[1,:],A0[0,:])
    
    for i in range(0,M):
        R1 = np.array([(np.cos(Alpha[i]),-np.sin(Alpha[i]),0),(np.sin(Alpha[i]),np.cos(Alpha[i]),0),(0,0,1)])
        R1t = R1.conj().transpose()
        
        R3R2_t = np.dot(R3t,R2t)
        act_UVW = np.array([u1[i,:],v1[i,:],w1[i,:]-b0])
        A1 = np.dot(R1t,np.dot(R3R2_t,act_UVW))
        
        u_fit[i,:] = A1[0,:]
        v_fit[i,:] = A1[1,:]
        w_fit[i,:] = A1[2,:]
    
    
    
    u_fit = np.swapaxes(u_fit,0,1).flatten()
    v_fit = np.swapaxes(v_fit,0,1).flatten()
    w_fit = np.swapaxes(w_fit,0,1).flatten()
    
    if 'Ts' in locals() and 'H2O' in locals() and 'CO2' in locals():
        return(timestamp, u_fit, v_fit, w_fit, Ts, CO2, H2O)
    elif 'Ts' in locals() and 'H2O' in locals():
        return(timestamp, u_fit, v_fit, w_fit, Ts, H2O)
    elif 'Ts' in locals() and 'CO2' in locals():
        return(timestamp, u_fit, v_fit, w_fit, Ts, CO2)
    elif 'H2O' in locals() and 'CO2' in locals():
        return(timestamp, u_fit, v_fit, w_fit, H2O, CO2)
    elif 'Ts' in locals():
        return(timestamp, u_fit, v_fit, w_fit, Ts)
    elif 'CO2' in locals():
        return(timestamp, u_fit, v_fit, w_fit, CO2)
    elif 'H2O' in locals():
        return(timestamp, u_fit, v_fit, w_fit, H2O)
    else:
        return(timestamp, u_fit, v_fit, w_fit)


def triple_rot (u, v, w, sub_size = 10, **kwargs):
    """
    Sonic Anemometer tilt correction algorithm using the triple rotation method see [1]
    
    Input data:
        u: 1D numpy array
            velocity as measured in u direction
        v: 1D numpy array
            velocity as measured in v direction
        w: 1D numpy array
            velocity as measured in w direction
        
        sub_size: int
            reshaping size used for averaging, default: 10 measurements 
            
        **kwargs: optional 1D array
            will look for timestamp otherwise creates an index as timestamp
            
    Output data:
        u_fit: 1D numpy array
            tilt corrected u-velocity
        
        v_fit: 1D numpy array
            tilt corrected v-velocity
        
        w_fit: 1D numpy array
            tilt corrected w-velocity
            
        timestamp: 1D numpy array
            (shortened) timestamp when one is provided, otherwise an index number
            
    
    """
    
    
    # Firstly finding the optionally provided timestamp
    
    timestamp = kwargs.get("timestamp")
    Ts = kwargs.get("Ts")
    CO2 = kwargs.get("CO2")
    H2O = kwargs.get("H2O")
    
    if timestamp is None:
        print("No timestamp provided! Creating artificial timestamp...")
        timestamp = np.arange(0,len(u))
    elif len(timestamp) != len(u):
        print("Timestamp does not match measurements! Using artificial timestamp...")
        timestamp = np.arange(0,len(u))
    
    
    round_len = int(len(u)/sub_size)
    
    # Cutting the timestamp to the length of the data
    
    if len(timestamp)%round_len != 0:
        timestamp = timestamp[0:len(timestamp)-len(timestamp)%round_len]
        try:
            Ts = Ts[0:len(Ts)-len(Ts)%round_len]
        except:
            pass
        try:
            CO2 = Ts[0:len(CO2)-len(CO2)%round_len]
        except:
            pass
        try:
            H2O = H2O[0:len(H2O)-len(H2O)%round_len]
        except:
            pass
    
    # reshaping data to optimize calculation
    
    u1 = reshape_data2D(u, round_len, sub_size)
    v1 = reshape_data2D(v, round_len, sub_size)
    w1 = reshape_data2D(w, round_len, sub_size)
    
    
    
    u_fit = np.zeros(u1.shape)
    v_fit = np.zeros(v1.shape)
    w_fit = np.zeros(w1.shape)
    
    
    
    
    for i in range(0,sub_size):
        # First rotation around z axis
        A01 = np.array((u1[i,:],v1[i,:]))
        R1 = np.arctan2(np.nanmean(A01[1,:]),np.nanmean(A01[0,:]))
        R1 = np.array([(np.cos(R1), np.sin(R1)),(-np.sin(R1),np.cos(R1))])
        A1 = np.dot(R1,A01)
        u2 = A1[0,:]
        v2 = A1[1,:]
    
        # Second rotation around y axis
        A02 = np.array((u2,w1[i,:]))
        R2 = np.arctan2(np.nanmean(A02[1,:]),np.nanmean(A02[0,:]))
        RotY = np.array([(np.cos(R2), np.sin(R2)),(-np.sin(R2),np.cos(R2))])
        A2 = np.dot(RotY,A02)
        u_fit[i,:] = A2[0,:]
        w2 = A2[1,:]
        
        # Third rotation around x axis
        A03 = np.array((v2,w2))
        covVW = np.nanmean(v2*w2)
        diffVW = np.nanvar(v2)-np.nanvar(w2)
        R3 = 0.5*np.arctan2(2*covVW,diffVW)
        RotX = np.array([(np.cos(R3), np.sin(R3)),(-np.sin(R3),np.cos(R3))])
        A3 = np.dot(RotX,A03)
        v_fit[i,:]=A3[0,:]
        w_fit[i,:]=A3[1,:]
    
    u_fit = np.swapaxes(u_fit,0,1).flatten()
    v_fit = np.swapaxes(v_fit,0,1).flatten()
    w_fit = np.swapaxes(w_fit,0,1).flatten()
    
    
    
    if 'Ts' in locals() and 'H2O' in locals() and 'CO2' in locals():
        return(timestamp, u_fit, v_fit, w_fit, Ts, CO2, H2O)
    elif 'Ts' in locals() and 'H2O' in locals():
        return(timestamp, u_fit, v_fit, w_fit, Ts, H2O)
    elif 'Ts' in locals() and 'CO2' in locals():
        return(timestamp, u_fit, v_fit, w_fit, Ts, CO2)
    elif 'H2O' in locals() and 'CO2' in locals():
        return(timestamp, u_fit, v_fit, w_fit, H2O, CO2)
    elif 'Ts' in locals():
        return(timestamp, u_fit, v_fit, w_fit, Ts)
    elif 'CO2' in locals():
        return(timestamp, u_fit, v_fit, w_fit, CO2)
    elif 'H2O' in locals():
        return(timestamp, u_fit, v_fit, w_fit, H2O)

    else:
        return(timestamp, u_fit, v_fit, w_fit)
        
    


