# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import os
import json

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
Numpy array is in the form of [row][col][frame]
Have to rearrange it so the numpy array is in the form [frame][row][col]
Find the min/max indice of the non zero values
This can be done by using the np.nonzero function which returns [rows][cols]
Then use the np.max and np.min functions to find the smallest value 
ALso have to figure out how to calculate the COP for each column of the frame
"""

'''
Define the object called patient
Will have 3D array of all the pressure data
Will have array of json object corresponding to frame number
'''



'''
Define all the constants
'''


'''
Define all the functions
'''

'''
The GetSymmetry function accepts a 2D array as an argument and returns 2
tuples of x,y coordinates
'''


def GetSymmetry(my_arr):
    x = np.nonzero(my_arr)
    max_index = np.where(np.max(x))
    min_index = np.where(np.min(x))
    return max_index, min_index


'''
The GetCOP function accepts a 3D array as an argument and returns two 3D arrays
which contain the longitudnal and lateral center of pressure locations
'''


def GetCOP(my_arr):
    width = np.size(my_arr, 1)
    length = np.size(my_arr, 2)
    x = np.arange(length)
    y = np.arange(width)
    X, Y = np.meshgrid(x, y)
    COP_long = np.einsum('ij,kij->kij', X, my_arr)
    COP_lat = np.einsum('ij,kij->kij', Y, my_arr)
    P_lat = np.sum(my_arr, axis=1)
    P_long = np.sum(my_arr, axis=2)
    COP_long = COP_long / P_long[:, :, None]
    COP_lat = COP_lat / P_lat[:, None, :]
    np.nan_to_num(COP_long)
    np.nan_to_num(COP_lat)
    return COP_long, COP_lat


'''
The GetAlpha function accepts a 3D array as an argument and returns a vecor of
floats. These floats are the coefficients of second degree polynomial that is
generated to estimate the center of pressure in the lateral direction for all
longitudnal points.
'''


def GetAlpha(my_arr):
    arrs = np.size(my_arr, 0)
    coeff = np.zeros(arrs)
    for array in range(arrs):
        x = np.arange(np.size(my_arr, 1))
        coeff[array] = np.polyfit(x, my_arr, 2)
    return coeff


'''
The GetTotalPressure function accepts a 3D array as an argument and returns
a vector of the 2D slice of each frame
'''


def GetTotalPressure(my_arr):
    vec_sum = np.sum(np.sum(my_arr, axis=1), axis=1)
    return vec_sum


'''
The GetPosture function accepts a 3D array as an argument and returns a vector
of booleans that is TRUE if the patient is in supine or prone positions and
returns a FALSE if patient is sideways
'''


def GetPosture(my_arr):
    posture = np.zeros(np.size(my_arr, 0), dtype=bool)
    COP_long, COP_lat = GetCOP(my_arr)
    COP_lat = np.sum(COP_lat, axis=1)
    coeff = GetAlpha(COP_lat)
    for i in range(np.size(coeff)):
        if (coeff[i] > 0.3) or (coeff[i] < -0.3):
            posture[i] = False
        else:
            posture[i] = True
    return posture


def GetDirName(name):
    Num_arr_Dir = input('Enter the directory name for the ', name,
                        ' directory: \n')
    Num_arr_Dir = os.path.normpath(name)
    return Num_arr_Dir


def PathExists(PathName, name):
    exists = os.path.exists(PathName)
    if not(exists):
        print('The location you have entered does not exist please enter a'
              'valid location')
        Num_arr_Dir = GetDirName(name)
    else:
        print('The file you have entered is: \n', PathName)
    return None


def GetPathNames():
    Num_arr_Dir = GetDirName('numpy')
    PathExists(Num_arr_Dir, 'numpy')
    Output_dir = GetDirName('output')
    PathExists(Num_arr_Dir, 'output')
    return Num_arr_Dir, Output_dir


def LoadArrays(num_arr_dir):
    all_arrs = []
    for file in os.listdir(num_arr_dir):
        filename = os.fsdecode(file)
        if filename.endswith('.npy'):
            file_path = os.path.join(num_arr_dir, filename)
            all_arrs.append(file_path)
            continue
        else:
            continue
    return all_arrs


'''
Function to load in all the meta data stored as json data
'''


def GetMetaData(PathName):
    


def main():
    num_arr_dir, output_dir = GetPathNames()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    