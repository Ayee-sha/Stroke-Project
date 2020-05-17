# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:50:30 2020

@author: BT_Lab
"""

"""
Analyze mattress data few things need to be done:

    1. Load the numpy arrays containing the pressure values for each frome
    2. For each frame calculate the total sum of the pressure, if the pressure
    is 0 make it 1
    3. Save sums in a vector with lenght equal to the number of frames minus 1
    4. Normalize each frame by dividing all elements in the frame by the
    element in the sum vector corresponding to the same frame
    5. Within the normalized frames find the minimum amd maximum row where a
    non zero pressure is seen
    6. Based on the max and min row index find a central axis to segment the
    data in left and right
        a. Do this by summing the row index from max and min and divinding by 2
    7. Within each half calculate the center of pressure (x, y) coordinates and
    variance in pressure
        a. Sum of all the normalized pressures in each half
        b. Center of pressure is normalized pressure at position (row, column)
        multiplied by the position
        c. This has to be done for each axis separately (lateral and
                                                         longitudnal)
        d. Calculate lateral variance, need average in the corresponding
        direction
    8. Determine the amount of change in center of pressure over time in both
    sides
    9. Determine the amount of change in variance in pressure over time in both
    sides
    10. Determine the center of pressure for all time for both sides combined
    11. Determine the varaince in variances
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import pylab

'''
Fucntion to acquire all the relevant directories
'''


def GetPathNames():
    """
    Gets input and output directory location from user input

    Returns
    -------
    Num_arr_Dir : str
        Location of the directory where the numpy arrays containing pressure
        data are located.
    Output_dir : str
        Location of the directory where the ouput should be saved to.

    """
    Num_arr_Dir = input('Enter the directory name for the numpy arrays \n')
    Num_arr_Dir = os.path.normpath(Num_arr_Dir)
    Output_dir = input('Enter the directory name for the output directory \n')
    Output_dir = os.path.normpath(Output_dir)
    return Num_arr_Dir, Output_dir


'''
Load numpy array
'''


def LoadArrays(num_arr_dir):
    """
    Populate an array with the all the files with .npy extension in a directory

    Parameters
    ----------
    num_arr_dir : str
        Directory containing the location where the numpy arrays are stored.

    Returns
    -------
    all_arrs : list
        List of all the files with .npy extension.

    """

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
Calculate the sum for all the frames of a patient
Returns a vector that contains the sum total of each frame for one patient
'''


def PressureTotal(num_arr):
    """
    Calculate the total sum of pressures on each frame

    The function sums all the pressure values for each frame and returns them
    as a vector. If the frame has no patinet then the sum is returned as 1.

    Parameters
    ----------
    num_arr : ndarray
        A 3D array in the form [frame][row][col].

    Returns
    -------
    sum_vec : ndarray
        A 1D vector containing the sum of the pressure for each frame. Length
        is equal to the number of frames.

    """
    sum_vec = num_arr.sum(axis=2).sum(axis=1)
    # Return sum as 1 instead of 0. Makes division possible.
    # Consider changing this to -1 instead
    for i in range(len(sum_vec)):
        if sum_vec[i] == 0:
            sum_vec[i] = 1
        else:
            continue
    return sum_vec


'''
Array normalization element wise
'''


def NormalizePressure(num_arr, sum_vec):
    """
    Normalizes the pressure area so the sum or pressure is equal to 1

    Parameters
    ----------
    num_arr : ndarray
        A 3D array in the form [frame][row][col].
    sum_vec : ndarray
        A 1D vector containing the sum of the pressure for each frame. Length
        is equal to the number of frames.

    Returns
    -------
    norm_arr : ndarray
        A 3D array in the form [frame][row][col]. Contains normalized pressure
        values.

    """

    norm_arr = num_arr / sum_vec[:, None, None]
    return norm_arr


'''
Find the central axis for each frame
'''


def MinRowNonZero(norm_arr):
    """
    Returns the first row index with a non-zero value

    Parameters
    ----------
    norm_arr : ndarray
        A 3D array in the form [frame][row][col]. Contains normalized pressure
        values.

    Returns
    -------
    row : int
        Return int value of the row index.

    """

    for row in range(48):
        for col in range(118):
            if norm_arr[row][col] != 0:
                return row
    # If there are no nonzero values return 0 (min row)
    return 0


def MaxRowNonZero(norm_arr):
    """
    Returns the last row index with a non-zero value

    Parameters
    ----------
    norm_arr : ndarray
        A 3D array in the form [frame][row][col]. Contains normalized pressure
        values.

    Returns
    -------
    row : int
        Returns an int value of row index.

    """

    for row in reversed(range(48)):
        for col in range(118):
            if norm_arr[row][col] != 0:
                return row
    # If there are no nonzero values return 47 (max row)
    return 47


def GetCentralAxis(norm_arr):
    """
    Return the line of symmetry, min, and max row index for each frame.

    The function calculates the line of symmetry or the row number which acts
    like the central axis that divides the patient in half. Does this by
    calculating the average between the max and min nonzero row indices.

    Parameters
    ----------
    norm_arr : ndarray
        A 3D array in the form [frame][row][col]. Contains normalized pressure
        values.

    Returns
    -------
    central_axis : ndarray
        1D array of int values of row indices which serve as the midline for
        the patient for each frame.
    min_row : int
        1D array of int values of the smallest row indicies with non zero
        value.
    max_row : int
        1D array of int values of the largest row indices with non zero
        value.

    """

    central_axis = np.zeros(np.ma.size(norm_arr, axis=2))
    min_row = np.zeros(np.ma.size(norm_arr, axis=2))
    max_row = np.zeros(np.ma.size(norm_arr, axis=2))
    for frame in range(np.ma.size(norm_arr, axis=2)):
        min_row[frame] = MinRowNonZero(norm_arr[:][:][frame])
        max_row[frame] = MaxRowNonZero(norm_arr[:][:][frame])
        central_axis[frame] = int((max_row + min_row) / 2 + 1)
    return central_axis, min_row, max_row


'''
Center of Pressure Calculation
         N
        ____
        \     xi * pi
  COP =  \  -----------
         /      pi
        /____
        i = 0

Need 3 center of pressures:
    1. The overall center of pressure
    2. The center of pressure on the left side
    3. The center of pressure on the right side
'''


def SumPressureDist(norm_arr, start, stop):
    """
    Returns (x,y) coordinates of the of the COP for each frame.

    Center of Pressure is calculated by the prodcut of the pressure and its
    sensor position. The function calculates over all columns, however it
    calculates between specfied rows [start, stop].

    Parameters
    ----------
    norm_arr : ndarray
        A 3D array in the form [frame][row][col]. Contains normalized pressure
        values.
    start : int
        Smallest row index from which the COP should start calculating.
    stop : int
        Largest row index from which the COP should start calculating.

    Returns
    -------
    x_bar : float
        x-location, corresponding to rows in the frame.
    y_bar : float
        y-location corresponding to columns in the frame.

    """

    col = np.ma.size(norm_arr, axis=1)
    x_bar, y_bar = 0
    # Multiply index directly as pressure is normalized
    for i in range(start, stop):
        for j in range(col):
            x_bar += norm_arr[i][j] * i
            y_bar += norm_arr[i][j] * j
    return x_bar, y_bar


def SumPressure(norm_arr, start, stop):
    """
    Calculates total pressure of a pressure array with normalized values.

    Calculates sum of normalized pressure values over all columns but specified
    rows [start, stop].

    Parameters
    ----------
    norm_arr : ndarray
        A 3D array in the form [frame][row][col]. Contains normalized pressure
        values.
    start : int
        Samllest row index, where summmation should begin.
    stop : int
        Largest row index, where summation should end.

    Returns
    -------
    p_tot : float
        Sum total of all pressure values in specified range.

    """

    row = np.ma.size(norm_arr, axis=0)
    col = np.ma.size(norm_arr, axis=1)
    p_tot = 0
    for i in range(row):
        for j in range(col):
            p_tot += norm_arr[i][j]
    return p_tot


def CenterOfPressure(norm_arr, central_axis, minmax):
    """
    Calculate the COP for left, right, and overall pressure frame.

    Calculates the Center of Prssure for the overall frame and the portion of
    frames which correspond to the right and left half of the patient.

    Parameters
    ----------
    norm_arr : ndarray
        A 3D array in the form [frame][row][col]. Contains normalized pressure
        values.
    central_axis : ndarray
        Row vector containing (x,y) coordinates respectively.
    minmax : ndarray
        Row vector containing minimum and maximum row index of non zero value
        in the norm_arr array.

    Returns
    -------
    COP_tot : ndarray
        (x,y) coordinates of the COP of entire frame.
    COP_left : ndarray
        (x,y) coordinates of the COP of left half of patient.
    COP_right : ndarray
        (x,y) coordinates fo the COP of right half of patient.

    """

    # frame = np.ma.size(norm_arr, axis=2)
    COP_tot, COP_left, COP_right = np.zeros_like(minmax)
    lef_sum_presdist, rig_sum_presdist = np.zeros_like(minmax)
    for frame in range(np.ma.size(norm_arr, axis=2)):
        COP_tot[frame] = SumPressureDist(norm_arr[:][:][frame], 0,
                                         np.ma.size(norm_arr, axis=0))
        lef_sum_presdist[:][frame] = SumPressureDist(norm_arr[:][:][frame],
                                                     central_axis[frame],
                                                     minmax[1][frame])
        rig_sum_presdist[:][frame] = SumPressureDist(norm_arr[:][:][frame],
                                                     minmax[0][frame],
                                                     central_axis[frame])
        lef_sum_pres = SumPressure(norm_arr[:][:][frame], central_axis[frame],
                                   minmax[1][frame])
        # Subtaract from 1 becasue the array is normalized
        rig_sum_pres = 1 - lef_sum_pres
    # Divide by the total pressure as the left and right half are no longer
    # normalized arrays despite containing normalized pressure values
    COP_left = np.divide(lef_sum_presdist, lef_sum_pres)
    COP_right = np.divide(rig_sum_presdist, rig_sum_pres)
    return COP_tot, COP_left, COP_right


'''
Calculate vector of average pressure in total frame, left and right frame

'''


'''
Test Code
'''

DirName = os.path.normpath(r'D:\Aakash\Masters\Research\Stroke\Numpy Arrays')
x = r'ID#256_PS0008R4S0035_20160117_180915_coloured.npy'
FileName = os.path.join(DirName, x)
Pressure = np.load(FileName)
frame1 = np.zeros((np.ma.size(Pressure, axis=0), np.ma.size(Pressure, axis=1)))
for i in range(np.ma.size(Pressure, axis=0)):
    for j in range(np.ma.size(Pressure, axis=1)):
        frame1[i][j] = Pressure[i][j][0]

sensor_width = 1.59
sensor_height = 1.59
x = np.arange(0, 187.62, sensor_width)
y = np.arange(0, 76.32, sensor_height)
X, Y = np.meshgrid(x, y)
aspectratio = 48/118

fig, ax = plt.subplots()
CS = ax.contourf(X, Y, frame1)
CB = plt.colorbar(CS, shrink=0.8, extend='both', orientation='horizontal')
CB.set_label('Pressure (mmHg)', rotation=0, labelpad=+16)
plt.title('Frame ' + str(frame))
plt.xlabel('Length of Bed (cm)')
plt.ylabel('Width of Bed (cm)')
plt.axis('equal')
plt.tight_layout()
plt.show()
