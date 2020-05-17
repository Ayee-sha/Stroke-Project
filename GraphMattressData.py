# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:13:11 2020

@author: Aakash
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import pylab
import cv2
import argparse

'''Global Constants'''
sensor_width = 1.59
sensor_height = 1.59
x = np.arange(0, 187.62, sensor_width)
y = np.arange(0, 76.32, sensor_height)
X, Y = np.meshgrid(x, y)
List_of_All_Files = ['ID#209_PS0008R4S0029_20151102_170707_coloured',
                     'ID#210_PS0008R4S0035_20151102_180313_coloured',
                     'ID#256_PS0008R4S0035_20160117_180915_coloured',
                     'ID#307_PS0008R4S0029_20160310_180802_coloured',
                     'ID#325_PS0008R4S0039_20160429_174754_coloured',
                     'ID#327_PS0008R4S0043_20160503_185516_coloured',
                     'ID#332_PS0008R4S031_20020101_070043_coloured',
                     'ID#346_PS0008R4S0029_20160529_023237_coloured',
                     'ID#348_PS0008R4S0029_20160530_185807_coloured',
                     'ID#349_PS0008R4S031_20020101_070227_coloured',
                     'ID#433_PS0008R4S0043_20161129_171447_coloured',
                     'ID#439_PS0008R4S0043_20161209_202808_coloured',
                     'ID#441_PS0008R4S0043_20170103_225426_coloured',
                     'ID#470_PS0008R4S0042_20170316_180537_coloured',
                     'ID#478_PS0008R4S0042_20170329_214302_coloured',
                     'ID#490_PS0008R4S0040_20170513_220753_coloured',
                     'ID#497_PS0008R4S0039_20170529_174502_coloured',
                     'ID#584_PS0008R4S0030_20171025_163438_coloured',
                     'ID#615_PS0008R4S0030_20171208_003849_coloured',
                     'ID#616_PS0008R4S031_20020101_070151_coloured']
ID_List = ['ID209', 'ID210', 'ID256', 'ID307', 'Id325', 'ID327', 'ID332',
           'ID346', 'ID348', 'ID349', 'ID433', 'ID439', 'ID441', 'ID470',
           'ID478', 'ID490', 'ID497', 'ID584', 'ID615', 'ID616']

ext = '.mp4'

''' Acquire Directory Name '''


def Get_Path_Name(index, List_of_All_Files, dir_name):
    temp = str(List_of_All_Files[index] + '.npy')
    PathName = os.path.join(dir_name, temp)
    return PathName


def Get_Dir_Name():
    input_dir_name = input("Enter the address of the input directory \n")
    print("The directory name you have entered is: ", input_dir_name, "\n")
    input_dir_name = os.path.normpath(input_dir_name)
    output_dir_name = input("Enter the address of the output directory \n")
    print("The directory name you have entered is: ", output_dir_name, "\n")
    output_dir_name = os.path.normpath(output_dir_name)
    return input_dir_name, output_dir_name


''' Load Numpy Array '''


def Get_Array(PathName):
    data = np.load(PathName)
    Number_of_Frames = np.size(data, 2)
    return data, Number_of_Frames


''' Graph Numpy Array '''


def Graph_Array(X, Y, Pressure, ImageName, index):
    for frame in range(index):
        Z = np.zeros((48, 118))
        for i in range(48):
            for j in range(118):
                Z[i][j] = Pressure[i][j][frame]
        fig, ax = plt.subplots()
        CS = ax.contourf(X, Y, Z)
        CB = plt.colorbar(CS, shrink=0.8, extend='both', orientation=
                          'horizontal')
        CB.set_label('Pressure (mmHg)', rotation=0, labelpad=+16)
        plt.title('Frame ' + str(frame))
        plt.xlabel('Length of Bed (cm)')
        plt.ylabel('Width of Bed (cm)')
        #plt.show()  #temporary line for debugging
        plt.savefig(ImageName[frame])
        plt.clf()
        plt.close('all')
        pylab.close(fig)
    return CS


''' Name File Loop '''


def Graph_Name(List_of_All_Files, Number_of_Frames, index, output_directory):
    ImageName = []
    for i in range(Number_of_Frames):
        if i < 10:
            temp_Name = str(List_of_All_Files[index] + '0000' + str(i) + '.png')
        elif i < 100:
            temp_Name = str(List_of_All_Files[index] + '000' + str(i) + '.png')
        elif i < 1000:
            temp_Name = str(List_of_All_Files[index] + '00' + str(i) + '.png')
        else:
            temp_Name = str(List_of_All_Files[index] + '0' + str(i) + '.png')
        temp_Name = os.path.join(output_directory, temp_Name)
        ImageName.append(temp_Name)
    return ImageName


''' Make New Directory for Graphs'''


def Make_Out_Dir(Output_DirName, ID_List, index):
    temp = os.path.join(Output_DirName, ID_List[index])
    if not os.path.exists(temp):
        os.mkdir(temp)
    return temp


''' Make Video From a folder'''
def Make_Video(dir_path, output):
    images = []
    for f in os.listdir(dir_path):
        if f.endswith('.png'):
            images.append(f)
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    cv2.imshow('video',frame)
    height, width, channels = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    for image in images:
    
        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)
    
        out.write(frame) # Write out frame to video
    
        cv2.imshow('video',frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
    return None



def create_video(IMG_PATH, VID_PATH):
    images = [img for img in os.listdir(IMG_PATH) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(IMG_PATH, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(VID_PATH, 0, 24, (width, height))

    print ("Creating the video")
    for image in images:
        video.write(cv2.imread(os.path.join(IMG_PATH, image)))

    cv2.destroyAllWindows()
    video.release()
    print ("Video Created")
    return None


''' Main Code '''


input_directory, output_directory = Get_Dir_Name()
for i in range(20):
    image_directory = Make_Out_Dir(output_directory, ID_List, i)
    PathName = Get_Path_Name(i, List_of_All_Files, input_directory)
    Pressure, Number_of_Frames = Get_Array(PathName)
    ImageName = Graph_Name(List_of_All_Files, Number_of_Frames, i, image_directory)
    graph = Graph_Array(X, Y, Pressure, ImageName, Number_of_Frames)
    FileName = ID_List[i] + ext
    VID_PATH = os.path.join(image_directory, FileName)
    create_video(image_directory, VID_PATH)