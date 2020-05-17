# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 23:04:33 2019

@author: Aakash
"""

import xlrd
import numpy as np
import os

def Get_Directory_Name():
    input_dir_name = input("Enter the address of the input directory \n")
    print("The directory name you have entered is: ", input_dir_name, "\n")
    input_dir_name = os.path.normpath(input_dir_name)
    
    output_dir_name = input("Enter the address of the directory where you would like you files to be saved \n")
    output_dir_name = os.path.normpath(output_dir_name)
    
    FileName = input("Enter the name of the file \n")
    PathName = os.path.join(input_dir_name, FileName)
    FileName = str(FileName[:-3])
    
    return input_dir_name, output_dir_name, PathName, FileName


def Read_Excel_To_Numpy(FileName):
    wb = xlrd.open_workbook(FileName)
    sheet = wb.sheet_by_index(2)
    Data_Frame_Number = int(sheet.nrows / 48)
    num_arr = np.zeros((48, 118, Data_Frame_Number))

    frame = 0
    row = 0
    for i in range(sheet.nrows): 
        if i < 48:
            row = i 
        else:
            row = i % 48
        if i != 0 and (i % 48 == 0):
            frame += 1
        for col in range(118):
            num_arr[row][col][frame] = sheet.cell_value(i, col)
    
    wb.release_resources()
    del wb
    return num_arr


def Get_Path_Name(index, List_of_All_Files, dir_name):
    PathName = os.path.join(dir_name, List_of_All_Files[index])
    return PathName
    
List_of_All_Files = ['ID#209_PS0008R4S0029_20151102_170707_coloured', 'ID#210_PS0008R4S0035_20151102_180313_coloured',
                     'ID#256_PS0008R4S0035_20160117_180915_coloured', 'ID#307_PS0008R4S0029_20160310_180802_coloured',
                     'ID#325_PS0008R4S0039_20160429_174754_coloured', 'ID#327_PS0008R4S0043_20160503_185516_coloured',
                     'ID#332_PS0008R4S031_20020101_070043_coloured', 'ID#346_PS0008R4S0029_20160529_023237_coloured',
                     'ID#348_PS0008R4S0029_20160530_185807_coloured', 'ID#349_PS0008R4S031_20020101_070227_coloured',
                     'ID#433_PS0008R4S0043_20161129_171447_coloured', 'ID#439_PS0008R4S0043_20161209_202808_coloured',
                     'ID#441_PS0008R4S0043_20170103_225426_coloured', 'ID#470_PS0008R4S0042_20170316_180537_coloured',
                     'ID#478_PS0008R4S0042_20170329_214302_coloured', 'ID#490_PS0008R4S0040_20170513_220753_coloured',
                     'ID#497_PS0008R4S0039_20170529_174502_coloured', 'ID#584_PS0008R4S0030_20171025_163438_coloured',
                     'ID#615_PS0008R4S0030_20171208_003849_coloured', 'ID#616_PS0008R4S031_20020101_070151_coloured']
                     
output_dir_name = input("Enter the address of the directory where you would like you files to be saved \n")
output_dir_name = os.path.normpath(output_dir_name)
 
for i in range(20):
    List_of_All_Files[i] = str(List_of_All_Files[i] + '.xlsx')
    
for j in range(20):
    PathName = Get_Path_Name(j, List_of_All_Files, output_dir_name)
    input_data = Read_Excel_To_Numpy(PathName)
    np.save(List_of_All_Files[j], input_data)
    
#input_dir_name, output_dir_name, PathName, FileName = Get_Directory_Name()
#input_data = Read_Excel_To_Numpy(PathName)


#np.save(FileName, input_data)