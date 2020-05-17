# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 08:00:31 2020

@author: BT_Lab
"""

import os
import json
import xlrd
from datetime import time


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
data = {}
'''#PatientID = '''
data['Patient'] = []


def Get_Path_Name(index, List_of_All_Files, in_dir_name, out_dir_name):
    input_ext = '.xlsx'
    output_ext = '.txt'
    temp_in = str(List_of_All_Files[index] + input_ext)
    temp_out = str(List_of_All_Files[index] + output_ext)
    PathName = os.path.join(in_dir_name, temp_in)
    outfile = os.path.join(out_dir_name, temp_out)
    return PathName, outfile


def Read_Excel_To_Array(FileName):
    wb = xlrd.open_workbook(FileName)
    sheet = wb.sheet_by_index(1)
    Data_Frame_Number = int((sheet.nrows - 3) / 14) - 1
    data = {}
    data['Patient'] = []
    for i in range(Data_Frame_Number):
        # use the commented code below to convert time when reading
        # raw_time = sheet.cell_value((14 * i + 4), 1)
        # raw_time = int(raw_time*24*3600)
        # my_time = time(raw_time//3600,(raw_time%3600)//60,raw_time%60)
        data['Patient'].append({
            'Frame': sheet.cell_value((14 * i + 2), 1),
            'Date': sheet.cell_value((14 * i + 3), 1),
            'Time': sheet.cell_value((14 * i + 4), 1),
            'Sensors': sheet.cell_value((14 * i + 5), 1),
            'Rows': sheet.cell_value((14 * i + 6), 1),
            'Columns': sheet.cell_value((14 * i + 7), 1),
            'COP Row': sheet.cell_value((14 * i + 8), 1),
            'COP Column': sheet.cell_value((14 * i + 9), 1),
            'Sensel Width': sheet.cell_value((14 * i + 10), 1),
            'Sensel Height': sheet.cell_value((14 * i + 11), 1),
            'Avg Pressure': sheet.cell_value((14 * i + 12), 1),
            'Peak Pressure': sheet.cell_value((14 * i + 13), 1),
            'Contact Area': sheet.cell_value((14 * i + 14), 1),
            })
    wb.release_resources()
    del wb
    return data


def Get_Dir_Name():
    input_dir_name = input("Enter the address of the input directory \n")
    print("The directory name you have entered is: ", input_dir_name, "\n")
    input_dir_name = os.path.normpath(input_dir_name)
    output_dir_name = input("Enter the address of the output directory \n")
    print("The output directory name you have entered is: ", output_dir_name,
          "\n")
    output_dir_name = os.path.normpath(output_dir_name)
    return input_dir_name, output_dir_name


def Write_Array_as_Json(FileName, data):
    with open(outfile, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


input_Directory, output_Directory = Get_Dir_Name()
for i in range(20):
    PathName, outfile = Get_Path_Name(i, List_of_All_Files, input_Directory,
                                      output_Directory)
    input_data = Read_Excel_To_Array(PathName)
    Write_Array_as_Json(outfile, input_data)
