# Stroke-Project

The repo contains files for analyzing pressure mattress data that is provided by the Ward of the 21st Century (W21C) from the Foothills Medical Center in Calgary. 

There project involves multiple steps:

## Acquire Data

The data is in the format of .xlsx documents. The original spreadsheets come with just one sheet. This has all the data for the patient. The data contains all the meta data followed by the raw pressure readings. The first step is to separate the meta data and the pressure arrays. This is done by putting a filter on the entire sheet. In the filter in column A first choose only the non-numerical values. These are:

1. Frame
2. Date
3. Time 
4. Sensel
5. Rows
6. Columns
7. COP Row
8. COP Column
9. Sensel Width (cm)
10. Sensel Height (cm)
11. Average Pressure (mmHg)
12. Peak Pressure
13. Contact Area
14. Sensel

Once the filter has been put in, copy the meta data and paste it in a new sheet within the same file. Next go to the first sheet and in the filter in column A select only numerical values. This will leave only the raw pressure data. Copy all the raw pressure values into a third sheet in the same file.

## Saving  Data

The ReadExcelRowbyRow.py file contains the basic code structure for opening the the excel file and extracting the raw pressure values and saving them as numpy arrays in a specified directory. The raw pressure data arrays have to be located in the third sheet. 

TODO: The file needs to be changed such that it accepts any excel file name not the ones that are listed within the file. User should have control over what sheet the data should be extracted from. 

MetaData.py extracts meta data from any excel file from the second sheet and extract it and store it as JSON objects. The list of JSON objects is stored as a TXT document in a specified directory.

TODO: The user needs to be able to choose what sheet number they want the data to be from

### Graphing Data

The GraphMattressData.py file uses the pre-saved numpy arrays to create contour plots of the pressure distribution and stitches the plots in a animation. 

TODO: The color bar for the pressure distribution is dynamic per frame, leading to subpar animation. A static value for the color bar needs to be setup. Instead of saving PNG images an animation needs to be setup instead that consumes less memory and storage. The graphs are inverted along the width of the bed; i.e., the left most side of the bed s now the right most side. This needs to be changed so left right correspond to each other when graphing. This is not the case with the numpy arrays. 

## Analyze Data

### Feature Extraction

COP.py and AnalyzeMatData.py are two files attempting to start feature extraction. They should be used as pseudo code for feature extraction.

### PCA

Principle component analysis serves to reduce the dimensionality of data to speed up calculation during classification. The PCA.py file performs PCA on a numpy array. The number of PCA components can be chosen. 

TODO: Determine the best value of PCA components without sacrificing accuracy of the classification system.

## Posture Classification

### kNN

The first part of the process is to clean the data so only frames with patients in supinated posture are considered. One method to accomplish this is by training a k-nearest neighbor algorithm to identify and classify pronated posture frames. The algorithm uses manually labelled data from all patients and combines into one array. This array is then split into a training set and testing set. The training set goes through a k-fold cross validation. The parameters are then tested on the testing set to evaluate the accuracy of the model. The model will be re-tested with PCA performed on the labelled data 

### SVM

In addition to kNN classification an SVM classification will also be used