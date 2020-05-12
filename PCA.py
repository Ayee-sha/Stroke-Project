import os
import numpy as np
import numpy.linalg as linalg
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def GetInputRootDir():
    """Gets directory pathway from user input"""

    input_dir_name = input("Enter the address of the root \
                           directory of the numpy arrays \n")
    input_dir_name = os.path.normpath(input_dir_name)
    return input_dir_name


def VerifyDir(dir_name):
    """Checks if directory exist else makes a new one"""

    if os.path.exists(dir_name):
        pass
    else:
        os.mkdir(dir_name)
    return None


def GetOutputRootDir():
    """Gets output directory pathway from user input"""

    output_dir_name = input("Enter the address of the root \
                            directory of the output numpy \
                            array \n")
    output_dir_name = os.path.normpath(output_dir_name)
    return output_dir_name


def GetInputFileNames(dir_name):
    """
    Makes a list of all numpy files in directory

    Parameters
    ----------
    dir_name : str
        The location of the Numpy arrays

    Returns
    -------
    list
        a list of filenames with locations
    """

    File_List = []
    for file in os.listdir(dir_name):
        if file.endswith(".npy"):
            File_List.append(os.path.join(dir_name, file))
    return File_List


'''
Load numpy arrays
'''


def LoadNumpyArray(FileName):
    """
    Load an numpy array into the program

    Parameters
    ----------
    FileName : str
        File name of the numpy array including location.

    Returns
    -------
    pres_arr : ndarray
        Return a numpy array of doubles.

    """

    pres_arr = np.load(FileName)
    # Get pressure array in form [frame][cols][rows]
    pres_arr = pres_arr.swapaxes(0, 2)
    return pres_arr


def PerformPCA(arr, PCA_components):
    """
    Perform Priciple Component analysis on the pressure ndarray

    The function calculates the covariacne between the first axis of the data
    this is the number of frames. The function calculates the covariance in
    frames by using the transpose method Q = X.T * X

    Parameters
    ----------
    arr : ndarray
        Pressure array from the patient data containing pressure values of
        type double.
    PCA_components : int
        Number of components required for the PCA.

    Returns
    -------
    X : ndarray
        A numpy array of PCA performed on the orignal data set.

    """
    X_bar = arr.mean(axis=0)
    Num_of_frames = arr.shape[0]
    for frame in range(Num_of_frames):
        phi = arr[frame] - X_bar
        phi_T = np.transpose(phi)
        if frame == 0:
            COV = np.dot(phi, phi_T)
        else:
            COV += np.dot(phi, phi_T)
    COV = COV / Num_of_frames
    eigenValues, eigenVectors = linalg.eig(COV)
    # rearrange eignevectors in ascending order
    indices = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[indices]
    eigenVec = eigenVectors[:, indices].T
    # Take projection of each pressure array
    # Onto the eigenspace using inner products
    eigenVec = eigenVec[:PCA_components]
    X = np.zeros((arr.shape[0], eigenVec.shape[0], arr.shape[2]))
    for frame in range(Num_of_frames):
        X[frame] = np.dot(eigenVec, arr[frame])
    return X


'''
Perform k-means estimation
'''


def GetLabels(X):
    km = KMeans(n_clusters=3)
    km.fit(X)
    km.predict(X)
    labels = km.labels_
    return labels


'''
Write labels to CSV file
'''


DirName = GetInputRootDir()
AllFiles = GetInputFileNames(DirName)
P = LoadNumpyArray(AllFiles[0])
P_mod = P.reshape(P.shape[0], -1)
Label_norm = GetLabels(P_mod)
Label_PCA = np.zeros((48, P.shape[0]))
for PCA_components in range(2, 50):
    new_arr = PerformPCA(P, PCA_components)
    mod_arr = new_arr.reshape(new_arr.shape[0], -1)
    Label_PCA[PCA_components-2] = GetLabels(mod_arr)
diff_mat = np.subtract(Label_PCA, Label_norm)
variation = np.count_nonzero(diff_mat, axis=1)
PCA_components = np.arange(2, 50)
plt.plot(PCA_components, variation)
plt.show()
