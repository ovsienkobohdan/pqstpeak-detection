import numpy as np
import pandas as pd
# from pt_detectors import WindowedPTDetection
# from qrs_detectors import r_peaks_detection, qs_peaks_detection



# print(np.array([[-4.555],[-4.565],[-4.55 ]]).reshape(-1).tolist())






my_pred = pd.read_csv('data/qtdb_annotations.csv')
jupyter_pred = pd.read_csv('data/qtdb_jupyter_annotations.csv')

ind = 1
print(my_pred['File'][ind])
print(my_pred['Windowed Peaks'][ind][:100])
print(jupyter_pred['P'][ind][:100])

print(jupyter_pred['files'][ind])
print(my_pred['Windowed Peaks'][ind][-100:])
print(jupyter_pred['P'][ind][-100:])


# import numpy as np


# def sum(a1, a2):
#     return [i+a2[0] for i in a1], [i+a1[0] for i in a2]



# a1 = [[0,1,1], [2,2,2], [4,3,3]]
# a2 = [[1,4,4], [3,5,5], [5,6,6]]
# print(np.array(list(map(sum, a1, a2))).shape)
# print(np.array(list(map(sum, a1, a2))))
# print(np.array(list(map(sum, a1, a2)))[:,0,:])
# print(np.array(list(map(sum, a1, a2)))[:,1,:])
