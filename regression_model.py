import os
import joblib

import wfdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from utils import get_heart_rate, strlist_to_list
QTDB_DATASET_PATH = "physionet.org/files/qtdb/1.0.0/"
SAVE_MODEL_PATH = "model/"

def regr_dataset_prep(true_peaks):
    true_peaks_sync = []
    ranges = []
    true_r_peaks = [val for i,val in enumerate(true_peaks) if (i-1)%3 == 0]
    ts, hr = get_heart_rate(true_r_peaks)
    j=4
    for i in range(len(ts)):
        found = False
        while not found:
            if true_peaks[j] == ts[i]:
                range_ = [true_peaks[j-2] - true_peaks[j-3], true_peaks[j] - true_peaks[j-1]]
                true_peaks_sync.append(true_peaks[j-1])
                true_peaks_sync.append(true_peaks[j])
                true_peaks_sync.append(true_peaks[j+1])
                ranges.append(range_)
                found = True
            j+=3
    if len(ts) != len(ranges):
        print("Error")
    return hr, ts, ranges, true_peaks_sync

# Dataset preparation
dataset = pd.read_csv('data/qtdb.csv')
true_peaks_dataset = strlist_to_list(dataset["True Peaks"])

X, y = [], []
l_s = len(dataset['File'])
for temp_file_index, temp_file in enumerate(dataset["File"]):
    
    print(f"{temp_file_index+1}/{l_s}")
    temp_file_path = os.path.join(QTDB_DATASET_PATH, temp_file)
    record_atr = wfdb.rdann(temp_file_path, 'q1c')
    record = wfdb.rdrecord(temp_file_path, channels=[0])
    l_range = 0
    r_range = len(record.p_signal)+1
    data_slice = np.ndarray.flatten(record.p_signal[l_range:r_range])

    true_peaks = true_peaks_dataset[temp_file_index]
    if len(true_peaks) > 6:
        hr, ts, ranges, true_peaks = regr_dataset_prep(true_peaks)

        rt = [i[0] for i in ranges]
        pr = [i[1] for i in ranges] 

        rt_alpha = [i-70 for i in rt]
        pr_alpha = [i-20 for i in pr]

        X.extend(hr)
        y.extend(rt_alpha)

# Model training and saving
X = np.array(X).reshape(-1,1)
model_rf = RandomForestRegressor(n_estimators=200, random_state=100)
model_rf.fit(X, y)

y_pred = model_rf.predict(X)
print(mean_squared_error(y, y_pred, squared=False))
print(r2_score(y, y_pred))

joblib.dump(model_rf , SAVE_MODEL_PATH+"rand_forest.joblib")
