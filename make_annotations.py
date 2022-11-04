import pandas as pd
import numpy as np

from utils import strlist_to_list
from qrs_detectors import r_peaks_detection
from pt_detectors import WindowedPTDetection
from pqrst_detection import pqrst_detection

dataset = pd.read_csv('data/qtdb.csv')
signals = strlist_to_list(dataset['Record'], float)
true_peaks = strlist_to_list(dataset['True Peaks'], int)
windowed_peaks = []
windowed_ab_peaks = []

l_s = len(signals) # del later
for sig_ind, sig in enumerate(signals):
    print(f"{sig_ind+1}/{l_s}")

    r_peaks = r_peaks_detection(sig)
    
    # pt_peaks_detector = WindowedPTDetection(signal=sig, r_peaks=r_peaks)
    # peaks = pqrst_detection(sig, pt_peaks_detector, qs_drop=True)
    # windowed_peaks.append(peaks)

    pt_ab_peaks_detector = WindowedPTDetection(signal=sig, r_peaks=r_peaks)
    a_values, b_values = np.arange(-5,40,0.5), np.arange(-15,50,0.5)
    pt_ab_peaks_detector.calc_best_params(true_peaks[sig_ind], a_values, b_values)
    peaks = pqrst_detection(sig, pt_ab_peaks_detector, qs_drop=True)
    windowed_ab_peaks.append(peaks)

# print(windowed_ab_peaks)
# dataset['Windowed Peaks'] = windowed_peaks
dataset['Windowed AB Peaks'] = windowed_ab_peaks
dataset.to_csv("data/qtdb_annotations.csv", index=False)