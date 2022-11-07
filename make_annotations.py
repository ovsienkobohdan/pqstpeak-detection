import pandas as pd
import numpy as np

from utils import strlist_to_list
from qrs_detectors import r_peaks_detection
from pt_detectors import WindowedPTDetection
from pqrst_detection import pqrst_detection
FS=250.
MODEL_PATH = "model/rand_forest.joblib"
DATASET_PATH = 'data/qtdb.csv'

dataset = pd.read_csv(DATASET_PATH)
signals = strlist_to_list(dataset['Record'], float)
true_peaks = strlist_to_list(dataset['True Peaks'], int)

windowed_peaks = []
windowed_ab_peaks = []
windowed_ab_coef_peaks = []
windowed_ab_coef_model_peaks = []

l_s = len(signals)
for sig_ind, sig in enumerate(signals):
    print(f"{sig_ind+1}/{l_s}")

    r_peaks = r_peaks_detection(sig)
    
    # Prediction on window 
    pt_peaks_detector = WindowedPTDetection(signal=sig, r_peaks=r_peaks, sampling_rate=FS)
    peaks = pqrst_detection(sig, pt_peaks_detector, qs_drop=True)
    windowed_peaks.append(peaks)

    # Prediction on window with best ab parameters
    pt_ab_peaks_detector = WindowedPTDetection(signal=sig, r_peaks=r_peaks, sampling_rate=FS)
    a_values, b_values = np.arange(-5,40,0.5), np.arange(-15,50,0.5)
    pt_ab_peaks_detector.calc_best_params(true_peaks[sig_ind], a_values, b_values)
    peaks = pqrst_detection(sig, pt_ab_peaks_detector, qs_drop=True)
    windowed_ab_peaks.append(peaks)

    # Prediction on window with best ab parameters + coef
    pt_abc_peaks_detector = WindowedPTDetection(signal=sig, r_peaks=r_peaks, sampling_rate=FS)
    a_values, b_values = np.arange(-5,40,1), np.arange(-15,50,1)
    pt_abc_peaks_detector.calc_best_params(true_peaks[sig_ind], a_values, b_values)
    pt_abc_peaks_detector.calc_adaptive_coef()
    peaks = pqrst_detection(sig, pt_abc_peaks_detector, qs_drop=True)
    windowed_ab_coef_peaks.append(peaks)

    # !!! Prediction with model can take a long time >1h (uncomment to run) !!!
    # pt_abc_peaks_model_detector = WindowedPTDetection(signal=sig, r_peaks=r_peaks, sampling_rate=FS)
    # a_values, b_values = np.arange(-5,40,1), np.arange(-15,50,1)
    # pt_abc_peaks_model_detector.calc_best_params(true_peaks[sig_ind], a_values, b_values)
    # pt_abc_peaks_model_detector.calc_adaptive_coef()
    # pt_abc_peaks_model_detector.init_model(MODEL_PATH)
    # peaks = pqrst_detection(sig, pt_abc_peaks_model_detector, qs_drop=True)
    # windowed_ab_coef_model_peaks.append(peaks)

dataset['Windowed Peaks'] = windowed_peaks
dataset['Windowed AB Peaks'] = windowed_ab_peaks
dataset['Windowed AB Coef Peaks'] = windowed_ab_coef_peaks
# dataset['Windowed AB Coef Model Peaks'] = windowed_ab_coef_model_peaks
dataset.to_csv("data/qtdb_annotations.csv", index=False)