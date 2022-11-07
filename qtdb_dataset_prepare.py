import os

import wfdb
import pandas as pd

from utils import get_input_files
QTDB_DATASET_PATH = "physionet.org/files/qtdb/1.0.0/"


def del_duplic(symbols, samples):
    clean_symbols = []
    clean_samples = []
    index = 0
    while index < len(symbols) - 1:
        if symbols[index] != symbols[index+1]:
            clean_symbols.append(symbols[index])
            clean_samples.append(samples[index])
        index+=1
    clean_symbols.append(symbols[len(symbols)-1])
    clean_samples.append(len(symbols)-1)
    return clean_symbols, clean_samples

def clear_rng(clean_samples, clean_symbols):
    true_peaks = []
    true_symbols = []

    for peak_index, peak in zip(clean_samples, clean_symbols):
        if peak != "(" and peak != ")":
            true_peaks.append(peak_index)
            true_symbols.append(peak)
    return true_peaks, true_symbols

def pnt_seq(true_peaks, true_symbols):
    indexes = []
    i = 1
    while i <= len(true_symbols)-2:
        if true_symbols[i] == "N":
            if true_symbols[i-1] == "p" and true_symbols[i+1] == "t": 
                indexes.append(i-1)
                indexes.append(i)
                indexes.append(i+1)
        i+=1
    peaks = [true_peaks[i] for i in indexes]
    symbols = [true_symbols[i] for i in indexes]
    return peaks, symbols

def dataset_prep(symbols, samples):
    clean_symbols, clean_samples = del_duplic(symbols, samples)
    true_peaks, true_symbols = clear_rng(clean_samples, clean_symbols)
    peaks, symbols = pnt_seq(true_peaks,true_symbols)
    return peaks, symbols

def save_qtdb_tocsv(db_list):
    db = pd.DataFrame({'File': db_list[0], 'Record': db_list[1], "True Peaks": db_list[2], "Auto Peaks": db_list[3]})
    db.to_csv("data/qtdb.csv", index=False)


input_files = get_input_files(QTDB_DATASET_PATH)

true_peaks_all, true_peaks_all_auto, records_all, file_id = [], [], [], []
for temp_file_index, temp_file in enumerate(input_files):

    temp_file_path = os.path.join(QTDB_DATASET_PATH, temp_file)
    record_atr = wfdb.rdann(temp_file_path, 'q1c')
    record_atr_auto = wfdb.rdann(temp_file_path, 'pu0')
    record = wfdb.rdrecord(temp_file_path, channels=[0])

    true_peaks, true_symbols = dataset_prep(record_atr.symbol, record_atr.sample)
    true_peaks_auto, true_symbols_auto = dataset_prep(record_atr_auto.symbol, record_atr_auto.sample)

    true_peaks_all.append(true_peaks)
    true_peaks_all_auto.append(true_peaks_auto)
    records_all.append(record.p_signal.reshape(-1).tolist())
    file_id.append(temp_file)

save_qtdb_tocsv([file_id, records_all, true_peaks_all, true_peaks_all_auto])