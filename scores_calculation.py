import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

def sync_peaks(prt_peaks, peaks, print_misssing=False):
    window_ind = []
    ind = []
    find_j = -2

    for i in range(1, len(peaks),3):
        find = False
        j = find_j + 3
        while not find and j <= len(prt_peaks)-2:
            if abs(peaks[i] - prt_peaks[j]) < 20:
                window_ind += [prt_peaks[j-1], prt_peaks[j], prt_peaks[j+1]]
                ind += [peaks[i-1], peaks[i], peaks[i+1]]
                find_j = j
                find = True
            else: 
                j += 3
    drops = len(peaks) - len(ind)
    if drops > 0 and print_misssing:
        print(f"When syncing missed values was found: {drops} ({drops*100/len(peaks)}%)")
    return ind, window_ind

def sync_peaks_all_records(prt_peaks, peaks):
    synced_peaks = list(map(sync_peaks, prt_peaks, peaks))
    return [i[0] for i in synced_peaks], [i[1] for i in synced_peaks]

def flatten(l):
    return [item for sublist in l for item in sublist]

def strlist_to_list(strl):
    full_str = []
    for i in strl:
        if i == "[]":
            full_str.append([])
        else:
            full_str.append([int(j) for j in i[1:-1].split(",")])
    return full_str

def group_tp_peaks(peaks):
    peaks_p = [val for i,val in enumerate(peaks) if i%3 == 0]
    peaks_t = [val for i,val in enumerate(peaks) if (i-2)%3 == 0]
    return [peaks_p, peaks_t]

def pt_peaks_all_records(all_peaks):
    pt_peaks = list(map(group_tp_peaks, all_peaks))
    return flatten([i[0] for i in pt_peaks]), flatten([i[1] for i in pt_peaks])

def accuracy(true_peaks, auto_peaks):
    acc = 0
    for p_, p_pred in zip(true_peaks, auto_peaks):
        if abs(p_ - p_pred) <= 5:
            acc+=1
    return int((acc * 100)/len(true_peaks))


def calc_scores(true_peaks, auto_peaks, desc):
    true_p, true_p_auto = sync_peaks_all_records(auto_peaks, true_peaks)

    peaks_p_all, peaks_t_all = pt_peaks_all_records(true_p)
    peaks_p_auto_all, peaks_t_auto_all = pt_peaks_all_records(true_p_auto)

    print(f"MAE score for {desc} p peaks:  {mean_absolute_error(peaks_p_all, peaks_p_auto_all)}")
    print(f"MAE score for {desc} t peaks:  {mean_absolute_error(peaks_t_all, peaks_t_auto_all)}")

    print(f"Accuracy for {desc} p peaks: {accuracy(peaks_p_all, peaks_p_auto_all)}%")
    print(f"Accuracy for {desc} t peaks: {accuracy(peaks_t_all, peaks_t_auto_all)}%")


# Comparing manual and auto detection from dataset
dataset = pd.read_csv('data/qtdb.csv')
true_peaks = strlist_to_list(dataset["True Peaks"])
auto_peaks = strlist_to_list(dataset["Auto Peaks"])
calc_scores(true_peaks, auto_peaks, desc="dataset's automatically determined waveform boundary measurements")
