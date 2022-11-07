import numpy as np

import os

def strlist_to_list(strl, type=int):
    full_str = []
    for i in strl:
        if i == "[]":
            full_str.append([])
        else:
            full_str.append([type(j) for j in i[1:-1].split(",")])
    return full_str

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

def get_input_files(main_dir):
    input_files = []
    for f in os.listdir(main_dir):
        if os.path.isfile(os.path.join(main_dir, f)) and not f.lower().startswith(".") and f.lower().endswith("q1c"):
            input_files.append(f[:-4])
    return input_files

def get_heart_rate(beats=None, sampling_rate=250.):

    if beats is None:
        raise TypeError("Please specify the input beat indices.")

    if len(beats) < 2:
        raise ValueError("Not enough beats to compute heart rate.")

    ts = beats[1:]
    hr = sampling_rate * (60. / np.diff(beats))

    indx = [i for i,val in enumerate(hr) if val>=40 and val <=200]
    ts = [ts[i] for i in indx]
    hr = [hr[i] for i in indx]

    return ts, hr