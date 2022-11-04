import h5py

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# import seaborn as sns
import scipy.signal as signal
# import pywt

import wfdb
import os

from biosppy import storage
from biosppy.signals import ecg

from sklearn.metrics import mean_absolute_error

def r_peaks_detection(signal, sampling_rate=250.):
  out = ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=False)
  return out["rpeaks"]


def qs_peaks_detection(signal, r_peaks, qs_range=20):
  qs_peaks = []
  for r_peak in r_peaks:
    if r_peak - qs_range >= 0:
      q_signal_slice = signal[r_peak - qs_range: r_peak]
      q_peak = np.argmin(q_signal_slice, axis=0) + r_peak - qs_range
      qs_peaks.append(q_peak)
    if r_peak + qs_range <= len(signal):
      s_signal_slice = signal[r_peak: r_peak + qs_range]
      s_peak = np.argmin(s_signal_slice, axis=0) + r_peak
      qs_peaks.append(s_peak)
  return qs_peaks

def pt_peaks_detection(signal, r_peaks, alpha=0, beta=0, coef=1):
  p_min = -20
  p_max = (-40 - beta) * coef
  t_min = 20
  t_max = (50 + alpha) * coef # or do not do +1 -- to try
  pt_peaks = []
  for r_peak in r_peaks:
    if r_peak + p_max >= 0:
      p_signal_slice = signal[int(r_peak + p_max):int(r_peak + p_min)]
      p_peak = np.argmax(p_signal_slice, axis=0) + r_peak + p_max
      pt_peaks.append(p_peak)
    if r_peak + t_max <= len(signal):
      t_signal_slice = signal[int(r_peak+t_min):int(r_peak+t_max)]
      t_peak = np.argmax(t_signal_slice, axis=0) + r_peak + t_min
      pt_peaks.append(t_peak)
  return pt_peaks

def pqrst_detection(signal, qs_drop=False, r_drop=False, alpha=0, beta=0, coef=1):
  peaks = []
  r_peaks = r_peaks_detection(signal)
  if not qs_drop:
    peaks.extend(qs_peaks_detection(signal, r_peaks))
  if not r_drop:
    peaks.extend(r_peaks)
  peaks.extend(pt_peaks_detection(signal, r_peaks, alpha=alpha, beta=beta, coef=coef))
  return sorted(peaks)

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

  for index, (peak_index, peak) in enumerate(zip(clean_samples, clean_symbols)):
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

def dataset_prep(symbol, sample):
  symbols = symbol
  samples = sample
  clean_symbols, clean_samples = del_duplic(symbols, samples)
  true_peaks, true_symbols = clear_rng(clean_samples, clean_symbols)
  peaks, symbols = pnt_seq(true_peaks,true_symbols)
  return peaks, symbols

def sync_peaks(prt_peaks, peaks):
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
  # peaks = ind
  # window_peaks = window_ind
  drops = len(peaks) - len(ind)
  if drops > 0:
    print(f"When syncing missed values was found: {drops} ({drops*100/len(peaks)}%)")
  return ind, window_ind

input_directory = 'physionet.org/files/qtdb/1.0.0/'
input_files = []
for f in os.listdir(input_directory):
    if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith(".") and f.lower().endswith("q1c"):
        input_files.append(f[:-4])
num_files = len(input_files)


# peaks_p_all = []
# peaks_n_all = []
# peaks_t_all = []

# peaks_p_window_all = []
# peaks_n_window_all = []
# peaks_t_window_all = []

# true_peaks_all = []
# true_peaks_all_window = []
# p_all = []
# for temp_file_index, temp_file in enumerate(input_files):
#   print(f"{temp_file_index+1}/{len(input_files)}")
#   temp_file_path = os.path.join(input_directory, temp_file)
#   record_atr = wfdb.rdann(temp_file_path, 'q1c')

#   record = wfdb.rdrecord(temp_file_path, channels=[0])

#   true_peaks, true_symbols = dataset_prep(record_atr.symbol, record_atr.sample)
#   l_range = 0
#   r_range = len(record.p_signal)+1

#   data_slice = np.ndarray.flatten(record.p_signal[l_range:r_range])
#   peaks = pqrst_detection(data_slice,qs_drop=True, alpha=0)
#   p_all.append(peaks)
  
#   true_p, true_p_wind = sync_peaks(peaks, true_peaks)
#   true_peaks_all.extend(true_p)
#   true_peaks_all_window.extend(true_p_wind)

#   peaks_p = [val for i,val in enumerate(true_p) if i%3 == 0]
#   peaks_p_all.extend(peaks_p)
#   peaks_n = [val for i,val in enumerate(true_p) if (i-1)%3 == 0]
#   peaks_n_all.extend(peaks_n)
#   peaks_t = [val for i,val in enumerate(true_p) if (i-2)%3 == 0]
#   peaks_t_all.extend(peaks_t)

#   peaks_p_wind = [val for i,val in enumerate(true_p_wind) if i%3 == 0]
#   peaks_p_window_all.extend(peaks_p_wind)
#   peaks_n_wind = [val for i,val in enumerate(true_p_wind) if (i-1)%3 == 0]
#   peaks_n_window_all.extend(peaks_n_wind)
#   peaks_t_wind = [val for i,val in enumerate(true_p_wind) if (i-2)%3 == 0]
#   peaks_t_window_all.extend(peaks_t_wind)

# df = pd.DataFrame({"files": input_files, "P": p_all})
# df.to_csv("data/qtdb_jupyter_annotations.csv", index=False)

# print(f"MAE score for window algorithm p peaks:  {mean_absolute_error(peaks_p_all, peaks_p_window_all)}")
# print(f"MAE score for window algorithm t peaks:  {mean_absolute_error(peaks_t_all, peaks_t_window_all)}")

# p = 0
# for p_, p_pred in zip(peaks_p_all, peaks_p_window_all):
#  if abs(p_ - p_pred) <= 5:
#    p+=1
# print("Accuracy for window algorithm p peaks: ",int((p * 100)/len(peaks_p_all)) ,"%")

# t = 0
# for t_, t_pred in zip(peaks_t_all, peaks_t_window_all):
#  if abs(t_ - t_pred) <= 5:
#    t+=1
# print("Accuracy for window algorithm t peaks: ",int((t * 100)/len(peaks_t_all)) ,"%")

alpha_list = np.arange(-5,40,0.5)
beta_list = np.arange(-15,50,0.5)

peaks_p_all = []
peaks_n_all = []
peaks_t_all = []

peaks_p_window_ab_all = []
peaks_n_window_ab_all = []
peaks_t_window_ab_all = []

true_peaks_all = []
true_peaks_all_window_ab = []

for temp_file_index, temp_file in enumerate(input_files):

  temp_file_path = os.path.join(input_directory, temp_file)
  record_atr = wfdb.rdann(temp_file_path, 'q1c')

  record = wfdb.rdrecord(temp_file_path, channels=[0])

  true_peaks, true_symbols = dataset_prep(record_atr.symbol, record_atr.sample)

  mae_t = []
  if len(true_peaks) > 3:
    l_range = 0
    r_range = len(record.p_signal)+1

    data_slice = np.ndarray.flatten(record.p_signal[l_range:r_range])
    for index, alpha in enumerate(alpha_list):
      true_peaks, true_symbols = dataset_prep(record_atr.symbol, record_atr.sample)
      peaks = pqrst_detection(data_slice,qs_drop=True, alpha=alpha)
      true_peaks, peaks = sync_peaks(peaks, true_peaks)
      peaks_t = [val for i,val in enumerate(true_peaks) if (i-2)%3 == 0]

      window_peaks_t = [val for i,val in enumerate(peaks) if (i-2)%3 == 0]

      mae_t.append(mean_absolute_error(peaks_t, window_peaks_t))

    alpha = alpha_list[np.argmin(mae_t)]

    mae_p = []

    true_peaks, true_symbols = dataset_prep(record_atr.symbol, record_atr.sample)

    for index, beta in enumerate(beta_list):
      peaks = pqrst_detection(data_slice,qs_drop=True, beta=beta )
      true_peaks, peaks = sync_peaks(peaks, true_peaks)

      peaks_p = [val for i,val in enumerate(true_peaks) if i%3 == 0]

      window_peaks_p = [val for i,val in enumerate(peaks) if i%3 == 0]
      if len(window_peaks_p) > 1:
        mae_p.append(mean_absolute_error(peaks_p, window_peaks_p))

    beta = beta_list[np.argmin(mae_p)]

    l_range = 0
    r_range = len(record.p_signal)+1

    data_slice = np.ndarray.flatten(record.p_signal[l_range:r_range])
    peaks = pqrst_detection(data_slice,qs_drop=True, alpha=alpha, beta=beta)

    true_peaks, true_symbols = dataset_prep(record_atr.symbol, record_atr.sample)

    true_peaks, peaks = sync_peaks(peaks, true_peaks)
    true_peaks_all.extend(true_p)
    true_peaks_all_window.extend(true_p_wind)

    peaks_p = [val for i,val in enumerate(true_peaks) if i%3 == 0]
    peaks_p_all.extend(peaks_p)
    peaks_n = [val for i,val in enumerate(true_peaks) if (i-1)%3 == 0]
    peaks_n_all.extend(peaks_n)
    peaks_t = [val for i,val in enumerate(true_peaks) if (i-2)%3 == 0]
    peaks_t_all.extend(peaks_t)

    window_peaks_p = [val for i,val in enumerate(peaks) if i%3 == 0]
    peaks_p_window_ab_all.extend(window_peaks_p)
    window_peaks_n = [val for i,val in enumerate(peaks) if (i-1)%3 == 0]
    peaks_n_window_ab_all.extend(window_peaks_n)
    window_peaks_t = [val for i,val in enumerate(peaks) if (i-2)%3 == 0]
    peaks_t_window_ab_all.extend(window_peaks_t)

print(f"MAE score for window method with best alpha/beta p peaks:  {mean_absolute_error(peaks_p_all, peaks_p_window_ab_all)}")
print(f"MAE score for window method with best alpha/beta t peaks:  {mean_absolute_error(peaks_t_all, peaks_t_window_ab_all)}")

p = 0
for p_, p_pred in zip(peaks_p_all, peaks_p_window_ab_all):
 if abs(p_ - p_pred) <= 5:
   p+=1
print("Accuracy for window method with best alpha/beta p peaks: ",int((p * 100)/len(peaks_p_all)) ,"%")

t = 0
for t_, t_pred in zip(peaks_t_all, peaks_t_window_ab_all):
 if abs(t_ - t_pred) <= 5:
   t+=1
print("Accuracy for window method with best alpha/beta t peaks: ",int((t * 100)/len(peaks_t_all)) ,"%")

df = pd.DataFrame({"files": input_files, "P": p_all})
df.to_csv("data/qtdb_jupyter_annotations.csv", index=False)