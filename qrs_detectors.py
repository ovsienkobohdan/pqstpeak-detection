from biosppy.signals import ecg
import numpy as np

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