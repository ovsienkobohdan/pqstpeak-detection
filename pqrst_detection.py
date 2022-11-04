from qrs_detectors import qs_peaks_detection

def pqrst_detection(signal, pt_detector, qs_drop=False, r_drop=False):
  peaks = []
  r_peaks = pt_detector.get_rpeaks()
  if not qs_drop:
    peaks.extend(qs_peaks_detection(signal, r_peaks))
  if not r_drop:
    peaks.extend(r_peaks)

  peaks.extend(pt_detector.detect())
  return sorted(peaks)