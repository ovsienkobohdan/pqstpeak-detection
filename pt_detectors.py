from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_absolute_error

from qrs_detectors import r_peaks_detection
from pqrst_detection import pqrst_detection
from utils import sync_peaks

class PTPeakDetection(ABC):
    """
    Base class for PT peaks detection on ECG signal. 
    """
    def __init__(self, signal: list, r_peaks: list) -> None:
        self.signal = signal
        self.r_peaks = r_peaks
    
    def get_rpeaks(self) -> list:
        return self.r_peaks

    @abstractmethod
    def detect(self) -> list:
        """Detect PT peaks on signal with annotated r_preaks"""
        pass


class WindowedPTDetection(PTPeakDetection):
    """
    Class for Windowed pt detection
    """
    def __init__(self, signal: str, r_peaks: str, 
                 alpha=0, beta=0, coef=1) -> None:
        super().__init__(signal, r_peaks)
        self.alpha = alpha
        self.beta = beta
        self.coef = coef
    
    def calc_best_params(self, true_peaks: list, a_values=[1,2,3], b_values=[1,2,3]):
        maes_peak = []
        for ind, alpha in enumerate(a_values):
            
            peaks = pqrst_detection(self.signal, WindowedPTDetection(self.signal, self.r_peaks, alpha=alpha), qs_drop=True)
            true_peaks, peaks = sync_peaks(peaks, true_peaks)

            peaks_t = [val for i,val in enumerate(true_peaks) if (i-2)%3 == 0]
            window_peaks_t = [val for i,val in enumerate(peaks) if (i-2)%3 == 0]

            if len(window_peaks_t) > 1:
                maes_peak.append(mean_absolute_error(peaks_t, window_peaks_t))
        
        self.alpha = a_values[np.argmin(maes_peak)] if len(maes_peak) > 0 else 1

        maes_peak = []
        for ind, beta in enumerate(b_values):
            peaks = pqrst_detection(self.signal, WindowedPTDetection(self.signal, self.r_peaks, beta=beta), qs_drop=True)
            true_peaks, peaks = sync_peaks(peaks, true_peaks)

            peaks_p = [val for i,val in enumerate(true_peaks) if i%3 == 0]
            window_peaks_p = [val for i,val in enumerate(peaks) if i%3 == 0]

            if len(window_peaks_p) > 1:
                maes_peak.append(mean_absolute_error(peaks_p, window_peaks_p))
        
        self.beta = b_values[np.argmin(maes_peak)] if len(maes_peak) > 0 else 1

    def detect(self):
        self.p_min, self.p_max = -20, (-40 - self.beta) * self.coef
        self.t_min, self.t_max = 20, (50 + self.alpha) * self.coef
        pt_peaks = []
        for r_peak in self.r_peaks:
            if r_peak + self.p_max >= 0:
                p_signal_slice = self.signal[int(r_peak + self.p_max):int(r_peak + self.p_min)]
                p_peak = np.argmax(p_signal_slice, axis=0) + r_peak + self.p_max
                pt_peaks.append(int(p_peak))
            if r_peak + self.t_max <= len(self.signal):
                t_signal_slice = self.signal[int(r_peak+self.t_min):int(r_peak+self.t_max)]
                t_peak = np.argmax(t_signal_slice, axis=0) + r_peak + self.t_min
                pt_peaks.append(int(t_peak))
        return pt_peaks


#   mae_t = []
#   if len(true_peaks) > 3:
#     l_range = 0
#     r_range = len(record.p_signal)+1

#     data_slice = np.ndarray.flatten(record.p_signal[l_range:r_range])
#     for index, alpha in enumerate(alpha_list):
#       true_peaks, true_symbols = dataset_prep(record_atr.symbol, record_atr.sample)
#       peaks = pqrst_detection(data_slice,qs_drop=True, alpha=alpha)
#       true_peaks, peaks = sync_peaks(peaks, true_peaks)
#       peaks_t = [val for i,val in enumerate(true_peaks) if (i-2)%3 == 0]

#       window_peaks_t = [val for i,val in enumerate(peaks) if (i-2)%3 == 0]

#       mae_t.append(mean_absolute_error(peaks_t, window_peaks_t))

#     alpha = alpha_list[np.argmin(mae_t)]

#     mae_p = []

#     true_peaks, true_symbols = dataset_prep(record_atr.symbol, record_atr.sample)

#     for index, beta in enumerate(beta_list):
#       peaks = pqrst_detection(data_slice,qs_drop=True, beta=beta )
#       true_peaks, peaks = sync_peaks(peaks, true_peaks)

#       peaks_p = [val for i,val in enumerate(true_peaks) if i%3 == 0]

#       window_peaks_p = [val for i,val in enumerate(peaks) if i%3 == 0]
#       if len(window_peaks_p) > 1:
#         mae_p.append(mean_absolute_error(peaks_p, window_peaks_p))

#     beta = beta_list[np.argmin(mae_p)]

# class WindowedABAdaptivePTDetection(WindowedPTDetection):
#     """
#     Class for Windowed pt detection with adaptive ab coef
#     """
#     def __init__(self, signal: str, r_peaks: str, 
#                  alphas_list: list, betas_list: list, coef=1) -> None:
#         super().__init__(signal, r_peaks, alpha=-1, beta=-1, coef=coef)
#         self.betas = betas_list
#         self.alphas = alphas_list
    
#     def calc_best_param(self, param="a"):
#         if param="a":
#             param_values = self.alphas
#             targ_peak = "t"
#         elif param="b":
#             param_values = self.betas
#             targ_peak ="p"
#         else: raise ValueError("wrong value for a param (a or b)")

#         mae_peaks = []
#         for ind, param in enumerate(param_values):

#     def detect(self):
#         super().detect(self)
#         return pt_peaks

