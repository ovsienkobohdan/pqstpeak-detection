from abc import ABC, abstractmethod
import joblib

import numpy as np
from sklearn.metrics import mean_absolute_error
from biosppy.signals import ecg

from pqrst_detection import pqrst_detection
from utils import sync_peaks, get_heart_rate

class PTPeakDetection(ABC):
    """
    Base class for PT peaks detection on ECG signal. 
    """
    def __init__(self, signal: list, r_peaks: list, sampling_rate: float) -> None:
        self.signal = signal
        self.r_peaks = r_peaks
        self.samp_rate = sampling_rate
    
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
    def __init__(self, signal: str, r_peaks: str, sampling_rate: float,
                 alpha=0, beta=0, coef=1) -> None:
        super().__init__(signal, r_peaks, sampling_rate)
        self.alpha = alpha
        self.beta = beta
        self.coef = coef
        self.model = None
    
    def init_model(self, model_path):
        self.model = joblib.load(model_path)

    def calc_best_params(self, true_peaks: list, a_values=[1,2,3], b_values=[1,2,3]):
        maes_peak = []
        for ind, alpha in enumerate(a_values):
            
            peaks = pqrst_detection(self.signal, WindowedPTDetection(self.signal, self.r_peaks, 
                                    self.samp_rate, alpha=alpha), qs_drop=True)
            true_peaks, peaks = sync_peaks(peaks, true_peaks)

            peaks_t = [val for i,val in enumerate(true_peaks) if (i-2)%3 == 0]
            window_peaks_t = [val for i,val in enumerate(peaks) if (i-2)%3 == 0]

            if len(window_peaks_t) > 1:
                maes_peak.append(mean_absolute_error(peaks_t, window_peaks_t))
        
        self.alpha = a_values[np.argmin(maes_peak)] if len(maes_peak) > 0 else 1

        maes_peak = []
        for ind, beta in enumerate(b_values):
            peaks = pqrst_detection(self.signal, WindowedPTDetection(self.signal, self.r_peaks,
                                    self.samp_rate, beta=beta), qs_drop=True)
            true_peaks, peaks = sync_peaks(peaks, true_peaks)

            peaks_p = [val for i,val in enumerate(true_peaks) if i%3 == 0]
            window_peaks_p = [val for i,val in enumerate(peaks) if i%3 == 0]

            if len(window_peaks_p) > 1:
                maes_peak.append(mean_absolute_error(peaks_p, window_peaks_p))
        
        self.beta = b_values[np.argmin(maes_peak)] if len(maes_peak) > 0 else 1

    def calc_adaptive_coef(self):
        _, self.hrs = get_heart_rate(self.r_peaks)
        self.mean_hr = np.mean(self.hrs)
        self.coef = [hr / self.mean_hr for hr in self.hrs]
    
    def init_regr_model(self, model):
        self.model = model
    
    def detect(self):
        if type(self.coef) is list:
            pt_peaks = []
            for i, c in enumerate(self.coef):
                p_min = -20
                if self.model is not None:
                    p_max = int((-40 - self.model.predict(np.array(self.hrs[i]).reshape(-1,1))) * c)
                else:
                    p_max = int((-40 - self.beta) * c)
                t_min = 20
                t_max = int((50 + self.alpha) * c)

                if i == 0 and self.r_peaks[i] + p_max >= 0 :
                    p_min = -20
                    if self.model is not None:
                        p_max = (-40 - self.model.predict(np.array(self.mean_hr).reshape(-1,1)))
                    else: 
                        p_max = (-40 - self.beta)
                    t_min = 20
                    t_max = (50 + self.alpha)
                    p_signal_slice = self.signal[int(self.r_peaks[i] + p_max):int(self.r_peaks[i] + p_min)]
                    if len(p_signal_slice) > 1:
                        p_peak = np.argmax(p_signal_slice, axis=0) + self.r_peaks[i] + p_max
                        pt_peaks.append(int(p_peak))

                p_signal_slice = self.signal[int(self.r_peaks[i+1] + p_max):int(self.r_peaks[i+1] + p_min)]
                if len(p_signal_slice) > 1:
                    p_peak = np.argmax(p_signal_slice, axis=0) + self.r_peaks[i+1] + p_max
                    pt_peaks.append(int(p_peak))
                t_signal_slice = self.signal[int(self.r_peaks[i]+t_min):int(self.r_peaks[i]+t_max)]
                if len(t_signal_slice) > 1:  
                    t_peak = np.argmax(t_signal_slice, axis=0) + self.r_peaks[i] + t_min
                    pt_peaks.append(int(t_peak))

            p_min = -20
            p_max = (-40 - self.beta)
            t_min = 20
            t_max = (50 + self.alpha)
            if self.r_peaks[len(self.r_peaks)-1] + t_max <= len(self.signal):
                t_signal_slice = self.signal[int(self.r_peaks[len(self.r_peaks)-1]+t_min):int(self.r_peaks[len(self.r_peaks)-1]+t_max)]
                t_peak = np.argmax(t_signal_slice, axis=0) + self.r_peaks[len(self.r_peaks)-1] + t_min
                pt_peaks.append(t_peak)
        else:
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