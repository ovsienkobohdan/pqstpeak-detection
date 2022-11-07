import pandas as pd
from sklearn.metrics import mean_absolute_error

from utils import strlist_to_list, sync_peaks_all_records

def flatten(l):
    return [item for sublist in l for item in sublist]

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

    print(f"MAE score for {desc} p peaks:  {round(mean_absolute_error(peaks_p_all, peaks_p_auto_all),2)}")
    print(f"MAE score for {desc} t peaks:  {round(mean_absolute_error(peaks_t_all, peaks_t_auto_all),2)}")

    print(f"Accuracy for {desc} p peaks: {accuracy(peaks_p_all, peaks_p_auto_all)}%")
    print(f"Accuracy for {desc} t peaks: {accuracy(peaks_t_all, peaks_t_auto_all)}%")


# Comparing manual and auto detection from dataset
dataset = pd.read_csv('data/qtdb.csv')
true_peaks = strlist_to_list(dataset["True Peaks"])
auto_peaks = strlist_to_list(dataset["Auto Peaks"])
calc_scores(true_peaks, auto_peaks, desc="dataset's automatically determined waveform boundary measurements")

# Comparing manual and windowed pt annotation
annotated_dataset = pd.read_csv('data/qtdb_annotations.csv')
auto_peaks = strlist_to_list(annotated_dataset["Windowed Peaks"])
calc_scores(true_peaks, auto_peaks, desc="window algorithm")

# Comparing manual and windowed pt annotation with adaptive ab
annotated_dataset = pd.read_csv('data/qtdb_annotations.csv')
auto_peaks = strlist_to_list(annotated_dataset["Windowed AB Peaks"])
calc_scores(true_peaks, auto_peaks, desc="window method with best alpha/beta")

# Comparing manual and windowed pt annotation with adaptive ab, coef
annotated_dataset = pd.read_csv('data/qtdb_annotations.csv')
auto_peaks = strlist_to_list(annotated_dataset["Windowed AB Coef Peaks"])
calc_scores(true_peaks, auto_peaks, desc="window method with best alpha/beta and coef")

# Comparing manual and windowed pt annotation with adaptive ab, coef and model predictions
# annotated_dataset = pd.read_csv('data/qtdb_annotations.csv')
# auto_peaks = strlist_to_list(annotated_dataset["Windowed AB Coef Model Peaks"])
# calc_scores(true_peaks, auto_peaks, desc="window method with best alpha/beta, coef and model")