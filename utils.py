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
    drops = len(peaks) - len(ind)
    if drops > 0:
        print(f"When syncing missed values was found: {drops} ({drops*100/len(peaks)}%)")
    return ind, window_ind