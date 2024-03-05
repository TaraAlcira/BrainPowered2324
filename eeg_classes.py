# class voor het inlezen + het extracten van kolom/indices van de taakjes uit de
# data
from scipy.io import loadmat
import numpy as  np
from collections import defaultdict
from scipy import signal
from bs4 import BeautifulSoup

class getEegData:
    def __init__(self):
        pass

    def readFile(self, filepath):
        data = loadmat(filepath)
        return np.array(data['data'])

    def extractColumn(self, col_idx, data):
        return data[:, col_idx]

    def extractOnsetTasks(self, data, n_tasks, col_idx):
        indices_tasks = {}

        # for each task, find the indices of the task
        for nth_task in range(1, n_tasks + 1):
            idx_task = np.where(data[:, col_idx] == float(nth_task))[0]

            # loop through the indices of the task and find the onset markers
            onset_markers = [idx_task[0]]
            for i in range(len(idx_task)-1):
                if idx_task[i+1] - idx_task[i] > 1:
                    onset_markers.append(idx_task[i+1])

            indices_tasks[nth_task] = onset_markers

        return indices_tasks

    def getUsedChannels(self, filepath):
        with open(filepath) as f:
            data = BeautifulSoup(f, "xml")

        # first find all the elements that contain the channel
        # after that extract the name of the channel
        return [elem.find_all('name')[0].get_text().split(" ")[0] for elem in data.find_all('channel')]


# class voor preprocessing
class PreprocessingEeg:
    def __init__(self, fs):
        self.fs = fs

    # function to split the data in chuncks and put in classes
    def getTrials(self, data, sec_task, channels, onset_tasks):
        # structure of the dictionaries: class[channel[trial]]
        tasks_raw = defaultdict(lambda : defaultdict(list))
        tasks_filtered = defaultdict(lambda : defaultdict(list))

        # loop through the channels
        for i, channel_name in enumerate(channels):
            filtered_data = self.notchAndBandpass(data[:, i])

            for nth_task, idxs in onset_tasks.items():
                for idx in idxs:
                    tasks_raw[nth_task][channel_name].append(data[idx:idx + (sec_task * self.fs), i])
                    tasks_filtered[nth_task][channel_name].append(filtered_data[idx: idx + (sec_task * self.fs)])

        return tasks_raw, tasks_filtered


    def notchAndBandpass(self, data_col):
        f_notch = 50
        Q = 30
        b, a = signal.iirnotch(f_notch, Q, self.fs)

        sos = signal.butter(N=10, Wn=[8, 15], btype="bandpass", output="sos", fs=self.fs)

        return signal.lfilter(b, a, signal.sosfilt(sos, data_col))

class visualizeEEG:
    def __init__(self, fs):
        self.fs = fs

    def plot_spectrum(channel, data):
        pass

# feature extraction methods


# class voor classficatie