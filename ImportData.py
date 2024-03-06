
# add your own imports if needed
import os
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import xmltodict
import xmljson
import json
from xml.dom import minidom
import numpy as np
from numpy import multiply
from scipy import signal
from collections import defaultdict
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup
import mne
from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess
)

from braindecode.datasets import (
    BaseDataset,
    BaseConcatDataset
)
from Classes import (
    Data,
)


SAMPLING_FREQUENCY = 512
LENGTH_SAMPLE = 5
data_directory_mat = 'BP_EEG_data/mat/'
data_directory_xml = 'BP_EEG_data/xml/'
f_notch = 50
Q = 30

data = Data(data_directory_matlab = data_directory_mat,
            data_directory_xml = data_directory_xml,
            sampling_frequency = SAMPLING_FREQUENCY,
            sample_length = 5,
            f_notch = f_notch,
            q = Q)

for xml, mat in data.get_files():
    raw = data.get_mne_raw(xml, mat)
    print(raw)
    break


dataset = BaseConcatDataset([BaseDataset(data.get_mne_raw(xml, mat)) for xml, mat in data.get_files()])

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000
# Factor to convert from V to uV
factor = 1e6

preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size)
]

# Transform the data
preprocess(dataset, preprocessors, n_jobs=-1)