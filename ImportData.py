
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
    df = data.get_filtered_df(xml, mat)
    print(df)
    break