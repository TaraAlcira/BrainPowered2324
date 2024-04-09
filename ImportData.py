
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

# from braindecode.preprocessing import (
#     Preprocessor,
#     exponential_moving_standardize,
#     preprocess
# )


# from braindecode.datasets import (
#     BaseDataset,
#     BaseConcatDataset
# )

from Classes import (
    Data,
)


import pandas as pd





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

    one_hot_df = data.get_df_with_onehot_encoded_marker(xml, mat)

    break

print(one_hot_df.columns)

# Assuming 'cluster_column' is the column representing clusters in your DataFrame
# You need to replace 'cluster_column' with the actual column name in your DataFrame.
cluster_column = 'marker_1.0'

# Create a new DataFrame to store the first row of each cluster for every one-hot encode column
first_row_df = pd.DataFrame()

# Iterate through each column in the one-hot encoded DataFrame
for column in one_hot_df[['marker_0.0', 'marker_1.0', 'marker_2.0', 'marker_3.0', 'marker_4.0']]:
    print(column)
    # Check if the column is a one-hot encoded column
    if one_hot_df[column].nunique() == 2 and set([0, 1]).issuperset(one_hot_df[column].unique()):
        # Get the first row of each cluster for the current column
        first_row = one_hot_df.groupby([cluster_column, column]).first().reset_index().groupby(cluster_column).first()[column]

        # Append the result to the new DataFrame
        first_row_df[column + "clustered"] = first_row

# Reset the index of the new DataFrame if needed
# first_row_df.reset_index(inplace=True)

# Display the resulting DataFrame
print(first_row_df)


quit()




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

    df = data.get_df_with_onehot_encoded_marker(xml, mat)


    # calculate clusters of features
    idx_condition_1 = df.groupby('marker_1.0').first()
    idx_condition_2 = df[df["marker_2.0"]]
    idx_condition_3 = df[df["marker_3.0"]]
    idx_condition_4 = df[df["marker_4.0"]]

    for nth_channel in range(len(channel_names)):

        channel_data = eeg_measurements[:, nth_channel]
        for idx_1, idx_2 in zip(idx_condition_1, idx_condition_2):

            condition_1_raw[channel_names[nth_channel]].append(channel_data[idx_1:idx_1 + fs * 10])
            condition_1_filt[channel_names[nth_channel]].append(signal.lfilter(b, a, signal.sosfilt(sos, channel_data)[idx_1:idx_1 + fs * 10]))

            condition_2_raw[channel_names[nth_channel]].append(channel_data[idx_2:idx_2 + fs * 10])
            condition_2_filt[channel_names[nth_channel]].append(signal.lfilter(b, a, signal.sosfilt(sos, channel_data)[idx_2:idx_2 + fs * 10]))

    # the shape of the data is nchannels, ntrials, nsamples so the data needs to be transposed ntrials, nchannels, nsamples
    x_condition_1_filt = np.array([condition_1_filt[channel] for channel in channel_names]).transpose(1, 0, 2)
    # needs to have the same length
    y_condition_1_filt = np.zeros(x_condition_1_filt.shape[0])

    x_condition_2_filt = np.array([condition_2_filt[channel] for channel in channel_names]).transpose(1, 0, 2)
    y_condition_2_filt = np.ones(x_condition_2_filt.shape[0])


    X_train_filt = np.concatenate([x_condition_1_filt, x_condition_2_filt, x_condition_3_filt, x_condition_4_filt])
    y_train_filt = np.concatenate([y_condition_1_filt, y_condition_2_filt, y_condition_3_filt, y_condition_4_filt])

    n_components = 2
    csp_model = CSP(n_components=n_components, reg=None, log=None, norm_trace=False)

    x_train_csp_filt = csp_model.fit_transform(X_train_filt, y_train_filt)


    X_train_filt_s, X_test_filt_s, y_train_filt_s, y_test_filt_s = train_test_split(X_train_filt, y_train_filt, test_size=0.2, shuffle=True)

    X_train_csp_filt = csp_model.fit_transform(X_train_filt_s, y_train_filt_s)
    X_test_csp_filt = csp_model.fit_transform(X_test_filt_s, y_test_filt_s)

    svm = SVC(kernel='linear')
    lda = LinearDiscriminantAnalysis()
    knn = KNeighborsClassifier(n_neighbors=3)

    svm.fit(X_train_csp_filt, y_train_filt_s)
    svm_test_predictions = svm.predict(X_test_csp_filt)
    print("Accuracy SVM filtered: ", accuracy_score(y_test_filt_s, svm_test_predictions))

    lda.fit(X_train_csp_filt, y_train_filt_s)
    lda_test_predictions = lda.predict(X_test_csp_filt)
    print("Accuracy LDA filtered: ", accuracy_score(y_test_filt_s, lda_test_predictions))

    knn.fit(X_train_csp_filt, y_train_filt_s)
    knn_test_predictions = knn.predict(X_test_csp_filt)
    print("Accuracy KNN filtered: ", accuracy_score(y_test_filt_s, knn_test_predictions))

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