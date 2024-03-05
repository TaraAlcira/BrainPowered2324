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

SAMPLING_FREQUENCY = 512
data_directory_mat = 'BP_EEG_data/mat/'
data_directory_xml = 'BP_EEG_data/xml/'



# Get the list of subjects in the data, and sort the list
subject_files_mat = os.listdir(data_directory_mat)
subject_files_mat.sort()

subject_files_xml = os.listdir(data_directory_xml)
subject_files_xml.sort()

subject_files = zip(subject_files_xml, subject_files_mat)


for file_xml, file_mat in subject_files:
    with open(data_directory_xml + file_xml, "r") as contents:
        content = contents.read()
        # print(content)
        try:
            soup = BeautifulSoup(content, 'xml')
            print("check1")
        except:
            soup = BeautifulSoup(content, 'lxml')
            print("check2")

        titles = soup.find_all('name')
        labels = [x.text for x in titles][:-1]
        print("Number of labels: ", len(labels))
        df = pd.DataFrame(scipy.io.loadmat(data_directory_mat + file_mat)["data"], columns=labels)

        # break

        markers = list(df["marker"].unique())
        print("Markers: ", markers)
        dfs_per_marker = [df[df["marker"] == item] for item in markers]
        print(dfs_per_marker)

        subject_matrix_no_marker = df.drop("marker", axis='columns')
        subject_matrix_no_marker = subject_matrix_no_marker.to_numpy()
        subject_matrix = df.to_numpy()
        print("Number of rows: ", len(subject_matrix))
        eeg_measurements = df.drop("marker", axis='columns')
        n_channels = len(eeg_measurements.columns)
        print(df.columns)

        fig, axs = plt.subplots(n_channels, sharex=True)
        # adjust sizes of plot
        fig.set_figheight(30)
        fig.set_figwidth(20)
        plt.subplots_adjust(hspace=1.0)

        for nth_channel in range(n_channels):
            axs[nth_channel].plot(eeg_measurements.index, eeg_measurements)
            axs[nth_channel].set_title(labels[nth_channel])

        # for ax in axs.flat:
        #     ax.set(xlabel='Timestamps', ylabel='Voltage (uV)')
        #
        # plt.show()

    # only one trial for testing purposes
    break

# frequency you want to remove
f_notch = 50
# the quality factor of the notch filter
Q = 30
# also in this function always give your sampling frequency to the function
b, a = signal.iirnotch(f_notch, Q, SAMPLING_FREQUENCY)

sos = signal.butter(N=10, Wn=[8, 15], btype="bandpass", output="sos", fs=SAMPLING_FREQUENCY)
# print the filter to see how it changes with the order
print("sos: ", sos)
print(df["marker"].unique())

condition_1_index = df[df["marker"] == '1'].index.tolist()
condition_2_index = df[df["marker"] == '2'].index.tolist()
condition_3_index = df[df["marker"] == '3'].index.tolist()
condition_4_index = df[df["marker"] == '4'].index.tolist()


fig, axs = plt.subplots(n_channels, sharex=True)
fig.set_figheight(30)
fig.set_figwidth(20)
plt.subplots_adjust(hspace=1.0)
# plt.title(f"Different channels EEG data {subject} after applying a notch and bandpass filter")
# print(f"{subject} EEG measurements filtered data")

for channel in eeg_measurements.columns:
    # filter the eeg measurement for the channel separately
    filtered_eeg = signal.sosfilt(sos, eeg_measurements[channel])

    # plot the filtered eeg signal
    # axs[nth_channel].plot(time_stamps[idx_condition_1[0]:], filtered_eeg[idx_condition_1[0]:])
    # axs[nth_channel].set_title(channel_names[nth_channel])

for ax in axs.flat:
    ax.set(xlabel='Timestamps', ylabel='Voltage (uV)')
plt.show()






# ---



# the shape of the data is nchannels, ntrials, nsamples so the data needs to be transposed ntrials, nchannels, nsamples
x_condition_1_filt = np.array([condition_1_filt[channel] for channel in channel_names]).transpose(1, 0, 2)
# needs to have the same length
y_condition_1_filt = np.zeros(x_condition_1_filt.shape[0])

x_condition_2_filt = np.array([condition_2_filt[channel] for channel in channel_names]).transpose(1, 0, 2)
y_condition_2_filt = np.ones(x_condition_2_filt.shape[0])

x_condition_1_raw = np.array([condition_1_raw[channel] for channel in channel_names]).transpose(1, 0, 2)
y_condition_1_raw = np.zeros(x_condition_1_raw.shape[0])

x_condition_2_raw = np.array([condition_2_raw[channel] for channel in channel_names]).transpose(1, 0, 2)
y_condition_2_raw = np.ones(x_condition_2_raw.shape[0])

X_train_filt = np.concatenate([x_condition_1_filt, x_condition_2_filt])
y_train_filt = np.concatenate([y_condition_1_filt, y_condition_2_filt])

X_train_raw = np.concatenate([x_condition_1_raw, x_condition_2_raw])
y_train_raw = np.concatenate([y_condition_1_raw, y_condition_2_raw])

n_components = 2
csp_model = CSP(n_components=n_components, reg=None, log=None, norm_trace=False)

# # returns n_samples, n_features_new
x_train_csp_filt = csp_model.fit_transform(X_train_filt, y_train_filt)
x_train_csp_raw = csp_model.fit_transform(X_train_raw, y_train_raw)

# class 0
plt.scatter(x_train_csp_filt[:, 0][y_train_filt == 0], x_train_csp_filt[:, 1][y_train_filt == 0], label='eyes closed')
# class 1
plt.scatter(x_train_csp_filt[:, 0][y_train_filt == 1], x_train_csp_filt[:, 1][y_train_filt == 1], alpha=0.7, label='eyes open')
plt.title('The different classes after CSP feature extraction with the filtered data')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend()
plt.show()

# class 0
plt.scatter(x_train_csp_raw[:, 0][y_train_raw == 0], x_train_csp_raw[:, 1][y_train_raw == 0], label='eyes closed')
# class 1
plt.scatter(x_train_csp_raw[:, 0][y_train_raw == 1], x_train_csp_raw[:, 1][y_train_raw == 1], alpha=0.3, label='eyes open')
plt.title('The different classes after CSP feature extraction with the raw data')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend()
plt.show()


# setting a random state so the experiment can be replicated
X_train_filt_s, X_test_filt_s, y_train_filt_s, y_test_filt_s = train_test_split(X_train_filt, y_train_filt, test_size=0.2, shuffle=True)
X_train_raw_s, X_test_raw_s, y_train_raw_s, y_test_raw_s = train_test_split(X_train_raw, y_train_raw, test_size=0.2, shuffle=True)

X_train_csp_filt = csp_model.fit_transform(X_train_filt_s, y_train_filt_s)
# class 0
plt.scatter(X_train_csp_filt[:, 0][y_train_filt_s == 0], X_train_csp_filt[:, 1][y_train_filt_s == 0], label='eyes closed')
# class 1
plt.scatter(X_train_csp_filt[:, 0][y_train_filt_s == 1], X_train_csp_filt[:, 1][y_train_filt_s == 1], alpha=0.7, label='eyes open')
plt.title('Training set with the filtered data')
plt.legend()
plt.show()


X_train_csp_raw = csp_model.fit_transform(X_train_raw_s, y_train_raw_s)
# class 0
plt.scatter(X_train_csp_raw[:, 0][y_train_raw_s == 0], X_train_csp_raw[:, 1][y_train_raw_s == 0], label='eyes closed')
# class 1
plt.scatter(X_train_csp_raw[:, 0][y_train_raw_s == 1], X_train_csp_raw[:, 1][y_train_raw_s == 1], alpha=0.3, label='eyes open')
plt.title('Training set with the raw data')
plt.legend()
plt.show()


X_test_csp_filt = csp_model.fit_transform(X_test_filt_s, y_test_filt_s)
# class 0
plt.scatter(X_test_csp_filt[:, 0][y_test_filt_s == 0], X_test_csp_filt[:, 1][y_test_filt_s == 0], label='eyes closed')
# class 1
plt.scatter(X_test_csp_filt[:, 0][y_test_filt_s == 1], X_test_csp_filt[:, 1][y_test_filt_s == 1], alpha=0.7, label='eyes open')
plt.title('Test set with the filtered data')
plt.legend()
plt.show()
X_test_csp_raw = csp_model.fit_transform(X_test_raw_s, y_test_raw_s)
# class 0
plt.scatter(X_test_csp_raw[:, 0][y_test_raw_s == 0], X_test_csp_raw[:, 1][y_test_raw_s == 0], label='eyes closed')
# class 1
plt.scatter(X_test_csp_raw[:, 0][y_test_raw_s == 1], X_test_csp_raw[:, 1][y_test_raw_s == 1], alpha=0.7, label='eyes open')
plt.title('Test set with the raw data')
plt.legend()
plt.show()



svm = SVC(kernel='linear')
lda = LinearDiscriminantAnalysis()
knn = KNeighborsClassifier(n_neighbors=3)

svm.fit(X_train_csp_filt, y_train_filt_s)
svm_test_predictions = svm.predict(X_test_csp_filt)
print("Accuracy SVM filtered: ", accuracy_score(y_test_filt_s, svm_test_predictions))

svm.fit(X_train_csp_raw, y_train_raw_s)
svm_test_predictions = svm.predict(X_test_csp_raw)
print("Accuracy SVM raw: ", accuracy_score(y_test_raw_s, svm_test_predictions))

lda.fit(X_train_csp_filt, y_train_filt_s)
lda_test_predictions = lda.predict(X_test_csp_filt)
print("Accuracy LDA filtered: ", accuracy_score(y_test_filt_s, lda_test_predictions))

lda.fit(X_train_csp_raw, y_train_raw_s)
lda_test_predictions = lda.predict(X_test_csp_raw)
print("Accuracy LDA raw: ", accuracy_score(y_test_raw_s, lda_test_predictions))

knn.fit(X_train_csp_filt, y_train_filt_s)
knn_test_predictions = knn.predict(X_test_csp_filt)
print("Accuracy KNN filtered: ", accuracy_score(y_test_filt_s, knn_test_predictions))

knn.fit(X_train_csp_raw, y_train_raw_s)
knn_test_predictions = knn.predict(X_test_csp_raw)
print("Accuracy KNN raw: ", accuracy_score(y_test_raw_s, knn_test_predictions))



print(df)
