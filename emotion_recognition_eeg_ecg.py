import numpy as np
import pyeeg

with open("ecg.txt") as ecg_data:
    ecg = []
    for line in ecg_data:
        ecg_ls = list(map(float, line.rstrip().split(",")))
        while ecg_ls[-1] == 0.0:
            del ecg_ls[-1]
        ecg.append(ecg_ls)

ecg_array=np.array([np.array(ecg_i) for ecg_i in ecg])

ecg_lengths = []
for ecg_i in ecg_array:
    ecg_lengths.append(len(ecg_i))

eeg_array=np.array([np.array(eeg_i) for eeg_i in eeg])

with open("valence.txt") as valence_data:
    valence = []
    for line in valence_data:
        val_ls = list(map(int, line.rstrip().split(",")))
        for i in range(0,len(val_ls)):
            if val_ls[i] > 3:
                val_ls[i] = 1
            else:
                val_ls[i]=0
        valence.append(val_ls)

valence_array = np.array([np.array(val_i) for val_i in valence])
valence_array = np.reshape(valence_array,(414,1))

with open("arousal.txt") as arousal_data:
    arousal = []
    for line in arousal_data:
        aro_ls = list(map(int, line.rstrip().split(",")))
        for i in range(0,len(val_ls)):
            if aro_ls[i] > 3:
                aro_ls[i] = 1
            else:
                aro_ls[i]=0
        arousal.append(aro_ls)

arousal_array = np.array([np.array(ar_i) for ar_i in arousal])
arousal_array = np.reshape(arousal_array,(414,1))

import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfilt, sosfreqz
from scipy.signal import sosfiltfilt
import matplotlib.pyplot as plt

# design bandpass butterworth filter
def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfiltfilt(sos, data)
        return y

sos = butter_bandpass(5, 20, 128, order=5)
w, h = sosfreqz(sos, worN=2000)
plt.plot(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.show()
plt.savefig("butterworth.pdf")

#filter eeg
filtered_eeg = []
lowcut_eeg = 5.0
highcut_eeg = 20.0
fs_eeg = 128.0
order = 20
for eeg_i in eeg:
    filtered_eeg.append(butter_bandpass_filter(np.asarray(eeg[0]), lowcut_eeg, highcut_eeg, fs_eeg, order))
    nsamples = len(eeg[0])
    t = np.linspace(0, 0.5, nsamples, endpoint=False)
    plt.plot(t, eeg_i, label='Filtered signal Hz')
    plt.savefig("filterd.pdf")

#filter ecg
filtered_ecg = []
lowcut_ecg = 0.02
highcut_ecg = 40.0
fs_ecg = 256.0
order = 20
for ecg_i in ecg:
    nsamples = len(ecg_i)
    t = np.linspace(0, T, nsamples, endpoint=False)
    plt.plot(t, ecg_i, label='Filtered signal Hz')
    filtered_ecg.append(butter_bandpass_filter(np.asarray(ecg_i), lowcut_ecg, highcut_ecg, fs_ecg, order))
    nsamples = len(ecg_i)
    t = np.linspace(0, T, nsamples, endpoint=False)
plt.plot(t, ecg_i, label='Filtered signal Hz')

# feature extraction eeg
fs_eeg = 128.0
band = [8,12,14,18]
feature_eeg = []
feature_ch_eeg = []
for eeg_i in filtered_eeg:
    feature_ch_eeg.append(pyeeg.bin_power(np.asarray(eeg_i), band, fs_eeg)[0])
for j in range(0,len(feature_ch_eeg)-1,14):
    feature_temp = []
    feature_temp = np.concatenate((feature_ch_eeg[j], feature_ch_eeg[j+1], feature_ch_eeg[j+2], feature_ch_eeg[j+3], feature_ch_eeg[j+4], feature_ch_eeg[j+5], feature_ch_eeg[j+6], feature_ch_eeg[j+7], feature_ch_eeg[j+8], feature_ch_eeg[j+9], feature_ch_eeg[j+10], feature_ch_eeg[j+11], feature_ch_eeg[j+12], feature_ch_eeg[j+13]))
    feature_eeg.append(feature_temp)

import biosppy.signals.ecg as becg

# feature extraction ecg
fs_ecg = 256.0
feature_ecg = []
k=0
for i in range(0,len(ecg)-1,2):
    feature_row = []
    feature_row.append(np.mean(np.diff(becg.ecg(ecg[i], sampling_rate=fs_ecg, show=False)['rpeaks']))*1000/fs_ecg)
    feature_row.append(np.median(np.diff(becg.ecg(ecg[i], sampling_rate=fs_ecg, show=False)['rpeaks']))*1000/fs_ecg)
    feature_row.append(np.std(np.diff(becg.ecg(ecg[i], sampling_rate=fs_ecg, show=False)['rpeaks']))*1000/fs_ecg)
    feature_row.append(np.diff(becg.ecg(ecg[i], sampling_rate=fs_ecg, show=False)['rpeaks']).min())
    feature_row.append(np.diff(becg.ecg(ecg[i], sampling_rate=fs_ecg, show=False)['rpeaks']).max())
    feature_row.append(np.mean(np.diff(becg.ecg(ecg[i], sampling_rate=fs_ecg, show=False)['heart_rate'])))
    feature_row.append(np.mean(np.diff(becg.ecg(ecg[i], sampling_rate=fs_ecg, show=False)['heart_rate'])))
    feature_row.append(np.mean(np.diff(becg.ecg(ecg[i], sampling_rate=fs_ecg, show=False)['heart_rate'])))
    feature_row.append(np.diff(becg.ecg(ecg[i], sampling_rate=fs_ecg, show=False)['heart_rate']).min())
    feature_row.append(np.diff(becg.ecg(ecg[i], sampling_rate=fs_ecg, show=False)['heart_rate']).max())
    feature_row.append(becg.ecg(ecg[i], sampling_rate=fs_ecg, show=False)['heart_rate'].max())
    feature_row.append(np.mean(np.diff(becg.ecg(ecg[i+1], sampling_rate=fs_ecg, show=False)['rpeaks']))*1000/fs_ecg)
    feature_row.append(np.median(np.diff(becg.ecg(ecg[i+1], sampling_rate=fs_ecg, show=False)['rpeaks']))*1000/fs_ecg)
    feature_row.append(np.std(np.diff(becg.ecg(ecg[i+1], sampling_rate=fs_ecg, show=False)['rpeaks']))*1000/fs_ecg)
    feature_row.append(np.diff(becg.ecg(ecg[i+1], sampling_rate=fs_ecg, show=False)['rpeaks']).min())
    feature_row.append(np.diff(becg.ecg(ecg[i+1], sampling_rate=fs_ecg, show=False)['rpeaks']).max())
    feature_row.append(np.mean(np.diff(becg.ecg(ecg[i+1], sampling_rate=fs_ecg, show=False)['heart_rate'])))
    feature_row.append(np.mean(np.diff(becg.ecg(ecg[i+1], sampling_rate=fs_ecg, show=False)['heart_rate'])))
    feature_row.append(np.mean(np.diff(becg.ecg(ecg[i+1], sampling_rate=fs_ecg, show=False)['heart_rate'])))
    feature_row.append(np.diff(becg.ecg(ecg[i+1], sampling_rate=fs_ecg, show=False)['heart_rate']).min())
    feature_row.append(np.diff(becg.ecg(ecg[i+1], sampling_rate=fs_ecg, show=False)['heart_rate']).max())
    feature_row.append(becg.ecg(ecg[i+1], sampling_rate=fs_ecg, show=False)['heart_rate'].max())
    feature_ecg.append(feature_row)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
feature_ecg = scaler.fit_transform(feature_ecg)

feature_eeg = np.array(feature_eeg)
feature_ecg = np.array(feature_ecg)

from sklearn.model_selection import train_test_split
ecg_train, ecg_test, arousal_train, arousal_test = train_test_split(feature_ecg, arousal_array, test_size = 0.25, random_state =0)
eeg_train, eeg_test, valence_train, valence_test = train_test_split(feature_eeg, valence_array, test_size = 0.25, random_state =0)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

svm_model=SVC(C=1000,gamma='auto',kernel='poly',degree=2)
svm_model.fit(ecg_train,arousal_train)
print("Training Score: {:.3f}".format(svm.score(ecg_train,arousal_train)))
print("Test score: {:.3f}".format(svm.score(ecg_test,arousal_test)))

svm_predict = svm_model.predict(ecg_test)

svm_model=SVC(C=1000,gamma='auto',kernel='poly',degree=2)
svm_model.fit(eeg_train,valence_train)
print("Training Score: {:.3f}".format(svm.score(eeg_train,valence_train)))
print("Test score: {:.3f}".format(svm.score(eeg_test,valence_test)))

svm_predict = svm_model.predict(eeg_test)







