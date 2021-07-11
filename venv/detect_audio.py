'''import librosa
import matplotlib.pyplot as plt
import librosa.display
import sklearn
import numpy as np

audio_data = "C:\\Users\\tsepe\\Downloads\\Video\\VOCALS ONLY--SING WITH ME CHALLENGE! ARCADE DUNCAN LAURENCE SINGING DUET shorts.wav"

x , sr = librosa.load(audio_data, sr=None)
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
# Computing the time variable for visualization
plt.figure(figsize=(12, 4))
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='b')

audio_check_list = []
centr_list = normalize(spectral_centroids).tolist()
for elem in centr_list:
    if elem > 0.2:
        audio_check_list.append(1)
    else:
        audio_check_list.append(0)
