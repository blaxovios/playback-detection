from __future__ import print_function
import moviepy.editor as mp
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import sklearn

# Select playback and non-playback
vid_playb = mp.VideoFileClip(r"C:\\Users\\tsepe\\Downloads\\Video\\Rita Ora - How We Do (Party) backing track fail.mp4")
vid_non = mp.VideoFileClip(r"C:\\Users\\tsepe\\Downloads\\Video\\SING WITH ME CHALLENGE! ARCADE DUNCAN LAURENCE SINGING DUET shorts.mp4")
vid_playb.audio.write_audiofile(r"C:\\Users\\tsepe\\Downloads\\Video\\AUDIO--Rita Ora - How We Do (Party) backing track fail.wav")
vid_non.audio.write_audiofile(r"C:\\Users\\tsepe\\Downloads\\Video\\AUDIO--SING WITH ME CHALLENGE! ARCADE DUNCAN LAURENCE SINGING DUET shorts.wav")
audio_playb = ("C:\\Users\\tsepe\\Downloads\\Video\\AUDIO--Rita Ora - How We Do (Party) backing track fail.wav")
audio_non = ("C:\\Users\\tsepe\\Downloads\\Video\\AUDIO--SING WITH ME CHALLENGE! ARCADE DUNCAN LAURENCE SINGING DUET shorts.wav")

y, sr = librosa.load(audio_non)
# And compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))

# We'll compare frames using cosine similarity, and aggregate similar frames
# by taking their (per-frequency) median value.
#
# To avoid being biased by local continuity, we constrain similar frames to be
# separated by at least 2 seconds.
#
# This suppresses sparse/non-repetetitive deviations from the average spectrum,
# and works well to discard vocal elements.

S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))

# The output of the filter shouldn't be greater than the input
# if we assume signals are additive.  Taking the pointwise minimium
# with the input spectrum forces this.
S_filter = np.minimum(S_full, S_filter)
# We can also use a margin to reduce bleed between the vocals and instrumentation masks.
# Note: the margins need not be equal for foreground and background separation
margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Once we have the masks, simply multiply them with the input spectrum
# to separate the components

S_foreground = mask_v * S_full
S_background = mask_i * S_full
# sphinx_gallery_thumbnail_number = 2

new_y = librosa.istft(S_foreground*phase)
vocal_audio = sf.write("C:\\Users\\tsepe\\Downloads\\Video\\VOCALS ONLY--SING WITH ME CHALLENGE! ARCADE DUNCAN LAURENCE SINGING DUET shorts.wav", new_y, sr)

audio_data = "C:\\Users\\tsepe\\Downloads\\Video\\VOCALS ONLY--SING WITH ME CHALLENGE! ARCADE DUNCAN LAURENCE SINGING DUET shorts.wav"
x , sr = librosa.load(audio_data, sr=None)

spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
plt.plot(t, normalize(spectral_centroids), color='b')

audio_check_list = []
centr_list = normalize(spectral_centroids).tolist()
for elem in centr_list:
    if elem > 0.2:
        audio_check_list.append(1)
    else:
        audio_check_list.append(0)