from __future__ import print_function
import cv2
import pytube
from PIL import Image, ImageDraw
import os
import face_recognition
import math
import moviepy.editor as mp
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

# Download playback song
url = 'https://www.youtube.com/watch?v=XzW07ro2uSk'
path='C:\\Users\\tsepe\\Downloads\\Video'
youtube = pytube.YouTube(url)
video = youtube.streams.get_highest_resolution()
video_playback = video.download(path)
print('Downloaded successfully') # Success

# Download non-playback song
url = 'https://www.youtube.com/watch?v=QiB97xZIzQY'
path='C:\\Users\\tsepe\\Downloads\\Video'
youtube = pytube.YouTube(url)
video = youtube.streams.get_highest_resolution()
video_non = video.download(path)
print('Downloaded successfully') # Success

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
print('Vocals Separated') # Success

# Extract frames
vidcap = cv2.VideoCapture(video_non)
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite('C:\\Users\\tsepe\\Downloads\\Video\\Frame Image\\'+'frame%d.jpg' % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  count += 1
print('Frames were extracted as images') # Success

# To make face_recognition work, see more at: https://github.com/ageitgey/face_recognition/issues/175#issue-257710508

# Save full paths of images from video frames in a list and use it because because face_recognition
# requires full filepath loaded whe nwe try to load_image_file
directory = ('C:\\Users\\tsepe\\Downloads\\Video\\Frame Image')
def get_filepaths(directory):
    file_paths = []  # List which will store all of the full filepaths.
    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths
full_file_paths = get_filepaths(directory)

mouth_check_list = []
print('The absolute paths of the images have filled a list') # Success

# Create a loop for face_recognition for all saved images to find open mouth
for x in full_file_paths:

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(x)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    # If you find any face in the frame image do the above. Else, continue.
    if face_landmarks_list:
        try:
            pil_image = Image.fromarray(image)

            top_lip = []
            bottom_lip = []

            for face_landmarks in face_landmarks_list:
                # Print the location of each facial feature in this image
                facial_features = [
                    'top_lip',
                    'bottom_lip'
                ]

                for facial_feature in facial_features:
                    top_lip = face_landmarks['top_lip']
                    bottom_lip = face_landmarks['bottom_lip']

            def get_lip_height(lip):
                sum=0
                for i in [2,3,4]:
                    # distance between two near points up and down
                    distance = math.sqrt( (lip[i][0] - lip[12-i][0])**2 +
                                          (lip[i][1] - lip[12-i][1])**2   )
                    sum += distance
                return sum / 3

            def get_mouth_height(top_lip, bottom_lip):
                sum=0
                for i in [8,9,10]:
                    # distance between two near points up and down
                    distance = math.sqrt( (top_lip[i][0] - bottom_lip[18-i][0])**2 +
                                          (top_lip[i][1] - bottom_lip[18-i][1])**2   )
                    sum += distance
                return sum / 3

            def check_mouth_open(top_lip, bottom_lip):
                top_lip_height =    get_lip_height(top_lip)
                bottom_lip_height = get_lip_height(bottom_lip)
                mouth_height =      get_mouth_height(top_lip, bottom_lip)

                # if mouth is open more than lip height * ratio, return true.
                ratio = 0.5
                if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
                    mouth_check_list.append(1)
                    return True
                else:
                    mouth_check_list.append(0)
                    return False
            # singer open mouth
            print('Is mouth open:', check_mouth_open(top_lip,bottom_lip) )
        except IndexError:
            print('No mouth found')
    else:
        continue
print('Face recognition is complete') # Success