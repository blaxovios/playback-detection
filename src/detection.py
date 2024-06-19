import cv2
import pytube
from PIL import Image
import face_recognition
import moviepy.editor as mp
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import librosa
import librosa.display
import logging

from src.functions import GeneralFunctions


class PlaybackDetection(object, GeneralFunctions):
    def __init__(self) -> None:
        self.local_logger()
        self.configs_dict = self.load_toml('configs/config.toml')
    
    def download_song(self) -> None:
        # Download song
        youtube = pytube.YouTube(self.configs_dict['GENERAL']['SONG_YOUTUBE_URL'])
        video = youtube.streams.get_highest_resolution()
        video.download(self.configs_dict['GENERAL']['VIDEO_DOWNLOAD_PATH'])
        logging.info('Downloaded successfully') # Success

    def audio_analysis(self) -> np.array:
        # Extract audio
        vid_non = mp.VideoFileClip(self.configs_dict['GENERAL']['VIDEO_DOWNLOAD_PATH'])
        vid_non.audio.write_audiofile(self.configs_dict['GENERAL']['VIDEOS_EXTRACTED_AUDIO_PATH'])

        # Start audio analysis
        y, sr = librosa.load(self.configs_dict['GENERAL']['VIDEOS_EXTRACTED_AUDIO_PATH'])
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

        # Extract only vocals
        new_y = librosa.istft(S_foreground*phase)
        sf.write(self.configs_dict['GENERAL']['VIDEOS_EXTRACTED_AUDIO_VOCAL_PATH'], new_y, sr)
        logging.info('Vocals Separated') # Success

        # Create boolean list from audio file with vocals with values representing if he sings or not.
        x , sr = librosa.load(self.configs_dict['GENERAL']['VIDEOS_EXTRACTED_AUDIO_VOCAL_PATH'], sr=None)

        spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
        # Computing the time variable for visualization
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)

        audio_check_list = []
        centr_list = self.normalize(spectral_centroids).tolist()
        for elem in centr_list:
            if elem > 0.2:
                audio_check_list.append(1)
            else:
                audio_check_list.append(0)

        # Extract frames
        vidcap = cv2.VideoCapture(self.configs_dict['GENERAL']['VIDEO_DOWNLOAD_PATH'])
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(self.configs_dict['GENERAL']['VIDEOS_EXTRACTED_IMAGES_FRAMES_PATH'] + 'frame%d.jpg' % count, image)     # save frame as JPEG file
            success,image = vidcap.read()
            count += 1
        logging.info('Frames were extracted as images') # Success

        # To make face_recognition work, see more at: https://github.com/ageitgey/face_recognition/issues/175#issue-257710508

        # Save full paths of images from video frames in a list and use it because because face_recognition
        # requires full filepath loaded whe nwe try to load_image_file
        full_file_paths = self.get_filepaths(self.configs_dict['GENERAL']['VIDEOS_EXTRACTED_IMAGES_FRAMES_PATH'])

        # Create boolean list with values representing if mouth is open or not.
        mouth_check_list = []
        logging.info('The absolute paths of the images have filled a list') # Success

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

                        top_lip_height = self.get_lip_height(top_lip)
                        bottom_lip_height = self.get_lip_height(bottom_lip)
                        mouth_height = self.get_mouth_height(top_lip, bottom_lip)

                        # if mouth is open more than lip height * ratio, return true.
                        ratio = 0.5
                        if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
                            mouth_check_list.append(1)
                        else:
                            mouth_check_list.append(0)
                        
                    # singer open mouth
                except IndexError:
                    logging.info('No mouth found')
            else:
                continue
        logging.info('Face recognition is complete') # Success

        # Check whether video is playback or not by boolean array equality
        a = np.array(audio_check_list)
        b = np.array(mouth_check_list)
        res = np.array(a == b)
        if res == False:
            logging.info('Video is not playback')
        else:
            logging.info('Video is playback')
        return res
    
    
if __name__ == '__main__':
    playback_detection = PlaybackDetection()
    playback_detection.download_song()
    playback_detection.audio_analysis()