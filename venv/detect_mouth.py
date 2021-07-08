from PIL import Image, ImageDraw
import os
import face_recognition
import math
import cv2
from youtube_download import video_non

# Save full paths of images from video frames in a list and use it because because face_recognition
# requires full filepath loaded whe nwe try to load_image_file
directory = ('C:\\Users\\tsepe\\Downloads\\Video\\Scene Images')
def get_filepaths(directory):
    file_paths = []  # List which will store all of the full filepaths.
    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.
full_file_paths = get_filepaths(directory)

# Create a loop for face_recognition for all saved images to find open mouth
for x in full_file_paths:

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(x)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))
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

        #　pil_image.save('test.png')

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
                return True
            else:
                return False
        # singer open mouth
        print('Is mouth open:', check_mouth_open(top_lip,bottom_lip) )
    except IndexError:
        print('No mouth found')

