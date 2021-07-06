import pytube

# Download playback song
url = 'https://www.youtube.com/watch?v=XzW07ro2uSk'
path='C:\\Users\\tsepe\\Downloads\\Video'
youtube = pytube.YouTube(url)
video = youtube.streams.get_highest_resolution()
video.download(path)
print('Downloaded successfully') # Success

# Download non-playback song
url = 'https://www.youtube.com/watch?v=dOOxlVUC08Q'
path='C:\\Users\\tsepe\\Downloads\\Video'
youtube = pytube.YouTube(url)
video = youtube.streams.get_highest_resolution()
video.download(path)
print('Downloaded successfully') # Success