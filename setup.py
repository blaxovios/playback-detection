from setuptools import setup

setup(
    name='playback_detection',
    version='0.1',
    packages=[''],
    url='https://github.com/blaxovios/playback_detection',
    license='',
    description='',
    setup_requires=[
    ],
    install_requires=[
        'pandas',
        'scipy',
        'opencv-python',
        'numpy',
        'Pillow',
        'cmake',
        'face_recognition',
        'matplotlib',
        'scikit-learn',
        'soundfile',
        'librosa',
        'moviepy'

    ],
    extras_require={
        'dev': [
            'pip-tools',
        ],
    },
    python_requires='>=3.8',
)