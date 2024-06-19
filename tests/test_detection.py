import unittest
import numpy as np
import detection

from src.functions import GeneralFunctions


class TestAudioProcessing(unittest.TestCase, GeneralFunctions):
    def setUp(self):
        configs = {'GENERAL': {'VIDEO_DOWNLOAD_PATH': 'path/to/video', 'VIDEOS_EXTRACTED_AUDIO_PATH': 'path/to/audio'}}
        self.audio_processor = detection.AudioProcessor(configs)

    def test_process_audio(self):
        result = self.audio_processor.process_audio()
        self.assertIsNotNone(result)  # Assuming process_audio returns some result

    def test_audio_extraction(self):
        self.audio_processor.process_audio()
        # Assuming there's a way to verify audio was extracted, e.g., file exists

    def test_spectrogram_computation(self):
        S_full, phase = self.audio_processor.process_audio()
        self.assertIsInstance(S_full, np.ndarray)
        self.assertIsInstance(phase, np.ndarray)

    def test_audio_separation(self):
        S_foreground, S_background = self.audio_processor.process_audio()
        self.assertIsInstance(S_foreground, np.ndarray)
        self.assertIsInstance(S_background, np.ndarray)


if __name__ == '__main__':
    unittest.main()