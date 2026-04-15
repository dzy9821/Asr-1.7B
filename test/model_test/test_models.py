"""
模型测试：验证 VAD 和 ITN 包装器的关键行为。
"""
import os
import sys
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models.itn.itn_wrapper import ITNProcessor
from models.vad.silero_vad_wrapper import SileroVAD


class TestSileroVAD(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.audio = np.random.randn(self.sample_rate * 2).astype(np.float32)
        self.vad = SileroVAD()

    def test_model_file_exists(self):
        self.assertTrue(os.path.exists(self.vad.model_path), self.vad.model_path)

    def test_get_speech_segments_shape_and_bounds(self):
        segments = self.vad.get_speech_segments(self.audio, self.sample_rate)
        self.assertIsInstance(segments, list)

        prev_end = -1
        for segment in segments:
            self.assertIsInstance(segment, tuple)
            self.assertEqual(len(segment), 2)
            start, end = segment
            self.assertIsInstance(start, int)
            self.assertIsInstance(end, int)
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, len(self.audio))
            self.assertLess(start, end)
            self.assertGreaterEqual(start, prev_end)
            prev_end = end

    def test_get_speech_segments_with_real_wav(self):
        wav_path = os.path.join(
            os.path.dirname(__file__),
            "../../models/vad/silero-vad-onnx/example/0.wav",
        )
        wav_path = os.path.abspath(wav_path)
        self.assertTrue(os.path.exists(wav_path), wav_path)

        audio, sr = sf.read(wav_path)
        segments = self.vad.get_speech_segments(audio, sr)

        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)
        for start, end in segments:
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, len(audio))
            self.assertLess(start, end)


class TestITNProcessor(unittest.TestCase):
    def setUp(self):
        self.itn = ITNProcessor()

    def test_process_returns_non_empty_string(self):
        text = "今天天气很好"
        result = self.itn.process(text)
        self.assertIsInstance(result, str)
        self.assertTrue(result)

    def test_process_performs_real_normalization(self):
        result_percent = self.itn.process("百分之九十五")
        self.assertEqual(result_percent, "95%")

        result_number = self.itn.process("二点五平方电线")
        self.assertIn("2.5", result_number)
        self.assertNotEqual(result_number, "二点五平方电线")

    def test_model_path_is_set(self):
        self.assertTrue(self.itn.model_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
