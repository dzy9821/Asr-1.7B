"""
Pipeline 测试：验证主流程中的调用链和拼接逻辑。
"""
import os
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import asr_pipeline


class TestASRPipeline(unittest.TestCase):
    def test_vad_split_audio_returns_vad_output(self):
        vad_model = Mock()
        vad_model.get_speech_segments.return_value = [(0, 1600), (2000, 3200)]
        audio = np.zeros(4000, dtype=np.float32)

        segments = asr_pipeline.vad_split_audio(audio, 16000, vad_model)

        self.assertEqual(segments, [(0, 1600), (2000, 3200)])
        vad_model.get_speech_segments.assert_called_once_with(audio, 16000)

    def test_main_runs_end_to_end_and_concatenates_results(self):
        audio = np.arange(3200, dtype=np.float32)
        segments = [(0, 1600), (1600, 3200)]

        with patch.object(asr_pipeline, "SileroVAD") as mock_vad_cls, patch.object(
            asr_pipeline, "ITNProcessor"
        ) as mock_itn_cls, patch.object(asr_pipeline, "load_audio") as mock_load, patch.object(
            asr_pipeline, "asr_recognize"
        ) as mock_asr:
            mock_vad = Mock()
            mock_vad.get_speech_segments.return_value = segments
            mock_vad_cls.return_value = mock_vad

            mock_itn = Mock()
            mock_itn.process.side_effect = ["甲", "乙"]
            mock_itn_cls.return_value = mock_itn

            mock_load.return_value = (audio, 16000)
            mock_asr.side_effect = ["text1", "text2"]

            result = asr_pipeline.main("dummy.wav")

        self.assertEqual(result, "甲乙")
        mock_vad.get_speech_segments.assert_called_once_with(audio, 16000)
        self.assertEqual(mock_asr.call_count, 2)
        self.assertEqual(mock_itn.process.call_count, 2)
        mock_load.assert_called_once_with("dummy.wav")


if __name__ == "__main__":
    unittest.main(verbosity=2)
