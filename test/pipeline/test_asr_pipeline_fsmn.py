import os
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import asr_pipeline_fsmn as pipeline_module


class TestASRPipelineFSMN(unittest.TestCase):
    def test_main_uses_fsmn_vad(self):
        with patch.object(pipeline_module, "FSMNVADWrapper") as mock_vad_cls, patch.object(
            pipeline_module, "ITNProcessor"
        ) as mock_itn_cls, patch.object(pipeline_module, "run_pipeline") as mock_run:
            mock_run.return_value = "ok"
            result = pipeline_module.main(
                "dummy.wav",
                asr_model="m",
                asr_base_url="http://localhost:8000/v1",
                asr_api_key="EMPTY",
            )

        self.assertEqual(result, "ok")
        mock_vad_cls.assert_called_once()
        mock_itn_cls.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs["audio_path"], "dummy.wav")
        self.assertEqual(kwargs["asr_model"], "m")
        self.assertEqual(kwargs["asr_base_url"], "http://localhost:8000/v1")
        self.assertEqual(kwargs["asr_api_key"], "EMPTY")
        self.assertIs(kwargs["vad_model"], mock_vad_cls.return_value)
        self.assertIs(kwargs["itn_model"], mock_itn_cls.return_value)

    def test_asr_recognize_builds_audio_url_payload(self):
        mock_client = Mock()
        mock_resp = Mock()
        mock_resp.choices = [Mock(message=Mock(content="识别结果"))]
        mock_client.chat.completions.create.return_value = mock_resp

        text = pipeline_module.asr_recognize(
            np.zeros(1600, dtype=np.float32),
            16000,
            model="Qwen3-ASR-1.7B",
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            client=mock_client,
        )
        self.assertEqual(text, "识别结果")

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "Qwen3-ASR-1.7B")
        url = kwargs["messages"][0]["content"][0]["audio_url"]["url"]
        self.assertTrue(url.startswith("data:audio/wav;base64,"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

