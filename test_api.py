import sys
import os
os.environ["TESTING"] = "true"
import unittest
from unittest.mock import patch, MagicMock

# Mock out heavy ML modules so tests can run instantly and offline
class MockModule(MagicMock):
    __path__ = []

sys.modules['torch'] = MockModule()
sys.modules['torchaudio'] = MockModule()
sys.modules['boto3'] = MockModule()
sys.modules['omegaconf'] = MockModule()
sys.modules['numpy'] = MockModule()
sys.modules['scipy'] = MockModule()
sys.modules['scipy.signal'] = MockModule()
sys.modules['librosa'] = MockModule()
sys.modules['audiocraft'] = MockModule()
sys.modules['audiocraft.models'] = MockModule()
sys.modules['audiocraft.models.loaders'] = MockModule()
sys.modules['audiocraft.models.musicgen'] = MockModule()
sys.modules['audiocraft.models.builders'] = MockModule()

from fastapi.testclient import TestClient

# Import the FastAPI app (which will now use mocked imports)
import app

class TestAIServerAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app.app)
        # AMT extension (#25) is tested in test_melody_extend; here the humming is "extended" by a rest
        patcher = patch('app.extend_melody', side_effect=lambda vectors, duration: (
            vectors + [{"pitch": 0, "onset_seconds": 1.3, "duration_seconds": duration - 1.3}], 1.3))
        self.mock_extend = patcher.start()
        self.addCleanup(patcher.stop)
        # Unit tests never call Gemini, even with GEMINI_API_KEY set: the default "rewrite" strategy then
        # falls back to simple ("pop, happy, with electric guitars"). LLM tests patch generate_text themselves.
        gemini = patch('gemini_client.generate_text', side_effect=RuntimeError("no Gemini in unit tests"))
        gemini.start()
        self.addCleanup(gemini.stop)
        
    @patch('app.load_model')
    @patch('app.convert_vectors_to_wav_tensor')
    @patch('app.save_output_wav')
    @patch('app.upload_via_presigned_url')
    @patch('requests.post')
    def test_generate_songs_endpoint(self, mock_post, mock_upload_presigned, mock_save, mock_convert, mock_load_model):
        mock_convert.return_value = MagicMock()
        mock_upload_presigned.return_value = None
        
        # Setup mock model returning a 10-second 32kHz output
        mock_output = MagicMock()
        mock_output.cpu.return_value.shape = (1, 320000)
        mock_model = MagicMock()
        mock_model.generate_with_chroma.return_value = [mock_output]
        app.model = mock_model

        response = self.client.post("/internal/v1/ai/generation/songs", json=self.generation_payload())

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json(), {"task_id": "task_123"})
        mock_model.set_generation_params.assert_called_with(duration=10.0)
        mock_model.generate_with_chroma.assert_called_once()
        self.assertEqual(mock_model.generate_with_chroma.call_args.args[0], ["pop, happy, with electric guitars"])
        mock_post.assert_called_with(
            "https://backend.com/api/v1/internal/callbacks/generation",
            json={
                "generated_audio_url": "https://mock-bucket.s3.amazonaws.com/generated/songs/task_123.wav",
                "duration_seconds": 10.0,
                "extended_melody_vectors": [
                    {"pitch": 60, "onset_seconds": 0.0, "start_time_seconds": None, "duration_seconds": 0.5},
                    {"pitch": 0, "onset_seconds": 0.5, "start_time_seconds": None, "duration_seconds": 0.3},
                    {"pitch": 64, "onset_seconds": 0.8, "start_time_seconds": None, "duration_seconds": 0.5},
                    {"pitch": 0, "onset_seconds": 1.3, "duration_seconds": 8.7}
                ],
                "humming_end_seconds": 1.3
            },
            timeout=10
        )

    @patch('app.load_model')
    @patch('app.convert_vectors_to_wav_tensor')
    @patch('app.upload_via_presigned_url')
    @patch('requests.post')
    def test_generate_songs_sends_failed_callback_when_melody_conversion_fails(self, mock_post, mock_upload_presigned, mock_convert, mock_load_model):
        mock_convert.side_effect = ValueError("melody_vectors has no pitched notes.")
        mock_model = MagicMock()
        app.model = mock_model

        response = self.client.post("/internal/v1/ai/generation/songs", json=self.generation_payload())

        self.assertEqual(response.status_code, 202)
        mock_model.generate_with_chroma.assert_not_called()
        mock_upload_presigned.assert_not_called()
        mock_post.assert_called_once_with(
            "https://backend.com/api/v1/internal/callbacks/generation",
            json={"generated_audio_url": "FAILED"},
            timeout=10
        )

    @patch('app.load_model')
    @patch('app.convert_vectors_to_wav_tensor')
    @patch('app.save_output_wav')
    @patch('app.upload_via_presigned_url')
    @patch('requests.post')
    def test_generate_songs_saves_requested_length(self, mock_post, mock_upload_presigned, mock_save, mock_convert, mock_load_model):
        # audiocraft generates whole 1/50 s frames; the saved file is cut or padded to the request (#21)
        mock_output = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_with_chroma.return_value = [mock_output]
        app.model = mock_model
        payload = self.generation_payload()
        payload["duration_seconds"] = 10.01

        with patch('app.fit_length', side_effect=lambda wav, n: wav) as mock_fit:
            self.client.post("/internal/v1/ai/generation/songs", json=payload)

        self.assertEqual(mock_fit.call_args.args[1], 320320)
        self.assertEqual(mock_post.call_args.kwargs["json"]["duration_seconds"], 10.01)

    def modification_payload(self):
        return {
            "task_id": "task_456",
            "melody_vectors": [{"pitch": 60, "onset_seconds": 0.0, "duration_seconds": 12.0}],
            "genre": "pop",
            "mood": "happy",
            "prompt": "with piano",
            "text_condition": "pop, happy, with piano",
            "edit_prompt": "more strings",
            "humming_end_seconds": 12.0,
            "source_audio_url": "https://mock-bucket.s3.amazonaws.com/audio/source.wav",
            "section_start_seconds": 10,
            "section_end_seconds": 20,
            "callback_url": "https://backend.com/api/v1/internal/callbacks/generation",
            "presigned_url": "https://mock-bucket.s3.amazonaws.com/audio/new_variation.wav?X-Amz-Signature=abc"
        }

    @patch('app.load_model')
    @patch('app.load_source_audio')
    @patch('app.regenerate_section')
    @patch('app.write_wav')
    @patch('app.upload_via_presigned_url')
    @patch('requests.post')
    def test_modify_regenerates_section(self, mock_post, mock_upload, mock_write, mock_regen, mock_source, mock_load_model):
        merged = MagicMock()
        merged.shape = (1, 960000)
        mock_regen.return_value = merged
        app.model = MagicMock()

        response = self.client.post("/internal/v1/ai/generation/songs/song_1/modifications",
                                    json=self.modification_payload())

        self.assertEqual(response.status_code, 202)
        args = mock_regen.call_args
        self.assertEqual(args.args[1], "pop, happy, with piano, more strings")
        self.assertEqual(args.args[4:6], (10, 20))
        self.assertEqual(args.kwargs["humming_end_seconds"], 12.0)
        mock_source.assert_called_once_with("https://mock-bucket.s3.amazonaws.com/audio/source.wav")
        mock_post.assert_called_with(
            "https://backend.com/api/v1/internal/callbacks/generation",
            json={
                "generated_audio_url": "https://mock-bucket.s3.amazonaws.com/audio/new_variation.wav",
                "duration_seconds": 30.0,
                "extended_melody_vectors": [
                    {"pitch": 60, "onset_seconds": 0.0, "start_time_seconds": None, "duration_seconds": 12.0}],
                "humming_end_seconds": 12.0,
                "text_condition": "pop, happy, with piano, more strings"
            },
            timeout=10
        )

    @patch('app.load_model')
    @patch('app.load_source_audio')
    @patch('app.regenerate_section')
    @patch('requests.post')
    def test_modify_sends_failed_callback_on_invalid_section(self, mock_post, mock_regen, mock_source, mock_load_model):
        mock_regen.side_effect = ValueError("section must be at least 5.0 seconds")
        app.model = MagicMock()

        self.client.post("/internal/v1/ai/generation/songs/song_1/modifications", json=self.modification_payload())

        mock_post.assert_called_once_with(
            "https://backend.com/api/v1/internal/callbacks/generation",
            json={"generated_audio_url": "FAILED"},
            timeout=10
        )

    def test_modify_rejects_missing_fields(self):
        for field in ("source_audio_url", "section_start_seconds", "section_end_seconds", "melody_vectors"):
            payload = self.modification_payload()
            del payload[field]
            response = self.client.post("/internal/v1/ai/generation/songs/song_1/modifications", json=payload)
            self.assertEqual(response.status_code, 422, field)

    def test_generate_songs_rejects_invalid_duration(self):
        for duration in (None, 0, -1, 30.5):
            payload = self.generation_payload()
            if duration is None:
                del payload["duration_seconds"]
            else:
                payload["duration_seconds"] = duration
            response = self.client.post("/internal/v1/ai/generation/songs", json=payload)
            self.assertEqual(response.status_code, 422, f"duration_seconds={duration}")

    def test_build_simple_description(self):
        simple = dict(strategy="simple")
        self.assertEqual(app.build_description("pop", "happy", None, **simple), "pop, happy")
        self.assertEqual(app.build_description("pop", "happy", "  ", **simple), "pop, happy")
        self.assertEqual(app.build_description("pop", "happy", "with piano", **simple), "pop, happy, with piano")
        # modify can arrive without genre/mood
        self.assertEqual(app.build_description("", None, "with piano", **simple), "with piano")
        with self.assertRaises(ValueError):
            app.build_description("pop", "happy", None, strategy="unknown")

    @patch('gemini_client.generate_text')
    def test_default_rewrite_without_prompt_uses_templates_only(self, mock_generate):
        # The default strategy is "rewrite" (#31); without a prompt it needs no LLM call
        self.assertEqual(
            app.build_description("pop", "upbeat", None),
            "modern pop track, polished production, punchy drums, electric bass, bright synths, piano, "
            "upbeat, energetic and cheerful",
        )
        self.assertEqual(
            app.build_description("Jazz", "CALM", "  "),
            "jazz piece, live band recording, upright bass, brushed drums, piano comping, saxophone, "
            "calm, relaxing and gentle",
        )
        # Values without a template pass through; empty fields are skipped
        self.assertEqual(app.build_description("k-pop", "happy", None), "k-pop, happy")
        self.assertEqual(app.build_description("", None, None), "")
        mock_generate.assert_not_called()

    @patch('gemini_client.generate_text')
    def test_llm_strategies(self, mock_generate):
        mock_generate.return_value = "acoustic guitar, soft drums"
        self.assertEqual(app.build_description("pop", "upbeat", "어쿠스틱 기타로", strategy="translate"), "pop, upbeat, acoustic guitar, soft drums")
        self.assertEqual(
            app.build_description("pop", "upbeat", "어쿠스틱 기타로", strategy="rewrite"),
            "modern pop track, polished production, acoustic guitar, soft drums, upbeat, energetic and cheerful",
        )
        self.assertEqual(
            app.build_description("pop", "upbeat", "어쿠스틱 기타로", strategy="rewrite_notemplate"),
            "acoustic guitar, soft drums",
        )
        # English prompts skip translation; unknown genre/mood pass through
        mock_generate.reset_mock()
        self.assertEqual(app.build_description("k-pop", "dreamy", "with piano", strategy="translate"), "k-pop, dreamy, with piano")
        mock_generate.assert_not_called()

    @patch('gemini_client.generate_text', side_effect=RuntimeError("quota exceeded"))
    def test_llm_failure_falls_back_to_simple(self, mock_generate):
        self.assertEqual(app.build_description("pop", "sad", "감성적으로", strategy="rewrite"), "pop, sad, 감성적으로")

    def generation_payload(self):
        return {
            "task_id": "task_123",
            "melody_vectors": [
                {"pitch": 60, "onset_seconds": 0.0, "duration_seconds": 0.5},
                {"pitch": 0, "onset_seconds": 0.5, "duration_seconds": 0.3},
                {"pitch": 64, "onset_seconds": 0.8, "duration_seconds": 0.5}
            ],
            "genre": "pop",
            "mood": "happy",
            "prompt": "with electric guitars",
            "duration_seconds": 10,
            "callback_url": "https://backend.com/api/v1/internal/callbacks/generation",
            "presigned_url": "https://mock-bucket.s3.amazonaws.com/generated/songs/task_123.wav?AWSAccessKeyId=mock..."
        }

    @patch('app.processor.preprocess_audio')
    @patch('app.processor.extract_f0')
    @patch('app.processor.hz_to_midi')
    @patch('app.processor.quantize_and_map')
    def test_extract_melody_endpoint(self, mock_map, mock_hz, mock_f0, mock_preprocess):
        mock_preprocess.return_value = MagicMock()
        mock_f0.return_value = []
        mock_hz.return_value = MagicMock()
        mock_map.return_value = [
            {"start_time_seconds": 0.0, "pitch": 60, "duration_seconds": 0.5}
        ]
        
        payload = {
            "s3_url": "https://mock-bucket.s3.amazonaws.com/uploads/humming/test.wav"
        }
        
        response = self.client.post("/api/v1/ai/melody-extract", json=payload)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [
            {"start_time_seconds": 0.0, "pitch": 60, "duration_seconds": 0.5}
        ])

if __name__ == '__main__':
    unittest.main()