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
sys.modules['pretty_midi'] = MockModule()
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
        
    @patch('app.load_model')
    @patch('app.convert_vectors_to_wav_tensor')
    @patch('app.torchaudio.save')
    @patch('app.upload_via_presigned_url')
    @patch('requests.post')
    def test_generate_songs_endpoint(self, mock_post, mock_upload_presigned, mock_save, mock_convert, mock_load_model):
        mock_convert.return_value = MagicMock()
        mock_upload_presigned.return_value = None
        
        # Setup mock model
        mock_model = MagicMock()
        mock_model.generate_with_chroma.return_value = [MagicMock()]
        app.model = mock_model
        
        payload = {
            "task_id": "task_123",
            "melody_vectors": [
                {"pitch": 60, "onset_seconds": 0.0, "duration_seconds": 0.5},
                {"pitch": 64, "onset_seconds": 0.5, "duration_seconds": 0.5}
            ],
            "genre": "pop",
            "mood": "happy",
            "reference_track": "Attention",
            "callback_url": "https://backend.com/api/v1/internal/callbacks/generation",
            "presigned_url": "https://mock-bucket.s3.amazonaws.com/generated/songs/task_123.wav?AWSAccessKeyId=mock..."
        }
        
        response = self.client.post("/internal/v1/ai/generation/songs", json=payload)
        
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json(), {"task_id": "task_123"})
        
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