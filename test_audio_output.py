import os
import tempfile
import unittest

import soundfile as sf
import torch

from audio_output import LIMITER_THRESHOLD, OUTPUT_SAMPLE_RATE, save_output_wav, soft_clip


class TestSoftClip(unittest.TestCase):
    def test_samples_at_or_below_threshold_are_unchanged(self):
        wav = torch.linspace(-LIMITER_THRESHOLD, LIMITER_THRESHOLD, 10001)
        self.assertTrue(torch.equal(soft_clip(wav), wav))

    def test_large_peaks_stay_below_one(self):
        wav = torch.tensor([1.0, 1.2, 1.564, 1.77, 2.0, -2.0, -1.77])
        limited = soft_clip(wav)
        self.assertLess(limited.abs().max().item(), 1.0)
        # Sign is kept
        self.assertTrue(torch.equal(torch.sign(limited), torch.sign(wav)))

    def test_monotonic_and_continuous_around_threshold(self):
        wav = torch.linspace(-2.0, 2.0, 40001, dtype=torch.float64)
        limited = soft_clip(wav)
        steps = torch.diff(limited)
        self.assertTrue((steps > 0).all())
        # Slope is at most 1 everywhere, so there is no jump at the threshold
        self.assertLessEqual(steps.max().item(), (wav[1] - wav[0]).item() * (1 + 1e-9))
        just_over = torch.tensor([LIMITER_THRESHOLD + 1e-6], dtype=torch.float64)
        self.assertAlmostEqual(soft_clip(just_over).item(), just_over.item(), places=9)


class TestSaveOutputWav(unittest.TestCase):
    def test_saves_pcm16_without_clipping(self):
        wav = torch.tensor([[0.0, 0.5, -0.5, 1.564, -2.0] * 6400])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.wav")
            save_output_wav(path, wav)
            info = sf.info(path)
            self.assertEqual(info.subtype, "PCM_16")
            self.assertEqual(info.samplerate, OUTPUT_SAMPLE_RATE)
            self.assertEqual(info.frames, wav.shape[-1])
            saved, _ = sf.read(path, dtype="float32")
        self.assertLess(abs(saved).max(), 1.0)
        # Samples below the threshold keep their level (up to 16-bit quantization)
        self.assertAlmostEqual(float(saved[1]), 0.5, delta=1 / 32768)


if __name__ == "__main__":
    unittest.main()
