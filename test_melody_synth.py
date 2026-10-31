import unittest

import numpy as np

from melody_synth import AMPLITUDE, synthesize_melody

SR = 16000


def note(pitch, onset, duration):
    return {'pitch': pitch, 'onset_seconds': onset, 'duration_seconds': duration}


def rms(audio, start_s, end_s):
    segment = audio[round(start_s * SR):round(end_s * SR)]
    return float(np.sqrt(np.mean(segment ** 2)))


class TestSynthesizeMelody(unittest.TestCase):
    def test_adjacent_notes_do_not_break(self):
        # A held note that wavers by a semitone: 62 -> 63 -> 62 back to back
        audio = synthesize_melody([note(62, 0.0, 0.2), note(63, 0.2, 0.2), note(62, 0.4, 0.2)], SR)
        # Around each boundary the tone keeps its full level (sine RMS = A / sqrt(2))
        for boundary in (0.2, 0.4):
            self.assertAlmostEqual(rms(audio, boundary - 0.01, boundary + 0.01), AMPLITUDE / np.sqrt(2), delta=0.02)
        # Continuous phase: no sample-to-sample jump larger than the steepest sine slope
        max_step = 2 * np.pi * 440 * 2 ** ((63 - 69) / 12) / SR * AMPLITUDE
        self.assertLessEqual(np.abs(np.diff(audio)).max(), max_step * 1.01)

    def test_rest_is_silent(self):
        audio = synthesize_melody([note(60, 0.0, 0.3), note(0, 0.3, 0.4), note(60, 0.7, 0.3)], SR)
        self.assertEqual(np.abs(audio[round(0.31 * SR):round(0.69 * SR)]).max(), 0.0)
        self.assertGreater(rms(audio, 0.1, 0.2), 0.3)
        self.assertGreater(rms(audio, 0.8, 0.9), 0.3)

    def test_phrase_edges_fade(self):
        audio = synthesize_melody([note(60, 0.0, 0.3), note(0, 0.3, 0.2), note(60, 0.5, 0.3)], SR)
        self.assertEqual(audio[0], 0.0)
        self.assertLess(abs(audio[round(0.3 * SR) - 1]), 0.01)
        self.assertLess(abs(audio[round(0.5 * SR)]), 0.01)

    def test_tiny_gap_is_held(self):
        # A 5 ms dropout between notes stays one continuous phrase
        audio = synthesize_melody([note(60, 0.0, 0.3), note(60, 0.305, 0.3)], SR)
        self.assertGreater(rms(audio, 0.295, 0.315), 0.3)

    def test_length_includes_trailing_rest(self):
        audio = synthesize_melody([note(60, 0.0, 1.0), note(0, 1.0, 0.5)], SR)
        self.assertEqual(len(audio), round(1.5 * SR))
        self.assertEqual(audio.dtype, np.float32)

    def test_start_time_seconds_fallback(self):
        audio = synthesize_melody([{'pitch': 60, 'start_time_seconds': 0.5, 'duration_seconds': 0.5}], SR)
        self.assertEqual(len(audio), SR)
        self.assertEqual(np.abs(audio[:round(0.49 * SR)]).max(), 0.0)

    def test_only_rests_raises(self):
        with self.assertRaises(ValueError):
            synthesize_melody([note(0, 0.0, 1.0)], SR)

    def test_frequency_matches_pitch(self):
        audio = synthesize_melody([note(69, 0.0, 1.0)], SR)
        spectrum = np.abs(np.fft.rfft(audio))
        peak_hz = np.argmax(spectrum) * SR / len(audio)
        self.assertAlmostEqual(peak_hz, 440.0, delta=1.0)


if __name__ == '__main__':
    unittest.main()
