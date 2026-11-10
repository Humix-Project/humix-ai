import unittest

import torch

from audio_output import LIMITER_THRESHOLD
from section_regen import (build_modify_text, fit_length, merge_section, plan_section, regenerate_section,
                           repeat_melody, select_melody)

SR = 32000


def note(pitch, onset, duration):
    return {'pitch': pitch, 'onset_seconds': onset, 'duration_seconds': duration}


class TestPlanSection(unittest.TestCase):
    def test_issue_examples(self):
        # (start, end) -> (context, tail) on a 30-second song, from the #13 table
        cases = {(10, 20): (5, 1), (3, 13): (3, 1), (2, 29): (2, 1), (1, 30): (1, 0), (0, 30): (0, 0)}
        for (start, end), expected in cases.items():
            self.assertEqual(plan_section(30, start, end), expected, (start, end))

    def test_total_never_exceeds_limit(self):
        for start in range(0, 26):
            for end in range(start + 5, 31):
                context, tail = plan_section(30, start, end)
                self.assertLessEqual(context + (end - start) + tail, 30)

    def test_invalid_section(self):
        for start, end in [(-1, 10), (10, 10), (20, 10), (0, 31)]:
            with self.assertRaises(ValueError):
                plan_section(30, start, end)


class TestSelectMelody(unittest.TestCase):
    def test_new_song_takes_window_and_cuts_edges(self):
        melody = [note(60, 0.0, 4.0), note(62, 4.0, 4.0), note(0, 8.0, 2.0), note(64, 10.0, 5.0)]
        selected = select_melody(melody, 3.0, 12.0, humming_end_seconds=8.0)
        self.assertEqual(selected, [note(60, 0.0, 1.0), note(62, 1.0, 4.0), note(0, 5.0, 2.0), note(64, 7.0, 2.0)])

    def test_old_song_repeats_humming_with_its_length(self):
        # Humming of 3 s (rest included) looped: second copy starts at 3 s
        humming = [note(60, 0.0, 2.0), note(0, 2.0, 1.0)]
        self.assertEqual(repeat_melody(humming, 6.0)[2:4], [note(60, 3.0, 2.0), note(0, 5.0, 1.0)])
        selected = select_melody(humming, 4.0, 7.0)
        self.assertEqual(selected, [note(60, 0.0, 1.0), note(0, 1.0, 1.0), note(60, 2.0, 1.0)])


class TestMerge(unittest.TestCase):
    def setUp(self):
        self.source = torch.full((1, 30 * SR), 0.1)

    def test_length_and_untouched_parts(self):
        generated = torch.full((1, 11 * SR), -0.2)
        merged = merge_section(self.source, generated, 10, 20, 1)
        self.assertEqual(merged.shape, self.source.shape)
        # Before the section and after the tail the original samples are kept exactly
        self.assertTrue(torch.equal(merged[..., :10 * SR], self.source[..., :10 * SR]))
        self.assertTrue(torch.equal(merged[..., 21 * SR:], self.source[..., 21 * SR:]))
        # Inside the section (after the start crossfade) it is the generated audio
        self.assertTrue(torch.allclose(merged[..., 11 * SR:20 * SR], torch.tensor(-0.2)))

    def test_crossfades_are_continuous(self):
        generated = torch.full((1, 11 * SR), -0.2)
        merged = merge_section(self.source, generated, 10, 20, 1)
        # No step bigger than the crossfade slope at either boundary
        self.assertLess(merged[0].diff().abs().max().item(), 0.01)

    def test_only_generated_part_is_limited(self):
        source = torch.full((1, 30 * SR), 0.95)  # above the limiter threshold, as saved
        generated = torch.full((1, 11 * SR), 1.5)
        merged = merge_section(source, generated, 10, 20, 1)
        self.assertTrue(torch.equal(merged[..., :10 * SR], source[..., :10 * SR]))
        self.assertLess(merged[..., 11 * SR:20 * SR].max().item(), 1.0)
        self.assertGreater(merged[..., 11 * SR:20 * SR].max().item(), LIMITER_THRESHOLD)

    def test_section_at_song_end_has_no_tail(self):
        generated = torch.full((1, 10 * SR + 123), -0.2)  # a little long: cut to fit
        merged = merge_section(self.source, generated, 20, 30, 0)
        self.assertEqual(merged.shape, self.source.shape)
        self.assertTrue(torch.allclose(merged[..., 21 * SR:], torch.tensor(-0.2)))

    def test_fit_length(self):
        self.assertEqual(fit_length(torch.ones(1, 10), 7).shape[-1], 7)
        padded = fit_length(torch.ones(1, 10), 12)
        self.assertEqual(padded.shape[-1], 12)
        self.assertEqual(padded[0, -1].item(), 0.0)


class TestModifyText(unittest.TestCase):
    def test_text_condition_with_edit_prompt(self):
        self.assertEqual(build_modify_text("kpop, sad, soft piano", " more strings ", "fallback"),
                         "kpop, sad, soft piano, more strings")
        self.assertEqual(build_modify_text("kpop, sad", None, "fallback"), "kpop, sad")

    def test_old_song_without_text_condition_uses_fallback(self):
        self.assertEqual(build_modify_text(None, "more strings", "pop, happy, guitar"), "pop, happy, guitar")
        self.assertEqual(build_modify_text("  ", None, "pop, happy"), "pop, happy")

    def test_at_most_1000_characters(self):
        self.assertEqual(len(build_modify_text("a" * 990, "b" * 20, "")), 1000)


class FakeConditioner:
    _use_masking = False


class FakeModel:
    """Records what regenerate_section asks for and returns silence of the requested length."""
    sample_rate = SR
    audio_channels = 1
    frame_rate = 50

    def __init__(self):
        self.conditioner = FakeConditioner()
        self.lm = type('LM', (), {'condition_provider': type('CP', (), {
            'conditioners': {'self_wav': self.conditioner}})()})()
        self.calls = []

    def set_generation_params(self, duration):
        self.duration = duration

    def _record(self, name, melody):
        self.calls.append({'name': name, 'melody': melody, 'masking': self.conditioner._use_masking})
        return torch.zeros(1, 1, round(self.duration * SR))

    def generate(self, descriptions):
        return self._record('generate', None)

    def generate_with_chroma(self, descriptions, melody, sample_rate):
        return self._record('generate_with_chroma', melody)

    def _prepare_tokens_and_attributes(self, descriptions, prompt, melody_wavs):
        self.melody_wavs = melody_wavs
        return None, torch.zeros(1, 4, round(prompt.shape[-1] / SR * self.frame_rate))

    def _generate_tokens(self, attributes, prompt_tokens):
        return None

    def generate_audio(self, tokens):
        return self._record('continuation', self.melody_wavs)


def fake_synthesize(melody_vectors, sample_rate):
    import numpy as np
    end = max(v['onset_seconds'] + v['duration_seconds'] for v in melody_vectors)
    return np.full(round(end * sample_rate), 0.1, dtype=np.float32)


class TestRegenerateSection(unittest.TestCase):
    def setUp(self):
        self.source = torch.full((1, 30 * SR), 0.1)
        # Notes until 12 s, rests after (the humming ended)
        self.melody = [note(60, 0.0, 12.0), note(0, 12.0, 18.0)]

    def test_section_with_notes_uses_melody_and_keeps_length(self):
        model = FakeModel()
        merged = regenerate_section(model, "pop", self.source, self.melody, 5, 10, fake_synthesize,
                                    humming_end_seconds=12.0)
        self.assertEqual(merged.shape, self.source.shape)
        self.assertEqual(model.duration, 5 + 5 + 1)  # context 5 + section 5 + tail 1
        self.assertEqual(model.calls[0]['name'], 'continuation')
        self.assertIsNotNone(model.calls[0]['melody'])
        self.assertFalse(model.calls[0]['masking'])

    def test_rest_only_section_is_generated_without_melody(self):
        model = FakeModel()
        regenerate_section(model, "pop", self.source, self.melody, 20, 30, fake_synthesize, humming_end_seconds=12.0)
        self.assertIsNone(model.calls[0]['melody'])
        self.assertTrue(model.calls[0]['masking'])
        # Masking is restored for the next request
        self.assertFalse(model.conditioner._use_masking)

    def test_whole_song_has_no_context(self):
        model = FakeModel()
        merged = regenerate_section(model, "pop", self.source, self.melody, 0, 30, fake_synthesize,
                                    humming_end_seconds=12.0)
        self.assertEqual(model.calls[0]['name'], 'generate_with_chroma')
        self.assertEqual(model.duration, 30)
        self.assertEqual(merged.shape, self.source.shape)

    def test_section_end_slightly_past_source_is_its_end(self):
        model = FakeModel()
        merged = regenerate_section(model, "pop", self.source, self.melody, 20, 30.05, fake_synthesize,
                                    humming_end_seconds=12.0)
        self.assertEqual(merged.shape, self.source.shape)

    def test_invalid_sections(self):
        for start, end in [(10, 14.9), (10, 31), (-1, 10)]:
            with self.assertRaises(ValueError, msg=(start, end)):
                regenerate_section(FakeModel(), "pop", self.source, self.melody, start, end, fake_synthesize)


if __name__ == '__main__':
    unittest.main()
