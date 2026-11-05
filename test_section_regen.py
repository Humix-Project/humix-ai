import unittest

import torch

from audio_output import LIMITER_THRESHOLD
from section_regen import fit_length, merge_section, plan_section, repeat_melody, select_melody

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


if __name__ == '__main__':
    unittest.main()
