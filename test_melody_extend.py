import unittest
from unittest.mock import patch

import melody_extend
from melody_correct import correct_extension
from melody_extend import extend_melody, pad_to


def note(pitch, onset, duration):
    return {'pitch': pitch, 'onset_seconds': onset, 'duration_seconds': duration}


# 4-second humming in 60~64 (range with margin: 58~66), then a rest
HUMMING = [note(60, 0.0, 1.0), note(64, 1.0, 1.0), note(62, 2.0, 1.5), note(0, 3.5, 0.5)]


class TestCorrectExtension(unittest.TestCase):
    def test_humming_part_is_unchanged(self):
        vectors = HUMMING + [note(80, 4.0, 1.0)]
        self.assertEqual(correct_extension(vectors, 4.0)[:4], HUMMING)

    def test_extension_is_moved_by_octaves_into_humming_range(self):
        corrected = correct_extension(HUMMING + [note(76, 4.0, 1.0), note(48, 5.0, 1.0), note(63, 6.0, 1.0)], 4.0)
        self.assertEqual([v['pitch'] for v in corrected[4:]], [64, 60, 63])

    def test_only_short_gaps_after_humming_are_filled(self):
        vectors = HUMMING + [note(60, 4.0, 0.5), note(62, 4.7, 0.5), note(64, 5.6, 0.5)]
        corrected = correct_extension(vectors, 4.0)
        # 0.2 s gap filled, 0.4 s gap kept; the humming's own gaps are not touched
        self.assertEqual(corrected[4]['duration_seconds'], 0.7)
        self.assertEqual(corrected[5]['duration_seconds'], 0.5)
        self.assertEqual(corrected[:4], HUMMING)

    def test_input_is_not_modified(self):
        vectors = HUMMING + [note(76, 4.0, 0.5), note(62, 4.6, 0.5)]
        before = [dict(v) for v in vectors]
        correct_extension(vectors, 4.0)
        self.assertEqual(vectors, before)


class TestExtendMelody(unittest.TestCase):
    def test_humming_close_to_song_length_is_not_extended(self):
        with patch.object(melody_extend, 'load_amt_model') as load:
            melody, end = extend_melody(HUMMING, 4.5)
        load.assert_not_called()
        self.assertEqual((melody, end), (HUMMING, 4.0))

    @patch.object(melody_extend, 'load_amt_model', return_value=object())
    @patch.object(melody_extend, 'vectors_to_events', return_value=[])
    @patch.object(melody_extend, '_amt_generate', return_value=[])
    @patch.object(melody_extend, 'events_to_vectors', return_value=[note(76, 4.0, 2.0), note(62, 6.1, 1.0)])
    def test_extension_is_corrected_and_padded(self, *mocks):
        melody, end = extend_melody(HUMMING, 10.0)
        self.assertEqual(end, 4.0)
        self.assertEqual(melody[:4], HUMMING)
        # Folded into range, 0.1 s gap filled, then a rest until the song length
        self.assertEqual(melody[4:], [note(64, 4.0, 2.1), note(62, 6.1, 1.0), note(0, 7.1, 2.9)])

    @patch.object(melody_extend, 'load_amt_model', side_effect=RuntimeError("no GPU memory"))
    def test_failure_falls_back_to_the_humming(self, _):
        melody, end = extend_melody(HUMMING, 30.0)
        self.assertEqual((melody, end), (HUMMING, 4.0))

    def test_start_time_seconds_is_accepted(self):
        humming = [{'pitch': 60, 'onset_seconds': None, 'start_time_seconds': 0.0, 'duration_seconds': 30.0}]
        melody, end = extend_melody(humming, 30.0)
        self.assertEqual(end, 30.0)
        self.assertEqual(melody[0]['onset_seconds'], 0.0)

    def test_pad_to(self):
        self.assertEqual(pad_to(HUMMING, 4.0), HUMMING)
        self.assertEqual(pad_to(HUMMING, 6.0)[-1], note(0, 4.0, 2.0))


if __name__ == '__main__':
    unittest.main()
