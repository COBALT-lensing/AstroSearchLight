import unittest

import numpy as np
import pytest

from searchlight.matched_filtering import (
    MatchedFilterTemplateLibrary,
    MatchedFilterTemplate,
    _UNSET,
    _resolve_threshold_sigmas,
    _trim_template_edges_to_length,
    detect_matched_filter_peaks,
    detect_matched_filter_peaks_batched,
    load_template_library,
    save_template_library,
)


class MatchedFilteringTest(unittest.TestCase):
    def test_detects_template_and_returns_light_curve_window(self):
        time = np.arange(200.0)
        template_time = np.arange(-20.0, 20.5, 0.5)
        template_flux = np.exp(-0.5 * (template_time / 4.0) ** 2)
        resampled_template_time = np.arange(-20.0, 21.0)
        resampled_template_flux = np.interp(
            resampled_template_time,
            template_time,
            template_flux,
        )
        flux = np.ones_like(time)
        start_index = 80
        flux[
            start_index : start_index + len(resampled_template_flux)
        ] += 0.2 * resampled_template_flux

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [
                MatchedFilterTemplate(
                    template_flux,
                    template_time,
                )
            ],
            prominence=0.1,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].template_index, 0)
        self.assertEqual(matches[0].start_index, 93)
        self.assertEqual(matches[0].peak_index, 100)
        self.assertEqual(matches[0].end_index, 107)
        self.assertAlmostEqual(matches[0].start_time, 93.613628, places=6)
        self.assertEqual(matches[0].peak_time, time[100])
        self.assertAlmostEqual(matches[0].end_time, 106.386372, places=6)

    def test_limits_matches_by_score_across_template_bank(self):
        time = np.arange(200.0)
        template_time = np.arange(-10.0, 11.0)
        narrow = np.exp(-0.5 * (template_time / 2.0) ** 2)
        wide = np.exp(-0.5 * (template_time / 5.0) ** 2)
        flux = np.ones_like(time)
        flux[90 : 90 + len(narrow)] += 0.2 * narrow

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [
                MatchedFilterTemplate(wide, template_time),
                MatchedFilterTemplate(narrow, template_time),
            ],
            prominence=0.1,
            min_score_sigma=None,
            max_matches=1,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].template_index, 1)

    def test_dedupes_overlapping_template_matches_by_default(self):
        time = np.arange(200.0)
        template_time = np.arange(-10.0, 11.0)
        narrow = np.exp(-0.5 * (template_time / 2.0) ** 2)
        wide = np.exp(-0.5 * (template_time / 5.0) ** 2)
        flux = np.ones_like(time)
        flux[90 : 90 + len(narrow)] += 0.2 * narrow

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [
                MatchedFilterTemplate(wide, template_time),
                MatchedFilterTemplate(narrow, template_time),
            ],
            prominence=0.1,
            min_score_sigma=None,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].template_index, 1)

    def test_can_return_overlapping_template_matches(self):
        time = np.arange(200.0)
        template_time = np.arange(-10.0, 11.0)
        narrow = np.exp(-0.5 * (template_time / 2.0) ** 2)
        wide = np.exp(-0.5 * (template_time / 5.0) ** 2)
        flux = np.ones_like(time)
        flux[90 : 90 + len(narrow)] += 0.2 * narrow

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [
                MatchedFilterTemplate(wide, template_time),
                MatchedFilterTemplate(narrow, template_time),
            ],
            prominence=0.1,
            min_score_sigma=None,
            dedupe_overlaps=False,
        )

        self.assertEqual(len(matches), 2)

    def test_positive_template_rejects_negative_flux_excursion_shoulders(self):
        time = np.arange(200.0)
        template_time = np.arange(-10.0, 11.0)
        template_flux = np.exp(-0.5 * (template_time / 2.0) ** 2)
        flux = np.ones_like(time)
        flux[40 : 40 + len(template_flux)] += 0.2 * template_flux
        flux[90 : 90 + len(template_flux)] -= 0.4 * template_flux
        flux[140 : 140 + len(template_flux)] += 0.15 * template_flux

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [MatchedFilterTemplate(template_flux, template_time)],
            prominence=0.01,
            min_score_sigma=None,
            dedupe_overlaps=False,
        )

        self.assertEqual([match.peak_index for match in matches], [50, 150])

    def test_negative_centered_templates_are_ignored(self):
        time = np.arange(100.0)
        template_time = np.arange(-5.0, 6.0)
        positive_template = np.exp(-0.5 * (template_time / 1.5) ** 2)
        negative_template = -positive_template
        flux = np.ones_like(time)
        flux[40 : 40 + len(positive_template)] += 0.2 * positive_template

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [
                MatchedFilterTemplate(negative_template, template_time),
                MatchedFilterTemplate(positive_template, template_time),
            ],
            prominence=0.1,
            min_score_sigma=None,
            dedupe_overlaps=False,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].template_index, 1)

    def test_can_allow_negative_centered_template_matches(self):
        time = np.arange(100.0)
        template_time = np.arange(-5.0, 6.0)
        positive_template = np.exp(-0.5 * (template_time / 1.5) ** 2)
        negative_template = -positive_template
        flux = np.ones_like(time)
        flux[40 : 40 + len(negative_template)] += 0.2 * negative_template

        default_matches = detect_matched_filter_peaks(
            time,
            flux,
            [MatchedFilterTemplate(negative_template, template_time)],
            prominence=0.1,
            min_score_sigma=None,
        )
        negative_matches = detect_matched_filter_peaks(
            time,
            flux,
            [MatchedFilterTemplate(negative_template, template_time)],
            prominence=0.1,
            min_score_sigma=None,
            allow_negative_peaks=True,
        )

        self.assertEqual(default_matches, [])
        self.assertEqual(len(negative_matches), 1)
        self.assertEqual(negative_matches[0].peak_index, 45)

    def test_long_template_over_flat_light_curve_has_no_peaks(self):
        matches = detect_matched_filter_peaks(
            np.arange(5.0),
            np.ones(5),
            [MatchedFilterTemplate(np.arange(6.0), np.arange(6.0))],
            prominence=0.1,
        )

        self.assertEqual(matches, [])

    def test_pads_short_light_curve_to_search_full_long_template(self):
        time = np.arange(10.0)
        template_time = np.arange(-10.0, 11.0)
        template_flux = np.exp(-0.5 * (template_time / 2.0) ** 2)
        flux = np.ones_like(time)
        flux += 0.2 * np.exp(-0.5 * ((time - 4.0) / 2.0) ** 2)

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [MatchedFilterTemplate(template_flux, template_time)],
            prominence=0.01,
            min_score_sigma=None,
            max_matches=1,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].template_index, 0)
        self.assertEqual(matches[0].peak_index, 4)
        self.assertGreaterEqual(matches[0].start_index, 0)
        self.assertLessEqual(matches[0].end_index, len(time) - 1)
        self.assertEqual(matches[0].peak_time, 4.0)
        self.assertGreaterEqual(matches[0].start_time, time[0])
        self.assertLessEqual(matches[0].end_time, time[-1])

    def test_rejects_padded_peak_centers_outside_observation(self):
        time = np.arange(10.0)
        template_time = np.arange(-10.0, 11.0)
        template_flux = np.exp(-0.5 * (template_time / 2.0) ** 2)
        flux = np.ones_like(time)
        flux[-3:] += 0.2 * template_flux[10:13]

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [MatchedFilterTemplate(template_flux, template_time)],
            prominence=0.01,
            min_score_sigma=None,
            dedupe_overlaps=False,
        )

        self.assertTrue(all(0 <= match.peak_index < len(time) for match in matches))
        self.assertTrue(all(match.end_time <= time[-1] for match in matches))

    def test_clips_reported_match_window_to_observation(self):
        time = np.arange(10.0)
        template_time = np.arange(-5.0, 6.0)
        template_flux = np.exp(-0.5 * (template_time / 1.5) ** 2)
        flux = np.ones_like(time)
        flux[6:] += 0.2 * template_flux[5:9]

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [MatchedFilterTemplate(template_flux, template_time)],
            prominence=0.01,
            min_score_sigma=None,
            max_matches=1,
        )

        self.assertEqual(len(matches), 1)
        self.assertLessEqual(matches[0].end_index, len(time) - 1)
        self.assertLessEqual(matches[0].end_time, time[-1])

    def test_permissive_default_score_threshold_keeps_low_absolute_score_peak(self):
        time = np.arange(200.0)
        template_time = np.arange(-10.0, 11.0)
        template_flux = np.exp(-0.5 * (template_time / 2.0) ** 2)
        flux = np.ones_like(time)
        flux[90 : 90 + len(template_flux)] += 0.01 * template_flux

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [MatchedFilterTemplate(template_flux, template_time)],
            prominence=0.001,
        )

        self.assertEqual(len(matches), 1)

    def test_can_request_stricter_score_threshold(self):
        time = np.arange(200.0)
        template_time = np.arange(-10.0, 11.0)
        template_flux = np.exp(-0.5 * (template_time / 2.0) ** 2)
        flux = np.ones_like(time)
        flux[90 : 90 + len(template_flux)] += 0.01 * template_flux

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [MatchedFilterTemplate(template_flux, template_time)],
            prominence=0.001,
            min_score_sigma=8.0,
        )

        self.assertEqual(matches, [])

    def test_can_disable_default_score_threshold(self):
        time = np.arange(200.0)
        template_time = np.arange(-10.0, 11.0)
        template_flux = np.exp(-0.5 * (template_time / 2.0) ** 2)
        flux = np.ones_like(time)
        flux[90 : 90 + len(template_flux)] += 0.01 * template_flux

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [MatchedFilterTemplate(template_flux, template_time)],
            prominence=0.001,
            min_score_sigma=None,
        )

        self.assertEqual(len(matches), 1)


def test_template_library_validates_shape_and_iterates_templates():
    parameters = np.asarray([(2.0,), (3.0,)], dtype=[("tau_eff", "f8")])
    library = MatchedFilterTemplateLibrary(
        flux=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        time=np.asarray([-0.5, 0.5]),
        parameters=parameters,
        metadata={"window_scale": 1.5},
    )

    assert len(library) == 2
    templates = list(library)
    assert np.array_equal(templates[0].flux, [1.0, 2.0])
    assert np.array_equal(templates[1].time, [-0.5, 0.5])
    assert templates[0].event_half_width == 3.0
    assert templates[1].event_half_width == 4.5

    with pytest.raises(ValueError, match="two-dimensional"):
        MatchedFilterTemplateLibrary(flux=np.asarray([1.0, 2.0]), time=[1.0, 2.0])


def test_trim_template_edges_to_length_keeps_center_samples():
    assert np.array_equal(_trim_template_edges_to_length(np.arange(7), 3), [2, 3, 4])
    assert np.array_equal(_trim_template_edges_to_length(np.arange(6), 3), [1, 2, 3])
    assert np.array_equal(_trim_template_edges_to_length(np.arange(3), 6), [0, 1, 2])


def test_resolve_threshold_sigmas_prefers_library_calibration_when_unset():
    metadata = {
        "calibration": {
            "selected": {
                "prominence_sigma": 2.5,
                "min_score_sigma": 3.0,
            }
        }
    }

    assert _resolve_threshold_sigmas(metadata, _UNSET, _UNSET) == (2.5, 3.0)
    assert _resolve_threshold_sigmas(metadata, 3.0, None) == (3.0, None)


def test_save_and_load_template_library_round_trips_arrays(tmp_path):
    path = tmp_path / "templates.npz"
    parameters = np.asarray([(4.5, 10.0)], dtype=[("a", "f8"), ("Macc", "f8")])

    saved_path = save_template_library(
        path,
        flux=np.asarray([[1.0, 1.2, 1.0]]),
        time=np.asarray([-1.0, 0.0, 1.0]),
        parameters=parameters,
        metadata={"source": "test"},
    )
    library = load_template_library(saved_path)

    assert saved_path == path
    assert np.array_equal(library.flux, [[1.0, 1.2, 1.0]])
    assert np.array_equal(library.time, [-1.0, 0.0, 1.0])
    assert np.array_equal(library.parameters, parameters)
    assert library.metadata == {"source": "test"}


def test_detect_can_load_templates_from_library_path(tmp_path):
    template_time = np.arange(-10.0, 11.0)
    template_flux = np.exp(-0.5 * (template_time / 2.0) ** 2)
    library_path = save_template_library(
        tmp_path / "templates.npz",
        flux=np.asarray([template_flux]),
        time=template_time,
    )
    time = np.arange(100.0)
    flux = np.ones_like(time)
    flux[40 : 40 + len(template_flux)] += 0.2 * template_flux

    matches = detect_matched_filter_peaks(
        time,
        flux,
        template_library_path=library_path,
        prominence=0.1,
        min_score_sigma=None,
    )

    assert len(matches) == 1
    assert matches[0].start_index == 46
    assert matches[0].peak_index == 50
    assert matches[0].end_index == 54


def test_library_tau_eff_limits_reported_match_window(tmp_path):
    template_time = np.arange(-10.0, 11.0)
    template_flux = np.exp(-0.5 * (template_time / 2.0) ** 2)
    parameters = np.asarray([(2.0,)], dtype=[("tau_eff", "f8")])
    library_path = save_template_library(
        tmp_path / "templates.npz",
        flux=np.asarray([template_flux]),
        time=template_time,
        parameters=parameters,
        metadata={"window_scale": 1.0},
    )
    time = np.arange(100.0)
    flux = np.ones_like(time)
    flux[40 : 40 + len(template_flux)] += 0.2 * template_flux

    matches = detect_matched_filter_peaks(
        time,
        flux,
        template_library_path=library_path,
        prominence=0.1,
        min_score_sigma=None,
    )

    assert len(matches) == 1
    assert matches[0].peak_index == 50
    assert matches[0].start_index == 48
    assert matches[0].end_index == 52


def test_batched_detector_matches_single_curve_results():
    time = np.arange(200.0)
    template_time = np.arange(-10.0, 11.0)
    template_flux = np.exp(-0.5 * (template_time / 2.0) ** 2)
    flux_a = np.ones_like(time)
    flux_b = np.ones_like(time)
    flux_a[40 : 40 + len(template_flux)] += 0.2 * template_flux
    flux_b[120 : 120 + len(template_flux)] += 0.2 * template_flux
    templates = [MatchedFilterTemplate(template_flux, template_time)]

    batched_matches = detect_matched_filter_peaks_batched(
        time,
        np.stack([flux_a, flux_b]),
        templates,
        prominence=0.1,
        min_score_sigma=None,
    )

    single_a = detect_matched_filter_peaks(
        time,
        flux_a,
        templates,
        prominence=0.1,
        min_score_sigma=None,
    )
    single_b = detect_matched_filter_peaks(
        time,
        flux_b,
        templates,
        prominence=0.1,
        min_score_sigma=None,
    )

    assert batched_matches == [single_a, single_b]
    assert [matches[0].peak_index for matches in batched_matches] == [50, 130]


def test_batched_detector_can_chunk_light_curves():
    time = np.arange(120.0)
    template_time = np.arange(-5.0, 6.0)
    template_flux = np.exp(-0.5 * (template_time / 1.5) ** 2)
    flux = np.ones((3, len(time)))
    for curve_index, start_index in enumerate([20, 50, 80]):
        flux[curve_index, start_index : start_index + len(template_flux)] += (
            0.2 * template_flux
        )

    unchunked = detect_matched_filter_peaks_batched(
        time,
        flux,
        [MatchedFilterTemplate(template_flux, template_time)],
        prominence=0.1,
        min_score_sigma=None,
    )
    chunked = detect_matched_filter_peaks_batched(
        time,
        flux,
        [MatchedFilterTemplate(template_flux, template_time)],
        prominence=0.1,
        min_score_sigma=None,
        batch_size=2,
    )

    assert chunked == unchunked
    assert [matches[0].peak_index for matches in chunked] == [25, 55, 85]


def test_detect_accepts_two_dimensional_flux_for_batched_search():
    time = np.arange(80.0)
    template_time = np.arange(-5.0, 6.0)
    template_flux = np.exp(-0.5 * (template_time / 1.5) ** 2)
    flux = np.ones((2, len(time)))
    flux[0, 20 : 20 + len(template_flux)] += 0.2 * template_flux
    flux[1, 50 : 50 + len(template_flux)] += 0.2 * template_flux

    matches = detect_matched_filter_peaks(
        time,
        flux,
        [MatchedFilterTemplate(template_flux, template_time)],
        prominence=0.1,
        min_score_sigma=None,
    )

    assert len(matches) == 2
    assert [curve_matches[0].peak_index for curve_matches in matches] == [25, 55]


def test_batched_detector_rejects_mismatched_flux_width():
    with pytest.raises(ValueError, match="same length as time"):
        detect_matched_filter_peaks_batched(
            np.arange(10.0),
            np.ones((2, 9)),
            [MatchedFilterTemplate(np.ones(3), np.arange(3.0))],
        )


def test_batched_detector_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        detect_matched_filter_peaks_batched(
            np.arange(10.0),
            np.ones((2, 10)),
            [MatchedFilterTemplate(np.ones(3), np.arange(3.0))],
            batch_size=0,
        )


def test_detector_rejects_negative_match_window_rel_height():
    with pytest.raises(ValueError, match="match_window_rel_height must be non-negative"):
        detect_matched_filter_peaks(
            np.arange(10.0),
            np.ones(10),
            [MatchedFilterTemplate(np.arange(3.0), np.arange(3.0))],
            match_window_rel_height=-0.1,
        )


def test_detect_rejects_templates_and_library_path_together(tmp_path):
    with pytest.raises(ValueError, match="either templates or template_library_path"):
        detect_matched_filter_peaks(
            np.arange(10.0),
            np.ones(10),
            [MatchedFilterTemplate(np.ones(3), np.arange(3.0))],
            template_library_path=tmp_path / "templates.npz",
        )


if __name__ == "__main__":
    unittest.main()
