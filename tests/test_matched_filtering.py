import unittest

import numpy as np
import pytest

from searchlight.matched_filtering import (
    MatchedFilterTemplateLibrary,
    MatchedFilterTemplate,
    _trim_template_edges_to_length,
    detect_matched_filter_peaks,
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
        self.assertEqual(matches[0].start_index, start_index)
        self.assertEqual(matches[0].peak_index, 100)
        self.assertEqual(matches[0].end_index, 120)
        self.assertEqual(matches[0].start_time, time[start_index])
        self.assertEqual(matches[0].peak_time, time[100])
        self.assertEqual(matches[0].end_time, time[120])

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

    def test_trims_template_longer_than_light_curve(self):
        matches = detect_matched_filter_peaks(
            np.arange(5.0),
            np.ones(5),
            [MatchedFilterTemplate(np.arange(6.0), np.arange(6.0))],
            prominence=0.1,
        )

        self.assertEqual(matches, [])

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
    library = MatchedFilterTemplateLibrary(
        flux=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        time=np.asarray([-0.5, 0.5]),
        parameters=np.asarray([(1.0,), (2.0,)], dtype=[("a", "f8")]),
    )

    assert len(library) == 2
    templates = list(library)
    assert np.array_equal(templates[0].flux, [1.0, 2.0])
    assert np.array_equal(templates[1].time, [-0.5, 0.5])

    with pytest.raises(ValueError, match="two-dimensional"):
        MatchedFilterTemplateLibrary(flux=np.asarray([1.0, 2.0]), time=[1.0, 2.0])


def test_trim_template_edges_to_length_keeps_center_samples():
    assert np.array_equal(_trim_template_edges_to_length(np.arange(7), 3), [2, 3, 4])
    assert np.array_equal(_trim_template_edges_to_length(np.arange(6), 3), [1, 2, 3])
    assert np.array_equal(_trim_template_edges_to_length(np.arange(3), 6), [0, 1, 2])


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
    assert matches[0].start_index == 40


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
