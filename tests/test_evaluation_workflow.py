"""Scientific and orchestration invariants for the complete evaluation workflow."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from searchlight.evaluation.datasets import EvaluationManifest, EvaluationSample, audit_overlap
from searchlight.evaluation.preprocessing import LightCurve, PreprocessingConfig, preprocess, read_lightcurve
from searchlight.evaluation.matching import match_events, event_observability
from searchlight.evaluation.injections import integrate_profile, inject_signal, build_injections
from searchlight.evaluation.ingestion import assign_splits, cache_product
from searchlight.evaluation.uncertainty import (group_bootstrap, zero_event_upper_bound,
    proportion_interval)


def curve():
    time = 2457000 + np.r_[np.arange(20), np.arange(30, 50)] / 720
    return LightCurve(time, np.ones(40), np.full(40, .001), np.zeros(40, np.int64),
                      np.full(40, 120.), {"time_unit": "day", "TICID": 123})


def sample(source, **kwargs):
    values = dict(sample_id="control", source=str(source), suite="spoc_injection", split="test",
                  group_id="tic:123", tic="123", label=None)
    values.update(kwargs)
    return EvaluationSample(**values)


class PreprocessingTests(unittest.TestCase):
    def test_gaps_quality_and_trim_are_explicit(self):
        raw = curve()
        raw.quality[4] = 1
        raw.flux[8] = np.nan
        clean, segments, report = preprocess(raw, PreprocessingConfig(trim_start_hours=.01))
        self.assertEqual(report["flagged_rows"], 1)
        self.assertEqual(report["removed_nonfinite_or_invalid_exposure"], 1)
        self.assertGreaterEqual(len(segments), 3)
        self.assertTrue(all(np.diff(s.time).max() < .003 for s in segments))
        self.assertLess(len(clean.time), len(raw.time))
        self.assertAlmostEqual(report["live_time_days"], len(clean.time) / 720)

    def test_empty_and_invalid_configs_fail(self):
        raw = curve()
        raw.quality[:] = 1
        with self.assertRaisesRegex(ValueError, "usable cadences"):
            preprocess(raw)
        for config in (PreprocessingConfig(gap_factor=1), PreprocessingConfig(trim_start_hours=-1),
                       PreprocessingConfig(exposure_seconds=-1)):
            with self.assertRaises(ValueError):
                config.validate()

    def test_spoc_local_fits_and_npz_preserve_equivalent_raw_columns(self):
        from astropy.io import fits
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = curve()
            columns = [fits.Column(name="TIME", format="D", array=raw.time - 2457000),
                       fits.Column(name="QUALITY", format="J", array=raw.quality)]
            for name, multiplier in (("SAP_FLUX", 2.), ("PDCSAP_FLUX", 3.)):
                columns += [fits.Column(name=name, format="D", array=raw.flux * multiplier),
                            fits.Column(name=name + "_ERR", format="D", array=raw.flux_err * multiplier)]
            data = fits.BinTableHDU.from_columns(columns)
            data.header["BJDREFI"] = 2457000
            data.header["TIMEDEL"] = 1 / 720
            data.header["INT_TIME"] = 1.98
            data.header["NUM_FRM"] = 60
            primary = fits.PrimaryHDU()
            primary.header["TICID"] = 123
            primary.header["SECTOR"] = 1
            source = directory / "spoc.fits"
            fits.HDUList([primary, data]).writeto(source)
            loaded = read_lightcurve(source, PreprocessingConfig(flux_product="SAP_FLUX"))
            self.assertEqual(loaded.metadata["TICID"], 123)
            self.assertAlmostEqual(loaded.exposure_seconds[0], 118.8)
            self.assertEqual(loaded.metadata["exposure_source"], "INT_TIME * NUM_FRM")
            np.testing.assert_allclose(loaded.flux, 2.)
            loaded.write(directory / "copy.npz")
            reread = read_lightcurve(directory / "copy.npz")
            with self.assertRaisesRegex(ValueError, "flux product"):
                read_lightcurve(directory / "copy.npz", PreprocessingConfig(flux_product="PDCSAP_FLUX"))
            for name in ("time", "flux", "flux_err", "quality", "exposure_seconds"):
                np.testing.assert_array_equal(getattr(loaded, name), getattr(reread, name))


class InjectionTests(unittest.TestCase):
    def test_exposure_average_matches_analytic_gaussian_integral(self):
        from scipy.special import erf
        width, exposure = .001, 600.
        result = integrate_profile(np.array([0.]), np.array([exposure]),
                                   lambda t: np.exp(-.5 * (t / width)**2), nodes=31)[0]
        half = exposure / 172800
        expected = width * np.sqrt(2 * np.pi) * erf(half / (np.sqrt(2) * width)) / (2 * half)
        self.assertAlmostEqual(result, expected, places=10)
        self.assertLess(result, 1.)

    def test_dilution_and_error_convention(self):
        raw = curve()
        injected = inject_signal(raw, lambda t: np.full(t.shape, .1), dilution=.25)
        np.testing.assert_allclose(injected.flux, 1.025)
        np.testing.assert_array_equal(injected.flux_err, raw.flux_err)
        np.testing.assert_array_equal(raw.flux, 1.)
        with self.assertRaises(ValueError):
            inject_signal(raw, lambda t: t, dilution=2)

    def test_paired_multi_event_injections_seed_and_unknown_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = curve()
            raw.write(directory / "raw.npz")
            manifest = EvaluationManifest((sample(directory / "raw.npz"),), metadata={"time_unit": "day"})
            one = build_injections(manifest, directory / "one", kind="gaussian", seed=1,
                                   events_per_curve=2, center_mode="gap")
            two = build_injections(manifest, directory / "two", kind="gaussian", seed=1,
                                   events_per_curve=2, center_mode="gap")
            self.assertIsNone(one.samples[0].label)
            self.assertEqual(one.samples[1].parent_id, "control")
            self.assertEqual(len(one.samples[1].parameters["events"]), 2)
            self.assertEqual(one.samples[1].parameters, two.samples[1].parameters)
            np.testing.assert_array_equal(read_lightcurve(one.samples[1].source).flux,
                                          read_lightcurve(two.samples[1].source).flux)
            self.assertEqual(one.samples[1].group_id, one.samples[0].group_id)


class MatchingTests(unittest.TestCase):
    def test_maximum_cardinality_prevents_greedy_loss(self):
        events = [{"time": 0., "tolerance_days": 1.}, {"time": 1., "tolerance_days": .3}]
        detections = [{"peak_time": .8}, {"peak_time": -.8}, {"peak_time": 9.}]
        result = match_events(events, detections)
        self.assertEqual(len(result["matches"]), 2)
        self.assertEqual(result["unmatched_detections"], [2])
        self.assertEqual(len({p["detection_index"] for p in result["matches"]}), 2)

    def test_duplicate_truth_does_not_double_count_detection(self):
        event = {"time": 0., "tolerance_days": .1}
        result = match_events([event, event], [{"peak_time": 0.}])
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(len(result["missed_events"]), 1)

    def test_gap_and_boundary_observability(self):
        raw = curve()
        _, segments, _ = preprocess(raw)
        gap = {"time": 2457000 + 25 / 720, "duration_days": 2 / 720, "tolerance_days": .01}
        self.assertFalse(event_observability(gap, segments)["observable"])
        observed = {"time": 2457000 + 10 / 720, "duration_days": 3 / 720, "tolerance_days": .01}
        self.assertTrue(event_observability(observed, segments)["observable"])


class ProvenanceTests(unittest.TestCase):
    def test_cache_detects_corruption_and_separates_products(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = directory / "raw.fits"
            raw.write_bytes(b"known bytes")
            cached = cache_product(raw, directory / "cache")
            self.assertEqual(cached.read_bytes(), b"known bytes")
            self.assertEqual(cache_product(raw, directory / "cache"), cached)
            cached.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                cache_product(raw, directory / "cache")

    def test_overlap_checks_content_even_when_identifiers_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            a, b = directory / "a", directory / "b"
            a.write_bytes(b"same")
            b.write_bytes(b"same")
            evaluation = EvaluationManifest((sample(a),))
            training = EvaluationManifest((sample(b, sample_id="other", group_id="other", tic="456", split="train"),))
            with self.assertRaisesRegex(ValueError, "overlap"):
                audit_overlap(evaluation, training)

    def test_sector_holdout_keeps_all_target_sectors_together(self):
        rows = [sample("a", sample_id="a", sector=1), sample("b", sample_id="b", sector=9)]
        assigned = assign_splits(rows, seed=123, held_out_sectors=[9])
        self.assertEqual([s.split for s in assigned], ["test", "test"])
        EvaluationManifest(assigned).validate()


class UncertaintyTests(unittest.TestCase):
    def test_group_resampling_is_invariant_to_duplicate_rows_within_groups(self):
        rows = [{"group_id": str(i), "truth": i % 2, "mf": {"score": float(i % 2)}} for i in range(4)]
        one = group_bootstrap(rows, "mf", .5, repeats=30, seed=4)
        two = group_bootstrap(rows + rows, "mf", .5, repeats=30, seed=4)
        self.assertEqual(one, two)
        self.assertEqual(one["intervals"]["recall"], [1., 1.])

    def test_count_uncertainty_bounds(self):
        upper = zero_event_upper_bound(100)
        self.assertGreater(upper, 0.02)
        self.assertLess(upper, 0.04)
        self.assertEqual(proportion_interval(0, 0), [None, None])
        self.assertGreater(proportion_interval(0, 100)[1], upper)


class DependencyMetadataTests(unittest.TestCase):
    def test_same_package_version_retains_distinct_git_revisions(self):
        from searchlight.evaluation.provenance import dependency_provenance
        distribution = Mock(version="1.2.3")
        with patch("searchlight.evaluation.provenance.importlib.metadata.distribution",
                   return_value=distribution):
            identities = []
            for revision in ("commit-a", "commit-b"):
                distribution.read_text.return_value = json.dumps({
                    "url": "https://github.com/COBALT-lensing/py-self-lensing.git",
                    "vcs_info": {"vcs": "git", "commit_id": revision}})
                identities.append(dependency_provenance("selflensing")["selflensing"])
        self.assertEqual(identities[0]["version"], identities[1]["version"])
        self.assertNotEqual(identities[0]["commit_id"], identities[1]["commit_id"])
        self.assertEqual(identities[1]["commit_id"], "commit-b")

    def test_registry_package_does_not_invent_a_source_commit(self):
        from searchlight.evaluation.provenance import dependency_provenance
        distribution = Mock(version="0.6.0")
        distribution.read_text.return_value = None
        with patch("searchlight.evaluation.provenance.importlib.metadata.distribution",
                   return_value=distribution):
            identity = dependency_provenance("rustlens")["rustlens"]
        self.assertEqual(identity, {"version": "0.6.0", "direct_url": None, "commit_id": None})


class FullWorkflowTests(unittest.TestCase):
    def test_unlabelled_control_and_multi_event_truth_with_frozen_threshold(self):
        from searchlight.evaluation.runner import evaluate
        from searchlight.evaluation.simulations import demo_templates
        from searchlight.evaluation.reporting import read_run
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = curve()
            raw.write(directory / "raw.npz")
            manifest = EvaluationManifest((sample(directory / "raw.npz"),), metadata={"time_unit": "day"})
            injected = build_injections(manifest, directory / "injected", kind="gaussian", seed=1,
                                         events_per_curve=2, center_mode="gap")
            evaluate(injected, demo_templates(), directory / "results", preprocessing={}, normalize_scores=True,
                     fixed_thresholds={"matched_filter": 0.}, bootstrap_repeats=10)
            results = read_run(directory / "results")
            self.assertIsNone(results["metrics"]["matched_filter"]["test"]["roc_auc"])
            self.assertEqual(results["event_summary"]["test"]["injected_events"], 2)
            self.assertEqual(results["candidate_rates"]["test"]["matched_filter"]["unlabelled_rows"], 1)
            intervals = results["metrics"]["matched_filter"]["test"]["uncertainty"]
            self.assertEqual(intervals["status"], "insufficient independent groups")
            self.assertEqual(intervals["intervals"]["recall"], [None, None])
            self.assertTrue((directory / "results/complete.json").exists())
            run = json.loads((directory / "results/run.json").read_text())
            self.assertEqual(run["dataset_metadata"], injected.metadata)
            self.assertTrue((directory / "results/metrics_by_suite.json").exists())
            self.assertTrue(all("control_has_candidate" in row for row in results["event_details"]))

    def test_trimmed_boundary_event_stays_in_truth_but_loses_coverage(self):
        raw = curve()
        event = {"time": raw.time[0], "duration_days": 4 / 720, "tolerance_days": .01}
        _, retained, _ = preprocess(raw, PreprocessingConfig(trim_start_hours=.3))
        self.assertFalse(event_observability(event, retained)["observable"])

    def test_same_tic_cannot_be_counted_as_different_bootstrap_groups(self):
        rows = (sample("a"), sample("b", sample_id="b", group_id="other"))
        with self.assertRaisesRegex(ValueError, "group_id"):
            EvaluationManifest(rows).validate()

    def test_config_paths_and_transfer_template_validation(self):
        from dataclasses import asdict
        from searchlight.evaluation.experiment import read_config, score
        from searchlight.evaluation.runner import template_fingerprint
        from searchlight.evaluation.simulations import demo_templates
        from searchlight.matched_filtering import _save_template_library_file as save_template_library
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = curve()
            raw.write(directory / "raw.npz")
            EvaluationManifest((sample(directory / "raw.npz"),)).write(directory / "manifest.json")
            templates = demo_templates()
            previous = {"template_sha256": template_fingerprint(templates), "normalize_scores": True,
                        "positive_class": "lensing", "preprocessing": asdict(PreprocessingConfig()),
                        "thresholds": {"matched_filter": 5.}}
            (directory / "previous.json").write_text(json.dumps(previous))
            # Different morphology must not silently inherit a calibration.
            save_template_library(directory / "templates.npz", time=templates[0].time,
                                  flux=np.array([templates[0].flux**2]))
            config = {"manifest": "manifest.json", "templates": "templates.npz", "output": "result",
                      "thresholds_from_run": "previous.json", "normalize_scores": True}
            (directory / "config.json").write_text(json.dumps(config))
            resolved = read_config(directory / "config.json")
            self.assertEqual(resolved["output"], str(directory / "result"))
            with self.assertRaisesRegex(ValueError, "identical template bank"):
                score(resolved)


class EnvelopeTests(unittest.TestCase):
    def test_continuous_envelope_matches_scalar_and_exposes_uncut_candidates(self):
        from searchlight.evaluation import matched_filter_envelope, score_matched_filter
        from searchlight.evaluation.simulations import demo_templates
        time = np.arange(512) / 720
        flux = 1 + .01 * np.exp(-.5 * ((time - .3) / .01)**2)
        templates = demo_templates()
        envelope = matched_filter_envelope(time, flux, templates)
        scalar = score_matched_filter("a", time, flux, templates)
        self.assertAlmostEqual(np.nanmax(envelope["score"]), scalar.score)
        self.assertTrue(envelope["candidates"])
        nearest = min(envelope["candidates"], key=lambda row: abs(row["peak_time"] - .3))
        self.assertGreater(nearest["prominence"], 0)
        self.assertLess(abs(nearest["peak_time"] - .3), .005)
        self.assertTrue(np.isnan(envelope["score"][0]))
        with self.assertRaisesRegex(ValueError, "regular"):
            matched_filter_envelope([0, 1, 3], [1, 2, 1], templates)


if __name__ == "__main__":
    unittest.main()
