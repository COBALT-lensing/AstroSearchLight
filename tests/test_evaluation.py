"""Offline contract tests: no downloads, checkpoints, or optional simulator."""

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import numpy as np

from searchlight.evaluation import (
    EvaluationManifest, EvaluationSample, cnn_score, score_cnn_paths,
    score_matched_filter,
)
from searchlight.matched_filtering import MatchedFilterTemplate, detect_matched_filter_peaks


def sample(**kwargs):
    defaults = dict(sample_id="a", source="a.fits", suite="simulated",
                    split="test", group_id="system-a", label="lensing",
                    label_provenance="offline fixture")
    defaults.update(kwargs)
    return EvaluationSample(**defaults)


class ManifestTests(unittest.TestCase):
    def test_manifest_fixture_loads_and_unknown_schema_fails(self):
        path = Path(__file__).parent / "fixtures/evaluation_manifest.json"
        self.assertEqual(EvaluationManifest.read(path).samples[0].sample_id, "sim-system-001")
        with self.assertRaisesRegex(ValueError, "schema_version"):
            EvaluationManifest((), schema_version=2).validate()

    def test_roundtrip_preserves_unknown_labels_and_metadata(self):
        manifest = EvaluationManifest((sample(label=None, label_provenance=None),),
                                      metadata={"seed": 23})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest.write(path)
            self.assertEqual(EvaluationManifest.read(path), manifest)

    def test_rejects_group_and_tic_leakage(self):
        base = sample(tic="TIC 000123")
        for derivative in (
            sample(sample_id="b", split="tuning"),
            sample(sample_id="b", split="tuning", group_id="other", tic="123"),
        ):
            with self.subTest(derivative=derivative), self.assertRaisesRegex(ValueError, "leakage"):
                EvaluationManifest((base, derivative)).validate()

    def test_invalid_manifests(self):
        cases = [
            (sample(), sample()),
            (sample(split="validation"),),
            (sample(label_provenance=None),),
            (sample(cadence_seconds=float("nan")),),
            (sample(parent_id="missing"),),
            (sample(parent_id="a"),),
            (sample(parent_id="b"), sample(sample_id="b", parent_id="a")),
            (sample(), sample(sample_id="b", group_id="other", parent_id="a")),
        ]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                EvaluationManifest(rows).validate()

    def test_parent_control_and_injection_share_partition(self):
        EvaluationManifest((sample(), sample(sample_id="injected", parent_id="a",
                                            parameters={"seed": 7}))).validate()


class ScoringTests(unittest.TestCase):
    def test_stable_softmax_and_class_order(self):
        result = cnn_score("a", ["flare", "lensing"], [1000, 1002], positive_class="lensing")
        self.assertAlmostEqual(result.score, 1 / (1 + np.exp(-2)))
        swapped = cnn_score("a", ["lensing", "flare"], [1002, 1000], positive_class="lensing")
        self.assertEqual(result.score, swapped.score)
        self.assertEqual(result.logits, (1000, 1002))

    def test_rejects_bad_logits_and_mapping(self):
        for classes, logits, positive in [
            (["x", "x"], [1, 2], "x"), (["x"], [1], "lensing"),
            (["x"], [np.nan], "x"), (["x"], [1, 2], "x"),
        ]:
            with self.subTest(classes=classes, logits=logits), self.assertRaises(ValueError):
                cnn_score("a", classes, logits, positive_class=positive)

    def test_offline_manifest_to_both_component_scores(self):
        class FakeCNN:
            classes = ["other", "lensing"]

            def classify_paths(self, paths):
                return {path: ["lensing", -2, 2] for path in paths}

        rows = (sample(), replace(sample(), sample_id="b"))
        manifest = EvaluationManifest(rows).validate()
        scores = score_cnn_paths(FakeCNN(), manifest.samples, positive_class="lensing")
        self.assertEqual([s.sample_id for s in scores], ["a", "b"])
        self.assertGreater(scores[0].score, 0.9)
        times = np.arange(101.)
        relative_time = np.arange(-10., 11.)
        signal = np.exp(-0.5 * (relative_time / 2)**2)
        template = MatchedFilterTemplate(signal, relative_time)
        flux = np.ones_like(times)
        flux[40:61] += signal
        result = score_matched_filter("a", times, flux, [template])
        self.assertEqual(result.sample_id, "a")
        self.assertEqual(result.peak_time, 50.)
        self.assertEqual(result.template_index, 0)
        # Same production correlation and normalization as accepted detections.
        detections = detect_matched_filter_peaks(times, flux, [template], backend="cpu")
        self.assertAlmostEqual(result.score, max(d.score for d in detections))
        # Rejected by detection thresholds, but still has a continuous score.
        self.assertEqual(detect_matched_filter_peaks(times, flux, [template],
                                                    height=1e10, backend="cpu"), [])
        baseline = score_matched_filter("b", times, np.ones_like(times), [template])
        self.assertEqual(baseline.score, 0.)
        self.assertGreater(result.score, baseline.score)

    def test_gap_and_no_templates_fail_explicitly(self):
        template = MatchedFilterTemplate(np.array([0., 1., 0.]), np.arange(-1., 2.))
        with self.assertRaisesRegex(ValueError, "regular"):
            score_matched_filter("a", [0, 1, 3, 4], [1, 2, 1, 1], [template])
        with self.assertRaisesRegex(ValueError, "No eligible"):
            score_matched_filter("a", np.arange(10.), np.ones(10), [])

    def test_long_template_and_invalid_inputs(self):
        template = MatchedFilterTemplate(np.exp(-np.arange(-20., 21.)**2 / 8),
                                         np.arange(-20., 21.))
        result = score_matched_filter("a", np.arange(10.), [1]*5 + [2] + [1]*4, [template])
        self.assertTrue(np.isfinite(result.score))
        self.assertIn(result.peak_time, np.arange(10.))
        with self.assertRaises(ValueError):
            score_matched_filter("a", np.arange(10.), np.ones(9), [template])
        with self.assertRaisesRegex(ValueError, "at least two"):
            score_matched_filter("a", [0.], [1.], [template])

    def test_bank_maximum_matches_independent_correlation_reference(self):
        times = np.arange(31.)
        flux = np.random.default_rng(23).normal(size=len(times))
        templates = [
            MatchedFilterTemplate(np.exp(-np.arange(-3., 4.)**2 / scale),
                                  np.arange(-3., 4.))
            for scale in (1., 4.)
        ]
        reference = []
        for template in templates:
            centered = template.flux - template.flux.mean()
            normalized = centered / np.linalg.norm(centered)
            reference.append(np.correlate(flux - flux.mean(), normalized, mode="valid"))
        bank = np.array(reference)
        winner, offset = np.unravel_index(np.argmax(bank), bank.shape)
        result = score_matched_filter("a", times, flux, templates)
        self.assertAlmostEqual(result.score, bank[winner, offset])
        self.assertEqual(result.template_index, winner)
        self.assertEqual(result.peak_time, times[offset + 3])


class MetricTests(unittest.TestCase):
    def test_tied_scores_are_grouped(self):
        from searchlight.evaluation.metrics import binary_metrics
        result = binary_metrics([0, 1, 0, 1], [0.5]*4, threshold=0.5)
        self.assertEqual(result["roc_auc"], 0.5)
        self.assertEqual(result["average_precision"], 0.5)
        self.assertEqual(result["confusion_matrix"], [[0, 2], [0, 2]])
        self.assertEqual(result["curve"]["thresholds"], [None, 0.5])

    def test_perfect_ranking_and_tuning_budget(self):
        from searchlight.evaluation.metrics import binary_metrics, select_threshold
        truth, scores = [0, 1, 0, 1], [0.1, 0.8, 0.2, 0.9]
        threshold = select_threshold(truth, scores, max_fpr=0)
        result = binary_metrics(truth, scores, threshold=threshold)
        self.assertEqual(threshold, 0.8)
        self.assertEqual(result["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual(result["roc_auc"], 1.)
        self.assertEqual(result["average_precision"], 1.)

    def test_no_class_and_reject_all_are_json_safe(self):
        import json
        from searchlight.evaluation.metrics import binary_metrics, select_threshold
        result = binary_metrics([0, 0], [1, 2], threshold=None)
        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["precision"])
        self.assertEqual(result["confusion_matrix"], [[2, 0], [0, 0]])
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            select_threshold([0, 0], [1, 2])
        with self.assertRaises(ValueError):
            binary_metrics([0, 2], [1, 2], threshold=1)


class RunnerTests(unittest.TestCase):
    def test_cnn_run_and_unrelated_peaks_do_not_count_as_recovery(self):
        from unittest.mock import patch
        from searchlight.evaluation.runner import evaluate, load_arrays
        from searchlight.evaluation.simulations import create_demo, demo_templates
        from searchlight.evaluation.reporting import read_run
        from searchlight.matched_filtering import MatchedFilterMatch

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest = create_demo(directory / "data", tuning_groups=2, test_groups=2)
            # Avoid optional FITS/model dependencies while exercising runner joins.
            mapping = {s.source + ".fits": load_arrays(s.source) for s in manifest.samples}
            rows = []
            for sample in manifest.samples:
                path = Path(sample.source + ".fits")
                path.write_bytes(Path(sample.source).read_bytes())
                rows.append(replace(sample, source=str(path)))
            manifest = EvaluationManifest(tuple(rows), metadata=manifest.metadata)

            class FakeCNN:
                MODEL_SAVE_PATH = directory / "weights.pth"
                CLASS_SAVE_PATH = directory / "classes.json"
                classes = ["other", "lensing"]
                split_sizes = [np.inf, 7, 2]
                max_segments_per_item = 24

                def classify_paths(self, paths):
                    return {path: ["lensing", 0., 1.] for path in paths}

            FakeCNN.MODEL_SAVE_PATH.write_bytes(b"offline fixture")
            FakeCNN.CLASS_SAVE_PATH.write_text('["other", "lensing"]')
            unrelated = MatchedFilterMatch(0, 0, 0, 1, 0., 0., 0.001, 10., 1.)
            with patch("searchlight.evaluation.runner.load_arrays", side_effect=lambda source: mapping[source]), \
                 patch("searchlight.matched_filtering.detect_matched_filter_peaks", return_value=[unrelated]):
                evaluate(manifest, demo_templates(), directory / "result", cnn=FakeCNN(), max_fpr=1.)
            result = read_run(directory / "result")
            self.assertIn("cnn", result["metrics"])
            self.assertTrue(all("cnn" in row for row in result["scores"]))
            truths = [row for row in result["events"] if row["recovered"] is not None]
            self.assertTrue(truths)
            self.assertFalse(any(row["recovered"] for row in truths))
            self.assertTrue(all(row["unmatched_detections"] == 1 for row in truths))

    def test_saved_outputs_and_test_labels_do_not_change_threshold(self):
        import json
        from searchlight.evaluation.runner import evaluate
        from searchlight.evaluation.simulations import create_demo, demo_templates
        from searchlight.evaluation.reporting import read_run, recovery_table

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest = create_demo(directory / "data", tuning_groups=3, test_groups=3)
            output = directory / "results"
            metrics = evaluate(manifest, demo_templates(), output)
            loaded = read_run(output)
            self.assertEqual(metrics, loaded["metrics"])
            self.assertEqual(len(loaded["scores"]), 12)
            self.assertEqual(len(loaded["run"]["source_hashes"]), 12)
            self.assertTrue((output / "roc_precision_recall.png").exists())
            self.assertTrue((output / "confusion_counts.pdf").exists())
            self.assertEqual(sum(row["injected"] for row in recovery_table(loaded)), 3)
            with self.assertRaises(FileExistsError):
                evaluate(manifest, demo_templates(), output)
            # Change the test population; tuning threshold must stay identical.
            changed = EvaluationManifest(tuple(replace(s, label="noise", parameters={})
                if s.split == "test" and s.sample_id.endswith("-injected") and "0000" in s.sample_id
                else s for s in manifest.samples), metadata=manifest.metadata)
            evaluate(changed, demo_templates(), directory / "changed")
            second = json.loads((directory / "changed/run.json").read_text())
            self.assertEqual(loaded["run"]["thresholds"], second["thresholds"])

    def test_generator_is_seeded_and_groups_do_not_cross_splits(self):
        from searchlight.evaluation.simulations import create_demo
        from searchlight.evaluation.runner import load_arrays
        with tempfile.TemporaryDirectory() as directory:
            first = create_demo(Path(directory) / "first", seed=8, tuning_groups=1, test_groups=1)
            second = create_demo(Path(directory) / "second", seed=8, tuning_groups=1, test_groups=1)
            for a, b in zip(first.samples, second.samples):
                self.assertEqual(a.parameters, b.parameters)
                np.testing.assert_array_equal(load_arrays(a.source)[1], load_arrays(b.source)[1])


if __name__ == "__main__":
    unittest.main()
