"""Audit saved decisions and class mappings without rerunning either detector.

Usage: python docs/validation/review_full_evaluation.py RESULTS OUTPUT.json
The result records input hashes; source results are never rewritten.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def audit(results):
    results = Path(results)
    read = lambda name: json.loads((results / (name+'.json')).read_text())
    scores, metrics, run, manifest = (read(name) for name in ['scores','metrics','run','manifest'])
    samples = {s['sample_id']:s for s in manifest['samples']}
    counts = dict(label_mismatches=0, decision_mismatches=0, cnn_probability_mismatches=0,
                  cnn_class_mapping_mismatches=0, combined_gate_mismatches=0)
    summaries = {}
    for row in scores:
        counts['label_mismatches'] += row['truth'] != int(samples[row['sample_id']]['label']==run['positive_class'])
        cnn = row.get('cnn')
        if cnn:
            classes = cnn['classes']
            expected_classes = run['cnn']['classes']
            counts['cnn_class_mapping_mismatches'] += classes != expected_classes or cnn['positive_class'] != run['positive_class']
            logits = np.asarray(cnn['logits']); probs = np.exp(logits-logits.max()); probs /= probs.sum()
            expected = probs[classes.index(run['positive_class'])]
            counts['cnn_probability_mismatches'] += not np.isclose(cnn['score'],expected,rtol=0,atol=1e-12)
        for component in metrics:
            threshold = run['thresholds'][component]
            expected = threshold is not None and row[component]['score'] >= threshold
            counts['decision_mismatches'] += expected != row[component]['accepted']
        counts['combined_gate_mismatches'] += ('combined' in row and
            row['combined']['accepted'] != (row['matched_filter']['accepted'] and row['cnn']['accepted']))
    matrix_mismatches = 0
    for component, splits in metrics.items():
        summaries[component] = {}
        for split, stored in splits.items():
            rows = [r for r in scores if r['split']==split and r['truth'] in [0,1]]
            matrix = np.zeros((2,2),int)
            for row in rows: matrix[row['truth'],int(row[component]['accepted'])] += 1
            matrix_mismatches += not np.array_equal(matrix,stored['confusion_matrix'])
            tn,fp,fn,tp = matrix.ravel().tolist()
            summaries[component][split] = dict(controls_rejected=tn,controls_accepted=fp,
                injections_rejected=fn,injections_accepted=tp,
                all_accepted=(tn+fn==0),all_rejected=(fp+tp==0),
                false_positive_rate=fp/(tn+fp),injection_acceptance=tp/(fn+tp))
    counts['confusion_matrix_mismatches'] = matrix_mismatches
    test = [r for r in scores if r['split']=='test']
    positives = sum(r['truth']==1 for r in test)
    return dict(results=str(results.resolve()), checks=counts, decisions=summaries,
        test_injected_to_control_ratio=positives/(len(test)-positives),
        test_all_accepted_precision=positives/len(test),
        input_sha256={name:hashlib.sha256((results/(name+'.json')).read_bytes()).hexdigest()
                      for name in ['scores','metrics','run','manifest']},
        scope='Saved-score bookkeeping audit only; does not establish detector correctness or calibrated probabilities')


if __name__=='__main__':
    report = audit(sys.argv[1])
    output = Path(sys.argv[2]); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks'],indent=2))
