import argparse
import random
from pathlib import Path

import torch

from tab import TAB, Evaluator


parser = argparse.ArgumentParser()
parser.add_argument('root', type=str)
root = Path(parser.parse_args().root)
tab = TAB(root, 'test')
evaluator = Evaluator(root)

preds = []
scores_p = {}
scores_l = {}
for c in tab.SEMANTICS:
    scores_p[c] = []
    scores_l[c] = {}
scores_p['w/o semantics'] = []
scores_l['w/o semantics'] = {}

num_pred = 0
num_bound = 0

for f in tab:
    bounds = []
    for bound in tab.get_boundaries(f):
        if random.random() < 0.2:
            if not bound['fuzzy']:
                scores_p[bound['semantics']].append(0)

            continue  # miss

        points = []
        for p in bound['keypoints']:
            points.append(p['xy'])

        num_bound = num_point = len(points)
        if random.random() < 0.3:
            num_point = random.randint(1, num_point)
        elif random.random() < 0.7:
            num_point = random.randint(num_point // 2, num_point)

        c = bound['semantics']
        bounds.append(
            {
                'semantics': c,
                'points': points[:num_point]
            }
        )

        if bound['fuzzy']:
            continue

        rec = num_point / num_bound
        scores_p[c].append(2 * rec / (1 + rec))  # pre = 1

    preds.append(
        {
            'sequence': f.sequence,
            'id': f.id,
            'boundaries': bounds
        }
    )
for c in tab.SEMANTICS:
    scores_p['w/o semantics'] += scores_p[c]

for c in list(tab.SEMANTICS) + ['w/o semantics']:
    _scores_p = torch.as_tensor(scores_p[c])
    num_pred = torch.sum(_scores_p != 0).item()
    num_bound = len(_scores_p)

    for t in Evaluator.T:
        m = _scores_p >= t
        tp = torch.sum(m).item()
        pre = tp / num_pred
        rec = tp / num_bound
        scores_l[c][t] = 2 * pre * rec / (pre + rec)

    scores_p[c] = torch.mean(_scores_p).item()

print('Scores should be')
print(Evaluator.tabulate(scores_p, scores_l))

print('\n------\nScores are')
print(Evaluator.tabulate(*evaluator(preds)))
