# The TAB Toolkit

[中文版](README_ZH.md)

A toolkit for reading, visualizing the [TAB](https://github.com/kaiopen/tab), a travelable area boundary dataset, and evaluating the detectors.

### Python Environment
- Matplotlib
- [PyTorch](https://pytorch.org)
- [KaiTorch](https://github.com/kaiopen/kaitorch)

### Installation
Run the following command in the `tab_kit/`:
```shell
pip install .
```

### Usage
1. Read the TAB.
```python
from kaitorch.pcd import PointCloudXYZIR
from tab import TAB


tab = TAB('path/to/TAB', 'train')

# Iterate the TAB frame by frame.
for f in tab:
    # Read the point cloud.
    pcd: PointCloudXYZIR = tab.get_pcd(f)

    # Clip the point cloud.
    # It is not needed to do so because the point cloud has already been clipped before.
    pcd = TAB.filter_pcd_(pcd)

    print(pcd.xyz)
    print(pcd.i)
    print(pcd.r)

    # Read labels.
    bounds = tab.get_bound(f)
    for bound in bounds:
        print(bound.keys())

```

More examples are available.

2. Do visualization.
```python
from tab import BEV, TAB, Visualizer


vis = Visualizer(
    BEV(mode=BEV.Mode.CONSTANT),
    width=2,
    save=True,
    dst='results'
)

tab = TAB('path/to/TAB', 'train')

for f in tqdm(tab):
    vis(f.seq, f.id, tab.get_pcd(f), tab.get_bound(f))

```

More examples are available.

3. Do evaluation.
```python
from tab import Evaluator


evaluator = Evaluator('path/to/TAB', 'test')

# Get prediction results.
# preds = ...

print(Evaluator.tabulate(*evaluator(preds)))

```

The example in detail is [`./test/eval.py`](./test/eval.py). More examples are available.