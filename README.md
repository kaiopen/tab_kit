# 可通行区域边缘数据集工具套件

[English](./README_EN.md)

一个用于读取、可视化 [TAB](https://github.com/kaiopen/tab) 数据以及评估检测模型的工具。

### Python 环境
- Matplotlib
- [PyTorch](https://pytorch.org)
- [KaiTorch](https://github.com/kaiopen/kaitorch)

### 安装
在 `tab_kit/` 路径运行以下指令：
```shell
pip install .
```

### 功能
1. 读取 TAB
```python
from kaitorch.pcd import PointCloudXYZIR
from tab import TAB


tab = TAB('path/to/TAB', 'train')

# 逐帧读取数据
for f in tab:
    # 读取点云
    pcd: PointCloudXYZIR = tab.get_pcd(f)

    # 裁减点云
    # 由于数据集提供的点云是裁减过的，所以该步骤非必要
    pcd = TAB.filter_pcd_(pcd)

    print(pcd.xyz)
    print(pcd.i)
    print(pcd.r)

    # 读取真值标签
    bounds = tab.get_bound(f)
    for bound in bounds:
        print(bound.keys())

```

更多样例可见。

2. 可视化 TAB 数据
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

更多样例可见。

3. 评估预测结果
```python
from tab import Evaluator


evaluator = Evaluator('path/to/TAB', 'test')

# 获取预测结果
# preds = ...

print(Evaluator.tabulate(*evaluator(preds)))

```

详细样例可见[`./test/eval.py`](./test/eval.py)；更多样例可见。