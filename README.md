# Phi RDS

机器人数据集 **SDK**（包名 `phi_rds`）。磁盘统一 **LeRobot v3**，字段对齐[《统一格式字段及命名标准》](https://zcnce50wan15.feishu.cn/wiki/Jitbwmov2iJMQpkpy6Zc0TPMnJf)。

## 安装

```bash
cd phi_rds
pip install -e ".[lerobot]"
```

## 30 秒

```python
from phi_rds import Episode, Segment, create_l2, fill_delta_chunks, open_dataset
import numpy as np

ep = Episode.from_arrays(
    "ep-000", fps=30.0, task="pick",
    subtasks=[Segment(0, 4, "reach"), Segment(4, 8, "grasp")],
    ee=np.tile(np.eye(4), (8, 1, 1)),
    ee_action=np.tile(np.eye(4), (8, 1, 1)),
    joints=np.zeros((8, 7)),
    joints_action=np.zeros((8, 7)),
    gripper=np.ones(8),
    gripper_action=np.ones(8),
)

w = create_l2("/data/out", robot="umi", name="demo")
ep = fill_delta_chunks(ep, w.schema.layout)  # L2 必做：显式计算 delta_chunk
w.add_episode(ep)
w.finalize()

ds = open_dataset("/data/out")
print(ds.level, ds.standard_version, len(ds))
```

完整文档 → [飞书《Phi RDS SDK 完整文档》](https://zcnce50wan15.feishu.cn/docx/Bn7Xd2p8Lovp57xOHTwcw2ASnAd)  
示例 → [`examples/end_to_end.py`](examples/end_to_end.py)  
字段标准 → [飞书](https://zcnce50wan15.feishu.cn/wiki/Jitbwmov2iJMQpkpy6Zc0TPMnJf)

## L1 / L2

| | 含义 |
|--|------|
| **L1** | → LeRobot v3；命名统一；布局贴近源数据 · `create_l1`（无 delta_chunk） |
| **L2** | 完整统一标准 + 版本号 · `create_l2`（含 delta_chunk，需 `fill_delta_chunks`） |

## 目录

```text
core/     Dataset · Episode · Body
schema/   spec · presets · layout
io/       lerobot
convert/  pose · frame · delta_chunk
```
