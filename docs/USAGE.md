# Phi RDS 使用文档

磁盘统一 **LeRobot v3**。对外只记三件事：`Dataset` / `Episode` / `Schema(L1|L2)`。

---

## 1. 完整流程（推荐照这个）

```python
from phi_rds import Episode, create_l2, open_dataset, encode_pose, change_frame
import numpy as np

# ----- 组装一条轨迹（数组即可）-----
ep = Episode.from_arrays(
    "ep-000",
    fps=30.0,
    task="pick_cup",
    ee=np.tile(np.eye(4), (8, 1, 1)),          # (N,4,4)
    ee_action=np.tile(np.eye(4), (8, 1, 1)),
    joints=np.zeros((8, 7)),
    joints_action=np.zeros((8, 7)),
    gripper=np.ones(8),
    gripper_action=np.ones(8),
)

# ----- 转换（delta_chunk 等）→ 写出 L2（encode 只序列化）-----
from phi_rds.convert import fill_delta_chunks
w = create_l2("/data/out", robot="umi", name="demo")
ep = fill_delta_chunks(ep, w.schema.layout)   # 转换阶段
w.add_episode(ep)                             # 写出阶段
w.finalize()

# ----- 打开 -----
ds = open_dataset("/data/out")                 # 按 robot_type 自动套 layout
print(ds.level, ds.standard_version)   # L2 / 2026.04
ep0 = ds[0]

# ----- 转换 -----
pose9 = encode_pose(ep0.state.arm_single.ee.numpy(), "pose9")

# ----- 训练（可选：官方 DataLoader）-----
official = ds.as_lerobot()
```

可运行脚本：

- 单臂：[`examples/end_to_end.py`](../examples/end_to_end.py)
- **AgiBot 假数据写出+加载**：[`examples/agibot_write_load.py`](../examples/agibot_write_load.py)
- **AgiBot 源 → L1/L2**：[`examples/agibot_l0_l1_l2.py`](../examples/agibot_l0_l1_l2.py)

```bash
cd phi_rds && pip install -e ".[lerobot]"
python examples/end_to_end.py
python examples/agibot_write_load.py
python examples/agibot_l0_l1_l2.py --max-episodes 1
python examples/agibot_l0_l1_l2.py --max-episodes 4 --ray   # Ray map 转换
```

### 转换 vs 写出 / Ray

| 阶段 | 做什么 | 入口 |
|--|--|--|
| **convert** | FK+align、换系、`fill_delta_chunks` | `phi_rds.convert.*` / 数据集自己的 convert |
| **write** | `encode` 只序列化已有列 → parquet | `DatasetWriter.add_episode` |
| **Ray map** | 只跑 convert（模块级 `convert_fn`） | `phi_rds.exec.RayExecutor` + `process_row` |
| **Ray reduce** | driver 上 `write_l2` | 见 `examples/agibot_l0_l1_l2.py` |

`delta_chunk` **不在** `encode` 里算；漏填会报错，提示先 `fill_delta_chunks`。

---

## 2. L1 / L2

| | 含义 | API |
|--|------|-----|
| **L1** | → LeRobot v3；命名统一；布局贴近源数据 | `create_l1(...)` |
| **L2** | 完整统一标准；带 `standard_version`（默认最新） | `create_l2(...)` |

标记写在 `meta/info.json`：`phi_level` / `phi_standard_version`。

机型预设：

```python
from phi_rds import schema_for
schema = schema_for("umi", level="L2")          # 单臂默认
schema = schema_for("g1", level="L2", arms="bimanual")
```

---

## 3. 进阶（一般用不到）

```python
# 自定义 Schema / Meta
from phi_rds import DatasetMeta, DatasetSchema, create_dataset, preset_layout
layout = preset_layout("custom", arms="single")
schema = DatasetSchema.l2(layout)
w = create_dataset(path, schema, DatasetMeta(name="x", version="v1", robot="custom", uri=""))

# 手拼 Layout / Slice：见 schema.layout
# 坐标系图换系：lookup_T / convert_axes —— 见 convert 子包
```

---

## 4. AgiBot L1→L2（仅 phi_rds）

```bash
pip install -e ".[lerobot,robots]"   # robots = kinpy FK
python examples/agibot_l0_l1_l2.py --max-episodes 1
python examples/viz_agibot_l2.py     # FK 闭环图 → …/viz_l2_poses.png
```

`torso/head.pose` 由 `phi_rds.robots.agibot`（URDF + `calib/agibot.json`）计算，**不 import phi_data**。

---

## 5. 目录

```text
core/      Dataset · Episode · Body …
schema/   spec · presets · layout
io/       lerobot · serialize
convert/  pose · frame
robots/   URDF FK · AgiBot G1
```
