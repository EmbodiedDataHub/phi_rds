# Phi RDS — 设计说明（内部）

> 给维护者 / 自己读代码用，不是对外 USAGE。  
> 对应源码树：`phi_rds/src/phi_rds/`。  
> 对外入口见 `docs/USAGE.md`；本文解释**为什么这样拆**以及**每个 dataclass 干什么**。

---

## 0. 要解决什么

机器人轨迹数据有三层张力：

1. **磁盘互通**：训练侧已有 LeRobot v3（`info.json` + parquet + mp4），不要再发明第二套磁盘格式。
2. **语义树**：人想按「手臂 / 位姿 / 夹爪 / 相机」读，而不是按扁平列名。
3. **布局契约**：写出前就要钉死列集合；读时要能无损还原几何树。

Phi RDS 的答案：

| 层 | 角色 |
|----|------|
| **内存对象** | `Episode` + `RobotBody` 几何树（payload 懒加载） |
| **布局契约** | `Layout`（Episode↔列）+ `DatasetSchema`（整集 features / L1·L2） |
| **磁盘** | 唯一形态 = 官方 LeRobot v3；额外写轻量 stamp / meta JSON |

**刻意不做的事**

- 不把平台流水线的 L0/L1/L2（采集→清洗→训练）做成 SDK 架构层。  
  SDK 里的 **L1 / L2 只是数据标识**（见 §3）。
- 不自研 native pack / 第二套存储；`EpisodeStore` 协议只为可替换适配器。
- 不强制 `FieldMap` 为权威；日常以几何树为准，`fields` 留给 free-form 列。

---

## 1. 包结构与职责

```text
phi_rds/
  __init__.py          # 对外薄门面：open/create_l1|l2 + 重导出
  keys.py              # 稳定字符串：部位、相机槽、LeRobot 列名
  types.py             # 共享 Literal 别名
  core/
    lazy.py            # LazyHandle / StorageRef
    body.py            # Pose → Arm → RobotBody
    episode.py         # Episode + CameraStream + Labels (+ Field*)
    meta.py            # DatasetMeta / EpisodeMeta / Span
    semantics.py       # Units / FrameGraph / EpisodeSemantics
    stats.py           # FeatureStats / LengthDistribution / *Stats
    dataset.py         # RobotDataSet / DatasetWriter / Index / Store
  schema/
    layout.py          # Layout / Feature / Slice + encode/decode
    spec.py            # DatasetSchema / FeatureSpec / StandardSpec
    presets.py         # schema_for / preset_layout（机型快捷）
  io/
    lerobot.py         # LeRobotEpisodeStore + read/write/create
    serialize.py       # dataclass ↔ JSON
  convert/
    pose.py            # PoseRep + encode/decode/convert
    frame.py           # T 矩阵 / FrameGraph lookup / 轴系
```

依赖方向（应保持单向）：

```text
keys / types
    ↑
lazy → body → episode
              ↑
meta / semantics / stats
              ↑
schema (layout, spec, presets)  ←→  convert (pose, frame)
              ↑
core.dataset (RobotDataSet, Writer)
              ↑
io.lerobot
              ↑
__init__ 门面
```

`convert` 不依赖 `io`；`schema.layout.encode` 只依赖 `Episode`/`RobotBody` 与 `convert.pose`。

---

## 2. 两条主路径

### 2.1 写出（Schema 先行）

```text
preset_layout(robot) 或手拼 Layout
        ↓
DatasetSchema.l1|l2(layout)   # 或 schema_for(...)
        ↓
DatasetWriter.create(root, schema, meta)
        ↓  每条
Episode (几何树) ──encode(layout)──► 列 dict
        ↓
LeRobotDataset.add_frame* → save_episode
        ↓
finalize → 回写 info.json stamp + episode_uids / dataset_meta / schema.json
```

Happy path：`create_l2` + `Episode.from_arrays`（门面封装了 Schema/Meta）。

### 2.2 读入（懒索引）

```text
open_dataset(root)
        ↓
read_lerobot：读 info.json → DatasetSchema.from_lerobot_info
             （无 layout 时按 robot_type 推断 preset_layout）
        ↓
RobotDataSet(meta, schema, index, store=LeRobotEpisodeStore)
        ↓  ds[i]
store.open(uid)：切 parquet 行 → decode(layout) → RobotBody → Episode
```

索引层只持有 `EpisodeIndexRow`（uid / 帧数 / status / task）；打开 Episode 才触达列数据。  
训练旁路：`ds.as_lerobot()` 直接返回官方 `LeRobotDataset`，不经几何树。

### 2.3 内存中的「双视图」

| 视图 | 用途 |
|------|------|
| `Episode.state` / `.action` : `RobotBody` | 主路径；encode/decode 读写对象 |
| `Episode.fields` : `FieldMap` | 可选权威表；`profile=free` 且无 layout 时按列名灌数 |
| `Episode.cameras` / `.labels` | 相机路径与任务标签便利字段 |

---

## 3. L1 / L2（数据标识，不是流水线）

| | L1 | L2 |
|--|----|----|
| 含义 | 源格式 → LeRobot v3；**命名统一**；布局可贴近源数据 | **完整统一标准**；布局按标准约定 |
| `DatasetSchema.level` | `"L1"` | `"L2"` |
| `standard_version` | 通常 `None` | **必填**（缺省 → `LATEST` = `2026.04`） |
| `profile` | `"free"` | `"standard"` |
| 构造 | `DatasetSchema.l1(layout)` / `create_l1` | `DatasetSchema.l2(layout)` / `create_l2` |

磁盘 stamp（写进 `meta/info.json`，官方 LeRobot 会 warn 未知字段，可忽略）：

```json
{
  "phi_schema_id": "umi_single",
  "phi_level": "L2",
  "phi_standard_version": "2026.04"
}
```

标准版本注册表：`schema/spec.py` 的 `REGISTRY` / `resolve_standard()`。发版 = 追加日历版本 `YYYY.MM` 并改 `LATEST`。

---

## 4. 对象总图

```text
RobotDataSet
├── DatasetMeta
│   ├── EpisodeSemantics ── Units, FrameGraph(FrameEdge…)
│   ├── DatasetStats ── FeatureStats, LengthDistribution
│   └── ViewInfo
├── DatasetSchema
│   ├── FeatureSpec*          # 给 LeRobot.create 的 features
│   └── Layout?               # Episode↔列绑定
│         └── Feature* ── Slice*
├── EpisodeIndex ── EpisodeIndexRow*
└── EpisodeStore
      └── open(uid) → Episode
            ├── EpisodeMeta ── EpisodeSpan ── VideoSpan
            ├── EpisodeSemantics
            ├── RobotBody (state) / RobotBody? (action)
            │     └── Arm? ── Pose?, Gripper?, JointGroup?
            ├── CameraStream* ── LazyHandle?
            ├── Labels ── Segment*
            └── FieldMap ── Field* ── LazyHandle + FieldSemantics

LazyHandle ── StorageRef ── VideoLocator?
PoseRep（convert）— 向量写法，不进 Episode 树
Decoded（layout.decode 中间结果）— 不持久化
```

---

## 5. Dataclass 全表

约定：`frozen=True` = 值对象，改用 `replace`；未 frozen = 可变容器或带缓存。

---

### 5.1 `types.py` — 字面量（非 dataclass）

| 别名 | 取值 |
|------|------|
| `DataLevel` | `"L1" \| "L2"` |
| `EpisodeStatus` | `"active" \| "deleted" \| "failed" \| "unknown"` |
| `StatsScope` | `"self" \| "parent" \| "stale"` |
| `FieldId` | `str`（点分字段 id） |

---

### 5.2 `core/lazy.py` — 懒加载

#### `StorageKind`（Enum）

`MEMORY` | `VIDEO_MP4` | `OTHER`

#### `VideoLocator` — frozen

| 字段 | 说明 |
|------|------|
| `path` | mp4 路径 |
| `from_timestamp` / `to_timestamp` | 时间窗（秒） |
| `fps` | 可选 |

#### `StorageRef` — frozen

| 字段 | 说明 |
|------|------|
| `kind` | `StorageKind` |
| `uri` | 通用 URI |
| `video` | `VideoLocator?` |

#### `LazyHandle[T]` — 可变（有 `_cache`）

| 字段 | 说明 |
|------|------|
| `storage` | `StorageRef` |
| `shape` / `dtype` | 元数据提示 |
| `_cache` / `_loader` | 物化缓存与加载器 |

工厂：`from_memory(value)`（立刻填 cache）、`from_video(path, …)`（只挂路径）。  
**设计点**：几何与相机默认持句柄；`.numpy()` / `.load()` 才解码。视频像素不进 Episode 构造路径。

---

### 5.3 `core/body.py` — 几何树

位姿**可按任意表示存储**，用 `Pose.rep` 标明；需要 4×4 时再 `as_matrix()`。  
`convert.pose` 的互转枢纽仍是齐次矩阵，但 Episode 内存不必始终是矩阵。

#### `Pose`

| 字段 | 说明 |
|------|------|
| `values` | `ndarray \| LazyHandle`，形状随 `rep` |
| `rep` | `"matrix"` → `(N,4,4)`；否则为 `convert.PRESETS` 名（如 `pose9` / `quat_wxyz`），值为 `(N,dim)` |

方法：`numpy()`（原样）/ `as_matrix()` / `as_rep` / `as_quat` / `as_euler` / `as_pose9`；  
类方法 `from_matrix` / `from_rep`（**原样存**，不强制转矩阵）/ `from_quat` / `from_euler`。  
常量：`MATRIX_REP = "matrix"`。

#### `JointGroup`

| 字段 | 说明 |
|------|------|
| `position` | `(N,J)` |
| `names` | 长度必须 = J |

#### `Gripper`

| 字段 | 说明 |
|------|------|
| `open` | `(N,)` 或 `(N,1)`；约定 **0=闭 1=开**（`normalized_01`） |

#### `Arm`

| 字段 | 说明 |
|------|------|
| `ee` | `Pose?` |
| `gripper` | `Gripper?` |
| `joints` | `JointGroup?` |

#### `RobotBody`

观测或动作一整棵几何；缺的部位留 `None`。

| 字段 | 说明 |
|------|------|
| `base` / `torso` / `head` | 躯干系 `Pose?` |
| `arm_left` / `arm_right` / `arm_single` | `Arm?` |
| `waist` / `head_joints` | 非臂关节 `JointGroup?` |

查询：`ee_matrix(part)` / `gripper_of` / `joint_group` / `joint_angles`；`with_ee` 批量替换位姿。  
`part` 字符串与 `keys.ARM_*` / `BASE` 等一致。

---

### 5.4 `core/episode.py` — 单条轨迹

#### `FieldKind`（Enum）

`POSE | JOINTS | GRIPPER | IMAGE | DEPTH | LABEL | SCALAR | VECTOR | DELTA_CHUNK | OTHER`

#### `Field`

| 字段 | 说明 |
|------|------|
| `id` | `FieldId` |
| `kind` | `FieldKind` |
| `semantics` | `FieldSemantics` |
| `payload` | `LazyHandle` |

#### `FieldMap`

可变字典包装：`put` / `prefix` / `as_dict`。  
**当前主路径可不填**；free schema 无 layout 时 Writer 从这里取列。

#### `Segment` — frozen

`start`, `end`, `text` — 子任务时间片（帧索引）。

#### `Labels` — frozen

`task: str`, `subtasks: tuple[Segment, ...]`

#### `CameraStream`

| 字段 | 说明 |
|------|------|
| `name` | 相机槽名（对齐 LeRobot `observation.images.*`） |
| `rgb` / `depth` | `LazyHandle?`；见下方 **§5.13 / 视频模型** |
| `intrinsic` / `extrinsic` | 可选标定 |

`from_arrays(cameras={名: 路径})` 只是写时简写（一集先挂一个源文件）。  
**读标准 v3 时**：同一 mp4 可被多集共享，关键是 `VideoLocator.from_timestamp/to_timestamp`。

#### `Episode`

| 字段 | 说明 |
|------|------|
| `meta` | `EpisodeMeta` |
| `state` | `RobotBody`（观测） |
| `action` | `RobotBody?` |
| `cameras` | `tuple[CameraStream, ...]` |
| `labels` | `Labels` |
| `semantics` | `EpisodeSemantics`（可与集级语义重复/覆盖） |
| `fields` | `FieldMap` |
| `source` | 可选溯源字符串 |

快捷：`uid` / `fps` / `num_frames` / `stats`；`from_arrays(...)` 入门构造器（单臂默认 `ARM_SINGLE`）。

---

### 5.5 `core/meta.py` — 元信息

#### `ViewInfo` — frozen

| 字段 | 说明 |
|------|------|
| `parent_uri` | 视图来源 |
| `selector` | 如 `select(3 uids)` |
| `is_view` | 是否为子集视图 |

`RobotDataSet.select/range/filter` 产生新数据集时填此项；**不复制底层 store**。

#### `DatasetMeta` — frozen

集级身份与挂载点。

| 字段 | 说明 |
|------|------|
| `name`, `version`, `robot` | 身份三元组 |
| `uri` | 磁盘根路径（resolve 后） |
| `fps` | 默认 30 |
| `schema_id` | 与 `DatasetSchema.id` 对齐 |
| `level` / `standard_version` | 可与 schema 冗余；`bind_schema` 时从 schema 灌入 |
| `semantics` | `EpisodeSemantics` |
| `stats` | `DatasetStats` |
| `view` | `ViewInfo` |
| `extras` | 扩展袋 |

`bind_schema(schema)` → 新 `DatasetMeta`（fps/robot/level/standard/schema_id 对齐）。

#### `VideoSpan` — frozen

`path`, `from_timestamp`, `to_timestamp` — Episode 级视频片段描述（进 `EpisodeSpan`）。

#### `EpisodeSpan` — frozen

`videos: dict[str, VideoSpan]`, `extras`

#### `EpisodeMeta` — frozen

| 字段 | 说明 |
|------|------|
| `uid` | 稳定 id（写入 `meta/episode_uids.json`） |
| `num_frames` | 帧数 |
| `fps`, `task` | |
| `status` | `EpisodeStatus` |
| `episode_index` | 在当前视图中的下标（视图可重编号） |
| `span` | `EpisodeSpan` |
| `stats` | `EpisodeStats` |
| `extras` | |

---

### 5.6 `core/semantics.py` — 语义 / 坐标系

#### `Units` — frozen

`length="m"`, `angle="rad"`, `gripper="normalized_01"`, `time="s"`

#### `FrameRef` — frozen

`expressed_in`, `body_of?`, `axes="FLU"` — 单字段参考系描述（预留，主路径多用 FrameGraph）。

#### `FrameEdge` — frozen

| 字段 | 说明 |
|------|------|
| `child` / `parent` | 坐标系名 |
| `kind` | `"extrinsic" \| "fk"` |
| `matrix_flat16` | 行主序 4×4，16 floats |

#### `FrameGraph` — frozen

`frames`, `edges`, `root` — 给 `convert.frame.lookup_T` 用。

#### `FieldSemantics` — frozen

`kind`, `role`（如 observation）, `part?`, `extras` — 挂在 `Field` 上。

#### `EpisodeSemantics` — frozen

| 字段 | 说明 |
|------|------|
| `semantic_level` | 如 `raw` / `name_unified` / `frame_unified`（读路径会按 L1/L2 填） |
| `robot_type` / `layout_id` | |
| `units` | `Units` |
| `default_frames` | `FrameGraph` |
| `extras` | |

---

### 5.7 `core/stats.py` — 统计

#### `FeatureStats` — frozen

`name`, `count`, `mean/std/min/max`（tuple，按维）, `quantiles`

#### `LengthDistribution` — frozen

集级「每集帧数」分布：`count/min/max/mean/std` + `p1…p99`；`from_values` / `quantile_map`。

#### `EpisodeStats` — frozen

`num_frames`, `extras`（刻意轻量）。

#### `DatasetStats` — frozen

| 字段 | 说明 |
|------|------|
| `num_episodes` / `num_frames` | |
| `feature_list` | 特征名列表 |
| `status_counts` | status → 计数 |
| `frame_count` | `LengthDistribution?` |
| `features` | `name → FeatureStats` |
| `scope` | `self` = 本视图自洽；`parent` = 特征统计仍来自父集 |
| `extras` | |

`with_counts(...)` 返回新实例。索引变更时 `EpisodeIndex.recount_stats()` 重算计数与长度分布。

---

### 5.8 `core/dataset.py` — 数据集与写出

#### `EpisodeStore`（Protocol，非 dataclass）

`list_metas()` / `open(uid)` / `exists(uid)`  
实现：`InMemoryEpisodeStore`、`LeRobotEpisodeStore`（`io/lerobot.py`）。

#### `EpisodeIndexRow` — frozen

`uid`, `num_frames`, `status`, `task`, `episode_index?`  
**轻量索引行**：列表 / filter / 分页只碰这些字段。

#### `EpisodeIndex`

`rows: tuple[EpisodeIndexRow, ...]` + 内部 `_by_uid`。  
`select` / `range` / `from_metas` / `recount_stats`。

#### `InMemoryEpisodeStore`

`_episodes` / `_metas` / `_order` — 测试与 `from_episodes` 用。

#### `DatasetWriter`

| 字段 | 说明 |
|------|------|
| `root` | 输出目录 |
| `schema` | 创建时钉死，不可中途改 features |
| `meta` | 已 `bind_schema` |
| `_ds` | 官方 `LeRobotDataset` |
| `_uids` | 写入顺序的 uid 列表 |
| `_finalized` | |

流程：`create` → `add_episode`（`encode` 或 `fields`）→ `finalize`（stamp + 辅助 JSON）。  
**高效点**：底层只 `create` 一次；每集 `add_frame` 循环 + `save_episode`。

#### `RobotDataSet`

| 字段 | 说明 |
|------|------|
| `meta` | `DatasetMeta` |
| `schema` | `DatasetSchema`（打开后固定） |
| `index` | `EpisodeIndex` |
| `store` | `EpisodeStore` |
| `_meta_cache` / `_episode_cache` | LRU 式小缓存（默认 32 集） |

入口：`open(path)` / `create(...)→Writer` / `from_episodes` / `load`。  
视图：`select` / `range` / `take` / `skip` / `filter` → 新 `RobotDataSet` 共享 store。  
旁路：`as_lerobot()`；`save(root)` → `write_dataset`。

属性：`level` / `standard_version` / `is_l1` / `is_l2`（schema 优先，meta 兜底）。

---

### 5.9 `schema/layout.py` — Episode ↔ 列

#### `Slice` — frozen

Feature 内一段连续列；**决定从 `RobotBody` 哪取、怎么编码**。

| 字段 | 说明 |
|------|------|
| `kind` | `pose` / `pose9` / `gripper` / `joint` / `delta_chunk` … |
| `from_` | `"state"` \| `"action"` |
| `part` | `keys` 部位 |
| `names` | 列名（标注 / 宽度） |
| `delta_from` | 相对位姿参考部位 |
| `fk_names` / `src_width` / `src_slots` | 关节重排与真实宽度 |
| `optional` | 缺失时填零 |
| `parts` / `chunk_size` / `direction` | `delta_chunk` 专用 |
| `pose_rep` | 覆盖默认 pose9 |

属性：`real_width`, `resolved_pose_rep`。

#### `Feature` — frozen

| 字段 | 说明 |
|------|------|
| `name` | LeRobot 列名，如 `observation.state.arm.tcp_pose` |
| `slices` | 横向拼接的多段 |
| `dtype` / `shape` | 写入 dtype；shape 可省略由 slices 推 |

`names` 属性 = 拼接所有 slice.names。

#### `Layout` — frozen

| 字段 | 说明 |
|------|------|
| `id` | 如 `umi_single` |
| `robot_type` | 机型 |
| `frame` | 默认参考系（常 `base`） |
| `features` | `Feature` 元组 |

#### `Decoded` — 可变，**仅 decode 中间产物**

| 字段 | 键 | 值 |
|------|----|----|
| `poses` / `deltas` | `(from_, part)` | `(N,4,4)` |
| `delta_refs` | `(from_, part)` | 参考 part 名 |
| `joints` | `(from_, part)` | `(N,J)` |
| `joint_names` | `(from_, part)` | 名元组 |
| `grippers` | `(from_, part)` | `(N,1)` |

`encode(episode, layout) → dict[str, ndarray]`  
`decode(columns, layout) → Decoded`（1-D 列会 reshape 成 `(N,1)`，兼容 HF squeeze）

---

### 5.10 `schema/spec.py` — 整集 Schema

#### `StandardSpec` — frozen

`version`, `doc`, `notes`；`is_latest`。由 `REGISTRY` 解析，不手造。

#### `FeatureSpec` — frozen

单个 LeRobot feature 的**磁盘布局声明**（给 `LeRobotDataset.create`）。

| 字段 | 说明 |
|------|------|
| `name`, `dtype`, `shape` | |
| `names` | 维名 |
| `pad` | 可选 padding 元数据 |
| `is_video` | `dtype=="video"` |
| `extras` | 透传官方字段 |

`to_lerobot()` / `from_lerobot(name, blob)`。

#### `DatasetSchema` — frozen

**创建 / 打开的第一步**。

| 字段 | 说明 |
|------|------|
| `id` | 常 = `layout.id` |
| `fps` | |
| `robot_type` | |
| `level` | `L1` / `L2` / `None` |
| `standard_version` | L2 必填 |
| `profile` | `"standard" \| "free"` |
| `features` | `FeatureSpec` 元组 |
| `layout` | `Layout?` — 有则 encode 几何树；无则走 fields |
| `extras` | |

工厂：`l1` / `l2` / `free` / `from_layout` / `from_lerobot_info`。  
`to_stamp()` → info.json 三字段；`to_lerobot_features()` → create 参数。

**Layout vs FeatureSpec 分工**

- `Layout`：语义绑定（从哪只手臂取 pose9）。  
- `FeatureSpec`：磁盘形状（dtype/shape/names）。  
- `from_layout`：由 Layout 推导 FeatureSpec 列表。

---

### 5.11 `schema/presets.py` — 机型快捷（函数，非 dataclass）

- `preset_layout(robot, arms=single|bimanual, joint_count=7)` → `Layout`  
- `schema_for(robot, level=L2, …)` → `DatasetSchema`

预设列：state/action × (tcp_pose, gripper, joint)，命名走 `keys.STATE_ARM_*` / `ACTION_ARM_*`。

---

### 5.12 `convert/pose.py`

#### `PoseRep` — frozen

一种位姿**向量写法**的完整语义（不是轨迹里的 Pose 对象）。

| 字段 | 说明 |
|------|------|
| `name` | 注册名，如 `pose9` |
| `rotation` | `mat_cols2` / `quat_wxyz` / `euler` / … |
| `translation` | `xyz` \| `none` |
| `euler_seq` / `angle_unit` | 欧拉专用 |

属性：`dim`, `rotation_dim`, `names`。  
预设表 `PRESETS`；API：`encode_pose` / `decode_pose` / `convert_pose` / `resolve_rep`。

默认磁盘列：**pose9** = xyz + 旋转矩阵前两列（6）= 9 维。

---

### 5.13 `io/lerobot.py` — 适配器 + **LeRobot v3 视频模型**

#### 视频怎么存（v3，与 v2.x 不同）

**一个 mp4 里可以有多集。** 路径按 chunk/file 索引共享；每集只记时间窗。

```text
videos/{video_key}/chunk-{chunk:03d}/file-{file:03d}.mp4   ← 物理文件（可很长）
meta/episodes/...parquet 每行：
  videos/{video_key}/chunk_index
  videos/{video_key}/file_index
  videos/{video_key}/from_timestamp   ← 本集在该 mp4 内的起点（秒）
  videos/{video_key}/to_timestamp     ← 终点（秒）
```

相对 v2（一集一个独立 mp4），v3 用 **segmentation** 省文件数、方便追加。

#### 官方怎么加载一帧

`LeRobotDataset` 取第 `ep_idx` 集、相对时间戳 `ts`（相对本集起点）时：

1. `path = meta.get_video_file_path(ep_idx, vid_key)`  
   ← 用该集的 `chunk_index` / `file_index` 填 `video_path` 模板  
2. `abs_ts = from_timestamp + ts`  
3. `decode_video_frames(path, [abs_ts], …)` 按**绝对秒** seek 解码  

训练 DataLoader 走这条路径；**不需要**先把整集像素读进 Episode。

#### 我们 SDK 怎么挂

`LeRobotEpisodeStore._load_cameras`：

- `LazyHandle` / `VideoLocator` = `{path, from_timestamp, to_timestamp, fps}`  
- **不解码**；多集可指向同一 `path`、不同时间窗  
- 看像素 → `ds.as_lerobot()`（官方 decode）

`EpisodeMeta.span.videos` 同步存 `VideoSpan(path, from, to)`，索引层也能看到分段信息。

`from_arrays(cameras={名: 路径})` 只是**写前**简写（源文件常是一集一段）；  
真正 finalize 进标准 v3 目录后，以 episodes 表里的 file_index + 时间窗为准。

#### `LeRobotEpisodeStore`

持有 `root`, `layout`, 官方 `LeRobotDataset`, uid 列表。  
`open(uid)`：切 parquet → `decode(layout)` → `RobotBody`；相机按上表挂句柄。

辅助：`_infer_preset_layout` / `read_lerobot` / `write_lerobot`。

磁盘旁路文件（finalize 写）：

| 路径 | 内容 |
|------|------|
| `meta/info.json` | 官方 + `phi_*` stamp |
| `meta/episode_uids.json` | uid 顺序 |
| `meta/dataset_meta.json` | `DatasetMeta` |
| `meta/schema.json` | schema 摘要（非完整 Layout） |
| `meta/semantics.json` | 可选 |

---

## 6. 命名体系（`keys.py`）

稳定常量，禁止在业务代码里散落魔法字符串：

- **部位**：`arm_left` / `arm_right` / `arm_single` / `base` / `torso` / `head` / `waist`
- **相机槽**：`head`, `wrist_left`, …
- **LeRobot 列**：`observation.state.arm.tcp_pose`, `action.arm.joint`, …
- **辅助**：`pose9_names(part)`, `arm_joint_names(part, n)`, `gripper_name(part)`

双臂时列名带 `left_` / `right_` 前缀；单臂无侧前缀。

---

## 7. 设计决策备忘

1. **几何树优先于扁平 Field**  
   encode/decode 围绕 `RobotBody`；FieldMap 是逃逸舱，不是主模型。

2. **Pose 带 `rep`，可存任意表示**  
   默认 `matrix`；也允许 `quat_wxyz` / `pose9` 等原样挂在树上。  
   encode / 换系需要 4×4 时走 `as_matrix()`；磁盘列仍由 Layout 的 `pose_rep` 决定。

3. **Schema 创建后不可变**  
   Writer 不支持中途加列；要新布局就新数据集（或上层做 migration）。

4. **视图不物化**  
   `select` 只换 Index + ViewInfo，共享 Store；统计 `scope=parent` 时特征级 stats 可继承。

5. **L1/L2 是 stamp，不是存储分叉**  
   磁盘格式相同；差异在 layout 约束与 `standard_version` 是否存在。

6. **打开可无手传 Layout**  
   依赖 `robot_type` + 预设；自定义 layout 的集仍应显式 `open(..., layout=)` 或未来把 Layout 序列化进 meta（当前 `schema.json` 不含 Slice 细节）。

7. **与官方 LeRobot 共存**  
   未知 `phi_*` 字段会被 warn；训练用 `as_lerobot()`，标注/质检用 Episode 树。

---

## 8. 扩展点（以后加代码时落这里）

| 想加什么 | 落点 |
|----------|------|
| 新机型预设 | `schema/presets.py` |
| 新标准版本 | `REGISTRY` + 改 `LATEST` |
| 新位姿向量写法 | `PoseRep` + `PRESETS` |
| 新 Slice kind | `layout._raw_slice` / `decode` |
| 新存储后端 | 实现 `EpisodeStore`，在 `io/` 注册 |
| 把 Layout 完整落盘 | 扩展 finalize / `from_lerobot_info` |

---

## 9. 与对外文档的关系

| 文档 | 读者 |
|------|------|
| `README.md` / `docs/USAGE.md` | 使用者：30 秒 API |
| `examples/end_to_end.py` | 可运行流水线 |
| `examples/understand_phi_rds.ipynb` | 交互走读 + 语法复习 |
| **本文 `docs/DESIGN.md`** | 维护者：架构与全部 dataclass |

读代码建议顺序：`types` → `lazy` → `body` → `episode` → `layout.encode` → `spec` → `dataset.Writer` → `io/lerobot`。  
交互走读：打开 `examples/understand_phi_rds.ipynb` 从上到下跑。
