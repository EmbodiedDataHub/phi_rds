# Phi L2 数据集元信息

磁盘：LeRobot v3。phi 盖戳：`phi_level=L2`、`phi_standard_version`。

飞书：[Phi L2 数据集元信息标准](https://zcnce50wan15.feishu.cn/docx/CBPSdrZ8yojJKLxda1ecD4jdnfh)  
相机内外参：[统一标准](https://zcnce50wan15.feishu.cn/wiki/Rj8IweQMLi4XECks7WGcg8denEb)

## 数据集级（`meta/info.json`）

| 字段 | 说明 |
|---|---|
| `codebase_version` | `v3.0` |
| `robot_type` | 机型，如 `yam` |
| `fps` | 帧率 |
| `total_episodes` / `total_frames` / `total_tasks` | 规模 |
| `splits` | 划分，如 `{"train":"0:N"}` |
| `data_path` / `video_path` | 路径模板 |
| `features` | 列定义（dtype / shape / names） |
| `phi_level` / `phi_standard_version` / `phi_schema_id` | phi 戳 |

YAM 双臂 L2 `features` 含：`observation|action.state.arm.{tcp_pose,tcp_pose.delta_chunk,gripper,joint}`，三路 `observation.images.{head,wrist_left,wrist_right}`（产品 `[480,640,3]`），以及索引列。无 torso/head.pose/base 空槽。

## Episode 级（`meta/episodes/*.parquet`）

### 官方列

`episode_index` · `tasks` · `length` · `dataset_from_index` · `dataset_to_index` · `data/*` · `videos/*` · `stats/*`

### 追加列

| 列 | 类型 | 说明 |
|---|---|---|
| `source.uid` | string | 源 id（唯一身份） |
| `source.path` | string | 相对路径，如 `val/.../episode.mcap` |
| `source.gripper_model` | string | 如 `linear_4310` |
| `source.images.head.topic` | string | 选用的顶视 topic（左/右/单路） |
| `source.observation.state.arm.tcp_pose.isnull` | bool | 源 state pose 是否缺失 |
| `source.action.arm.tcp_pose.isnull` | bool | 源 action pose 是否缺失 |
| `source.observation.images.*.shape` | list\<int\> | 原始图 `[H,W,C]` |
| `cameras` | string(JSON) | 见下 |

废弃：`episode_uid` · `camera_calib` · `source` · `source.ee`

### `cameras` 列

列名 `cameras`。JSON **即**标准中的 cameras 对象（无外层包装）：

```json
{
  "head": {
    "intrinsic": {
      "model": "pinhole",
      "fx": 245.59, "fy": 245.59, "cx": 332.61, "cy": 231.04,
      "distortion": { "model": "none", "coeffs": {} }
    },
    "extrinsic": {
      "parent_frame": "mid_base",
      "convention": "parent_from_camera",
      "rotation": { "matrix": [[1,0,0],[0,1,0],[0,0,1]] },
      "translation": { "xyz_m": [0, 0, 0] }
    }
  },
  "wrist_left": { "intrinsic": {…}, "extrinsic": {…} },
  "wrist_right": { "intrinsic": {…}, "extrinsic": {…} }
}
```

每相机仅 `intrinsic` + `extrinsic`，字段定义见 [相机内外参统一标准](https://zcnce50wan15.feishu.cn/wiki/Rj8IweQMLi4XECks7WGcg8denEb)。内参对应产品分辨率图像。
