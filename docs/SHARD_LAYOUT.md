# 分片 → 数据集：目录与标记文件

ABC-130K（及其他大批量）转 L2 时走两步：

1. **分片写出**：每个 Job 写一个「单片」小库  
2. **合并成数据集**：多个单片 flatten / copy merge 成最终可训练目录  

权威脚本：

| 步骤 | 脚本 |
|------|------|
| 分片 | [`examples/abc130k_task_shards.py`](../examples/abc130k_task_shards.py) |
| 合并（平坦并行） | [`examples/merge_task_shards_flat.py`](../examples/merge_task_shards_flat.py) |
| 合并（快拷） | [`examples/merge_shards_fast.py`](../examples/merge_shards_fast.py) |
| 校验 | [`examples/verify_abc_task_shards.py`](../examples/verify_abc_task_shards.py) |

元信息字段含义见 [`DATASET_META.md`](DATASET_META.md)。

---

## 1. 三层目录（不要混读）

```text
① 分片根（Job 账本层）
   manifest.jsonl     ← 分片 Job 账本（谁写了、状态如何）
   _SUCCESS           ← 全部分片 Job 完成
   by_global/part-*/  ← 或 by_task/{task}/part-*/

② 单片（一个 Job 的产物 = 独立小 L2）
   _SUCCESS
   shard_meta.json
   data/  videos/  meta/{info,stats,episodes}/…

③ 数据集（merge 后的最终目录）
   _SUCCESS
   data/  videos/
   meta/{info,stats,tasks*,episodes}/…   ← 写库主数据（权威）
```

| 层 | 职责 | 权威文件 |
|----|------|----------|
| 分片根 | 调度 / 对账 / 重跑 | `manifest.jsonl` |
| 单片 | 并行写出单元 | `shard_meta.json` + 本片 `meta/` |
| 数据集 | 训练入口 | `meta/`（只信这里） |

数据流：

```text
Job 写出单片 (②)
      ↓ 登记
分片根 manifest.jsonl (①)
      ↓ merge
数据集 meta/ + data/ + videos/ (③)
```

---

## 2. 完整树形

### ① 分片根

```text
{shards_root}/                          # 例：…/abc_l2_train
├── manifest.jsonl                      # 一行一个 shard
├── _SUCCESS                            # 全部分片成功后才写
└── by_global/                          # --shard-mode global（推荐）
    ├── part-00000/
    ├── part-00001/
    └── …
# 或
└── by_task/                            # --shard-mode task（旧）
    └── {task_name}/
        ├── part-000/
        └── …
```

### ② 单片（`by_global/part-XXXXX/`）

与正式 L2 同形，多两个标记文件：

```text
{shards_root}/by_global/part-00000/
├── _SUCCESS                            # 本片完成；重跑可跳过
├── shard_meta.json                     # 本片范围 / uid 列表
├── data/                               # parquet 轨迹
├── videos/                             # mp4
└── meta/
    ├── info.json
    ├── stats.json                      # 或 stats/
    ├── episodes/                       # episode 索引 parquet
    └── …                               # 其它 sidecar（schema 等）
```

### ③ 数据集（merge 输出）

```text
{dataset}/                              # 例：…/abc_l2_train_merged
├── _SUCCESS                            # merge 完成摘要
├── data/
├── videos/
└── meta/                               # ★ 主数据
    ├── info.json
    ├── stats.json
    ├── tasks.jsonl / tasks.parquet     # 视写出实现
    ├── episodes/
    └── …
```

> 命名约定：**不叫「训练库」**，统一叫 **数据集**（disk = LeRobot v3 + phi stamp）。

---

## 3. 标记文件内容 & 谁写的

### 3.1 单片 `_SUCCESS`

- **含义**：本片 `data/` + `meta/` + `videos/` 已落盘，可被 merge / 重跑跳过。  
- **内容**：纯文本一行 `ok`（不是 JSON）。  
- **写出时机**：`convert_shard` 在 `shard_meta.json` 写完之后。

```python
# examples/abc130k_task_shards.py → convert_shard
ok = dest / "_SUCCESS"
# …
(root / "shard_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
ok.write_text("ok\n")
```

跳过逻辑（已有 `_SUCCESS` 且 `meta/info.json` 存在则 SKIP）：

```python
if ok.is_file() and not overwrite and (dest / "meta" / "info.json").is_file():
    info = json.loads((dest / "meta" / "info.json").read_text())
    return {"shard_id": shard.shard_id, "status": "SKIP", ...}
```

示例内容：

```text
ok
```

### 3.2 `shard_meta.json`（仅单片有）

- **含义**：本片 Job 的自描述（不进最终数据集；merge 可选用）。  
- **写出**：同上，`convert_shard` 末尾。

字段示例：

```json
{
  "shard_id": "part-00000",
  "split": "train",
  "task": "",
  "tasks": ["pick_cup", "open_drawer"],
  "mode": "global",
  "part": 0,
  "n_episodes": 50,
  "uids": ["ep-…", "ep-…"],
  "source_paths": ["val/…/episode.mcap", "…"]
}
```

对应代码：

```python
meta = {
    "shard_id": shard.shard_id,
    "split": shard.split,
    "task": shard.task,
    "tasks": tasks_in,
    "mode": shard.mode,
    "part": shard.part,
    "n_episodes": len(episodes),
    "uids": [ep.uid for ep in episodes],
    "source_paths": [str(getattr(ep, "source", "")) for ep in episodes],
}
(root / "shard_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
```

### 3.3 分片根 `manifest.jsonl`

- **含义**：**分片 Job 账本**——计划了哪些 shard、实际状态、输出路径。  
- **格式**：JSONL，一行一个 shard。  
- **写出**：全部 Job 跑完后 `write_manifest(...)`。

```python
# examples/abc130k_task_shards.py
def write_manifest(path: Path, shards: list[ShardSpec], results: list[dict]) -> None:
    by_id = {r.get("shard_id"): r for r in results}
    with path.open("w", encoding="utf-8") as f:
        for sh in shards:
            row = {
                "shard_id": sh.shard_id,
                "split": sh.split,
                "task": sh.task,
                "part": sh.part,
                "n_episodes": sh.n_episodes,
                "mcaps": list(sh.mcaps),
                "out": str(sh.out_dir(path.parent)),
                "status": by_id.get(sh.shard_id, {}).get("status"),   # OK | SKIP | FAIL…
                "n_frames": by_id.get(sh.shard_id, {}).get("n_frames"),
                "error": by_id.get(sh.shard_id, {}).get("error"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# main 末尾
write_manifest(out / "manifest.jsonl", shards, results)
```

单行示例：

```json
{"shard_id":"part-00000","split":"train","task":"","part":0,"n_episodes":50,"mcaps":["…"],"out":"…/by_global/part-00000","status":"OK","n_frames":12000,"error":null}
```

### 3.4 分片根 `_SUCCESS`

- **含义**：计划中的分片全部 `OK` 或 `SKIP`，可以开 merge。  
- **内容**：一小段 JSON（shard / episode 计数）。  
- **条件**：`all(status in ("OK","SKIP"))` 才写；有失败则不写。

```python
ok = all(r.get("status") in ("OK", "SKIP") for r in results)
if ok:
    (out / "_SUCCESS").write_text(
        json.dumps({"shards": len(results), "episodes": n_ep}) + "\n"
    )
```

示例：

```json
{"shards": 42, "episodes": 2100}
```

### 3.5 数据集 `_SUCCESS`（merge 产物）

- **含义**：合并完成，`meta/` 已是权威主数据。  
- **内容**：JSON，带 `mode` 与规模字段（两种 merge 脚本字段略有不同）。

`merge_task_shards_flat.py`：

```python
(out / "_SUCCESS").write_text(
    json.dumps(
        {
            "mode": "flat_parallel_pack",
            "shards": len(plans),
            "episodes": total_ep,
            "frames": total_fr,
            "tasks": len(task_to_idx),
            "source_video_files": n_video_files,
            "packed_video_files": n_packed,
            "data_files": n_data_files,
            "workers": args.workers,
            "source": str(args.shards_root),
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
```

`merge_shards_fast.py`：

```python
(out / "_SUCCESS").write_text(
    json.dumps(
        {
            "mode": "copy_merge",
            "shards": len(plan["jobs"]),
            "episodes": plan["total_episodes"],
            "frames": plan["total_frames"],
            "tasks": len(plan["task_to_idx"]),
            "video_bytes": video_bytes,
        },
        indent=2,
    )
    + "\n"
)
```

示例（flat）：

```json
{
  "mode": "flat_parallel_pack",
  "shards": 42,
  "episodes": 2100,
  "frames": 504000,
  "tasks": 18,
  "source_video_files": 6300,
  "packed_video_files": 210,
  "data_files": 42,
  "workers": 64,
  "source": "/mnt/.../abc_l2_train"
}
```

---

## 4. 读写约定（防踩坑）

| 需求 | 看哪里 |
|------|--------|
| Job 是否齐、哪片挂了 | 分片根 `manifest.jsonl` |
| 某片能不能跳过 | 单片 `_SUCCESS` + `meta/info.json` |
| 某片覆盖了哪些 episode | 单片 `shard_meta.json` |
| 能不能开 merge | 分片根 `_SUCCESS` 存在 |
| 训练 / 统计 / task 列表 | **只信数据集 `meta/`** |
| 数据集是否 merge 完 | 数据集 `_SUCCESS` |

约定：

1. `_SUCCESS` 一律**最后写**（目录内容就绪后再 touch/write）。  
2. 单片 `_SUCCESS` 是纯文本 `ok\n`；根 / 数据集 `_SUCCESS` 是 **JSON 一行或 indent JSON**。  
3. merge **不依赖** `shard_meta.json` 作为主数据；最终以写出的数据集 `meta/info.json` 等为准。  
4. 重跑单片：`--overwrite`，会先 `rmtree(dest)` 再写。

---

## 5. 最小自检片段

```python
from pathlib import Path
import json

def shard_ready(part: Path) -> bool:
    return (part / "_SUCCESS").is_file() and (part / "meta" / "info.json").is_file()

def load_manifest(root: Path) -> list[dict]:
    return [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines() if l.strip()]

def dataset_ready(ds: Path) -> bool:
    return (ds / "_SUCCESS").is_file() and (ds / "meta" / "info.json").is_file()
```

---

## 6. 命令速查

```bash
# 分片
python examples/abc130k_task_shards.py \
  --source …/data/val --out …/abc_l2_val \
  --shard-mode global --max-per-part 50

# 合并
python examples/merge_task_shards_flat.py \
  --shards-root …/abc_l2_val --out …/abc_l2_val_merged --workers 64

# 校验单片树
python examples/verify_abc_task_shards.py --root …/abc_l2_val
```
