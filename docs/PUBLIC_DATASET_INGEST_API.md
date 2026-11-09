# 公开数据集 Meta 入库

飞书：[Meta 入库接口步骤](https://zcnce50wan15.feishu.cn/wiki/NY77wojDEiWJv8kFwNFcN9acnyd)

仅元信息入库（不搬运 OSS）。范围：上传、失败续传、删除平台记录。API：`/api/v3`。

前置：数据在 OSS 规范路径；已用 `read_dataset_meta` / `read_episode_meta` 得到 JSON。

---

## 一、上传元信息

流程：登录 → 查项目 → 建 Dataset → 按 split 建 SubDataset → 写 Episode meta（并归属）。

| 步骤 | 接口 | 输入 | 输出 | 状态 |
|------|------|------|------|------|
| 登录 | `POST /login/access-token` | `username`,`password` | `200` → `access_token` | 现有 |
| 查项目 | `GET /projects/` | `project_name=公开数据集` | `200` → `project_id` | 现有 |
| 建 Dataset | `POST /datasets/` | `dataset_name`,`project_id`；可选 `description`,`metadata` | `201` → `dataset_id` | 现有 |
| 建 SubDataset | `POST /sub-datasets/` | `sub_dataset_name`；建议 `dataset_id`,`dataset_type`,`metadata`。不可传 `data_url` | `201` → `sub_dataset_id` | 现有 |
| 批量写 Episode 并归属 | `POST /episodes/batch` | 见下 | 见下 | **需新增** |
| 单条写 Episode | `POST /episodes/` | `episode_name`；可选 `metadata`。不可传 `sub_dataset_id` | `201` 草稿，通常无归属 | 现有 |

现状：单条创建不能指定归属；挂载要求验证上传清单，公开集纯 meta 通常无法挂载。闭环依赖新增 batch，或后端调整挂载策略。

### 批量接口（需新增）`POST /episodes/batch`

| 项 | 说明 |
|----|------|
| 请求 | `sub_dataset_id` + `episodes[]`（`episode_name`；`metadata` 含 `episode_index`）。Header 建议 `idempotency-key`。单次 200～500 条 |
| 响应 | `201`（含 `episode_ids`，创建时归属）；或 `202`+`task_id`，轮询至 `completed`/`failed` |
| 成功判定 | 同步看成功码与 ID；异步看 `completed`（`202` 不算完成）。核对 `sub_dataset_id` 或子集下 index 数量 |

```json
{
  "sub_dataset_id": "s_ds_…",
  "episodes": [
    { "episode_name": "0", "metadata": { "episode_index": 0, "length": 120 }, "is_available": true }
  ]
}
```

---

## 二、中途失败与续传

| 原则 | 做法 |
|------|------|
| 防打崩 | 小分片、默认串行（最多 2 路）；`429/5xx` 指数退避 |
| 落盘进度 | 每片成功写 checkpoint |
| 幂等 | `public:{sub_dataset_id}:ep[{start},{end})` |
| 续跑 | checkpoint ∪ `GET /episodes/?sub_dataset_id=` → 只投剩余 |
| 禁止 | 因单次失败整库删除重传（整份错误走第三节） |

待对齐：分片事务语义；重复 `episode_index` 是 `409` 还是覆盖；batch 幂等/异步。

---

## 三、删除上传（仅清库）

只删平台 DB 元信息，**不删 OSS**。现有 DELETE 语义为保留物理文件。

**UI：** 若走同一套 DELETE，仅为下架库内记录；OSS L2 保留。若有「清 OSS」能力须单独标明。

| 场景 | 接口 | 说明 |
|------|------|------|
| 错一个 split | `DELETE /sub-datasets/{sub_dataset_id}` | 最小删除单位；级联 Episode；保留物理文件 |
| 整份 Dataset | `DELETE /datasets/{dataset_id}` | 确认子集残留则继续删 SubDataset |
| 单条 Episode | — | 无单条 DELETE |
| 公开数据集 Project | 不用于纠错 | 级联软删下级，仍保留物理文件 |

成功：DELETE `200`；再 GET 不可见；OSS 可重新入库。未归属草稿可能成孤儿，清理方式待确认。

---

## 给后端确认（需求 + 问题清单）

背景：公开数据集仅做**元信息入库**（OSS 已有 L2，不搬运 data/videos）。现网可用：建 Dataset / SubDataset（含 metadata）、单条建 Episode。缺口：SubDataset 无法写 `data_url`；Episode 创建无法指定 `sub_dataset_id`；挂载依赖验证上传清单；无批量写 Episode 接口。

### 一、需新增接口

`POST /api/v3/episodes/batch`

| 项 | 说明 |
|----|------|
| 用途 | 批量写入 Episode metadata，并在创建时归属到指定 SubDataset（公开集不走验证上传清单挂载） |
| 请求 | 必填 `sub_dataset_id`；必填 `episodes[]`（每项：`episode_name`、`metadata` 须含 `episode_index`；可选 `is_available`）。建议支持 Header `idempotency-key`。单次建议上限 200～500 条 |
| 响应 | 小批量同步 `201`（返回 `episode_ids` 等）；大批量可 `202` + `task_id`，客户端轮询至完成/失败（`202` 不算入库完成） |
| 权限 | 目标 SubDataset WRITE（或等价权限） |

```json
{
  "sub_dataset_id": "s_ds_…",
  "episodes": [
    { "episode_name": "0", "metadata": { "episode_index": 0, "length": 120 }, "is_available": true }
  ]
}
```

另请评估：SubDataset 创建/更新是否增加可写 `data_url`（或按 OSS 规范路径由服务端自动填充）。现网 `SubDatasetCreate` / `SubDatasetUpdate` 均无该字段。

### 二、待确认问题

| 序号 | 问题 | 对应接口 | 期望 / 备注 |
|------|------|----------|-------------|
| 1 | SubDataset 的 `data_url` 如何写入？（当前创建/更新接口无该字段） | `POST /sub-datasets/`、`PATCH /sub-datasets/{id}` | 允许传入，或按 OSS 规范路径自动填充 |
| 2 | `sub_dataset_name` 如何命名？ | `POST /sub-datasets/` | 是否约定与 split 一致（`train`/`val`/`test`），并与 `dataset_type`、OSS 目录对齐 |
| 3 | Episode 批量上传失败是否整批回滚？同一子集重复上传的 Episode 元信息如何处理（跳过/覆盖）？是否支持请求级幂等？大批量同步还是异步？ | `POST /episodes/batch`（需新增） | 期望整批回滚；重复不产生双份；幂等键可安全重试 |
| 4 | 删除数据集如何处理？当前接口是否会删除 OSS 数据？删 Dataset 是否级联全部 SubDataset？ | `DELETE /datasets/{dataset_id}`、`DELETE /sub-datasets/{sub_dataset_id}` | 公开集纠错仅清库、不删 OSS；请确认 UI 删除是否同此语义 |

依据现网 OpenAPI（`api-reference.md`，2026-11-03）。
