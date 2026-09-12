"""Step124: freeze and register the official multi-date research model."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SCRIPT_VERSION = "1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training/configs/multidate_model_finalization.yaml")
    parser.add_argument("--mode", choices=("check", "run"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle: value = yaml.safe_load(handle)
    if not isinstance(value, dict): raise ValueError("Config root must be a mapping")
    return value


def resolve(root: Path, value: str) -> Path:
    path = Path(value); return path if path.is_absolute() else root / path


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"); temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0]) if rows else []
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8"); temporary.replace(path)


def logger_for(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True); logger = logging.getLogger("step124"); logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size): digest.update(block)
    return digest.hexdigest()


def checkpoint_manifest(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    model = config["model"]; checkpoint_root = resolve(root, model["checkpoint_root"]); rows: list[dict[str, Any]] = []
    for seed in tqdm(model["official_ensemble_seeds"], desc="Registering official checkpoints", unit="checkpoint", dynamic_ncols=True):
        path = checkpoint_root / f"seed_{seed}" / f"{model['checkpoint_kind']}.pt"
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if int(checkpoint["seed"]) != int(seed): raise RuntimeError(f"Seed mismatch: {path}")
        rows.append({
            "model_id": model["model_id"], "seed": int(seed), "checkpoint_kind": model["checkpoint_kind"], "path": str(path),
            "size_bytes": path.stat().st_size, "sha256": sha256(path), "checkpoint_epoch": int(checkpoint["epoch"]),
            "best_epoch": int(checkpoint["best_epoch"]), "best_validation_total_loss": float(checkpoint["best_validation_total_loss"]),
            "config_fingerprint": checkpoint["config_fingerprint"], "strict_load_required": True,
        })
    if len({row["config_fingerprint"] for row in rows}) != 1: raise RuntimeError("Official checkpoint config fingerprints differ")
    return rows


def evidence_index(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    paths = [
        (115, "spatiotemporal split", config["inputs"]["step115_summary"]),
        (117, "formal five-seed training", config["inputs"]["step117_summary"]),
        (118, "frozen Test evaluation", config["inputs"]["step118_summary"]),
        (119, "generalization confidence intervals", "training/metrics/multidate_reporting/summary.json"),
        (120, "ablation implementation preflight", "training/reports/ablation_preflight/summary.json"),
        (121, "formal ablation training", "training/metrics/multidate_ablation/formal_training_summary.json"),
        (122, "frozen ablation evaluation", "training/metrics/multidate_ablation_evaluation/evaluation_summary.json"),
        (123, "paired bootstrap ablation reporting", config["inputs"]["step123_summary"]),
    ]
    rows: list[dict[str, Any]] = []
    for step, role, relative in tqdm(paths, desc="Indexing evidence", unit="step", dynamic_ncols=True):
        path = resolve(root, relative); payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"step": step, "role": role, "path": str(path), "status": payload.get("status", payload.get("state", "UNKNOWN")), "sha256": sha256(path)})
    return rows


def strict_metrics(config: dict[str, Any], root: Path) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(resolve(root, config["inputs"]["step118_metrics"]))
    frame = frame[(frame["dataset_split"] == "unseen_points_unseen_weather") & (frame["model"] == "ensemble_mean")]
    result: dict[str, dict[str, float]] = {}
    for target in tqdm(("shade", "tmrt_celsius", "utci_celsius"), desc="Freezing strict metrics", unit="target", dynamic_ncols=True):
        row = frame[frame["target"] == target].iloc[0]
        result[target] = {metric: float(row[f"{target}_{metric}"]) for metric in ("mae", "rmse", "r2", "bias", "pearson")}
    return result


def model_card(config: dict[str, Any], step115: dict[str, Any], metrics: dict[str, dict[str, float]], ranges: pd.DataFrame, registry: dict[str, Any]) -> str:
    train = ranges[ranges["weather_split"] == "train"].set_index("feature")
    def span(field: str, digits: int = 2) -> str:
        row = train.loc[field]; return f"{row['minimum']:.{digits}f}–{row['maximum']:.{digits}f}"
    dates = step115["dates"]
    return f"""# {config['model']['model_id']} 模型卡

## 模型定位

南京中心城区夏季高温情景下的街景遮荫率与平均辐射温度预测模型，并结合标准UTCI公式计算热舒适指标。正式版本为五种子完整模型集成，架构选择在Test评价前冻结。

## 正式模型

- 模型ID：`{config['model']['model_id']}`
- 架构：`{config['model']['architecture']}`
- 集成种子：{', '.join(map(str, config['model']['official_ensemble_seeds']))}
- 参数量：每个子模型1,491,778
- 输入：北向对齐全景的语义结构特征、全局DINOv2、8方向DINOv2、太阳几何和逐小时气象变量
- 直接输出：Shade rate、Tmrt
- 派生输出：UTCI（使用预测Tmrt与标准气象输入计算）

## 数据范围

- 点位：8,975个南京中心城区街景点
- 日期：14个2024年7–8月代表性夏季高温天气
- 小时：06:00–18:00
- Train日期：{', '.join(dates['train'])}
- Validation日期：{', '.join(dates['validation'])}
- Test日期：{', '.join(dates['test'])}
- Train气温范围：{span('air_temperature')} °C
- Train相对湿度范围：{span('relative_humidity')} %
- Train风速范围：{span('wind_speed')} m/s
- Train总短波辐射范围：{span('global_shortwave_radiation')} W/m²
- Train太阳高度角范围：{span('solar_altitude')}°

## 最严格冻结Test表现

“未见点位×未见天气”共34,996条记录：

| 目标 | MAE | RMSE | R² |
|---|---:|---:|---:|
| Shade rate | {metrics['shade']['mae']:.4f} | {metrics['shade']['rmse']:.4f} | {metrics['shade']['r2']:.4f} |
| Tmrt (°C) | {metrics['tmrt_celsius']['mae']:.4f} | {metrics['tmrt_celsius']['rmse']:.4f} | {metrics['tmrt_celsius']['r2']:.4f} |
| UTCI (°C) | {metrics['utci_celsius']['mae']:.4f} | {metrics['utci_celsius']['rmse']:.4f} | {metrics['utci_celsius']['r2']:.4f} |

## 消融证据

- 短波辐射、太阳几何和气温/湿度是跨天气Tmrt/UTCI泛化的核心变量。
- 方向DINOv2、绝对方位编码和方向交叉注意力主要支持遮荫的空间泛化。
- 显式语义结构、风速输入及Shade→Tmrt连接在当前结构中没有稳定增量，但不代表对应物理过程不重要。
- 消融结果仅用于解释，不用于Test后重新选择架构；正式模型仍保持预先冻结的完整五种子集成。

## 适用范围

适用于与训练范围相近的南京中心城区、夏季高温、06:00–18:00情景。其他相近日期可以在气象输入仍处于训练支持范围时进行预测，但应标记为模型外推风险，尤其是跨季节、跨城市、夜间、极端风雨和训练范围外气象条件。

## 重要限制

- 所有跨日期模拟复用2024-07-29的DSM、树木DSM和阴影几何，因此验证重点是气象变化，不是真实逐日期植被与阴影变化。
- 模型是SOLWEIG标签监督的代理模型，不是现场实测因果模型。
- UTCI对风速仍具有物理依赖；风速消融只检验风速是否帮助网络预测Tmrt。
- 规划干预收益应使用项目独立的单调响应模型或SOLWEIG情景模拟，不能直接把观测预测模型当作因果效应模型。

## 完整性与复现

- 正式checkpoint：{len(registry['official_checkpoints'])}个，均保存SHA-256与配置指纹。
- 模型选择：仅使用Validation点位×Validation天气。
- Test后架构切换：禁止。
- 运行环境、输入指纹、训练历史、Test预测、Bootstrap和消融证据均由Step115–Step123登记。
"""


def handoff_text(config: dict[str, Any]) -> str:
    return f"""# 推理交接说明

1. 使用`{config['model']['training_config']}`中的模型尺寸和输入字段顺序。
2. 加载`{config['model']['scaler']}`，禁止在新数据上重新拟合Scaler。
3. 严格加载登记表中的五个`best.pt`，分别预测Shade与Tmrt后取算术平均。
4. 使用`training/src/utci.py`和原始气温、相对湿度、风速计算每个种子的UTCI，再对UTCI取集成平均。
5. 输入必须保持北向全景和8方向窗口真实方位；不得自动旋转或重新估计方向。
6. 推理前检查气象变量是否超出模型卡中的Train范围，并在输出中保留外推标记。
7. 本登记只冻结研究模型，不自动替换现有路径规划网页中的生产模型；应用集成需要单独版本升级与回归测试。
"""


def run(config: dict[str, Any], root: Path, overwrite: bool) -> dict[str, Any]:
    output = resolve(root, config["outputs"]["registry_directory"]); output.mkdir(parents=True, exist_ok=True)
    registry_path = output / "model_registry.json"
    if registry_path.exists() and not overwrite: raise FileExistsError("Step124 registry exists; use --overwrite")
    step115 = json.loads(resolve(root, config["inputs"]["step115_summary"]).read_text(encoding="utf-8"))
    step117 = json.loads(resolve(root, config["inputs"]["step117_summary"]).read_text(encoding="utf-8"))
    step118 = json.loads(resolve(root, config["inputs"]["step118_summary"]).read_text(encoding="utf-8"))
    step123 = json.loads(resolve(root, config["inputs"]["step123_summary"]).read_text(encoding="utf-8"))
    for name, payload, expected in (("Step115", step115, "PASS"), ("Step117", step117, "COMPLETE"), ("Step118", step118, "COMPLETE"), ("Step123", step123, "PASS")):
        if payload.get("status") != expected: raise RuntimeError(f"{name} status is not {expected}")
    checkpoints = checkpoint_manifest(config, root); evidence = evidence_index(config, root); metrics = strict_metrics(config, root)
    ranges = pd.read_csv(resolve(root, config["inputs"]["dynamic_ranges"]))
    support = {split: {row.feature: {"minimum":float(row.minimum), "maximum":float(row.maximum), "mean":float(row.mean)} for row in group.itertuples()} for split, group in ranges.groupby("weather_split")}
    registry = {
        "model_id":config["model"]["model_id"], "version":"1.0.0", "frozen_at":dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "status":config["decision"]["deployment_status"], "architecture":config["model"]["architecture"], "official_model":config["decision"]["official_model"],
        "selection_basis":config["decision"]["selection_basis"], "ablations_role":config["decision"]["ablations_role"],
        "test_adaptive_architecture_change":False, "official_checkpoints":checkpoints, "strict_test_metrics":metrics,
        "weather_support":support, "spatiotemporal_split":step115, "input_fingerprints":step117["input_fingerprints"],
        "scaler":{"path":str(resolve(root, config["model"]["scaler"])), "sha256":sha256(resolve(root, config["model"]["scaler"]))},
        "training_config":{"path":str(resolve(root, config["model"]["training_config"])), "sha256":sha256(resolve(root, config["model"]["training_config"]))},
        "evidence_index":"evidence_index.csv", "checkpoint_manifest":"checkpoint_manifest.csv", "model_card":"MODEL_CARD.md", "inference_handoff":"INFERENCE_HANDOFF.md",
    }
    atomic_json(registry_path, registry); atomic_csv(output / "checkpoint_manifest.csv", checkpoints); atomic_csv(output / "evidence_index.csv", evidence)
    atomic_text(output / "MODEL_CARD.md", model_card(config, step115, metrics, ranges, registry)); atomic_text(output / "INFERENCE_HANDOFF.md", handoff_text(config))
    return {"step":124, "status":"PASS", "model_id":config["model"]["model_id"], "official_checkpoint_count":len(checkpoints), "evidence_steps":[row["step"] for row in evidence], "strict_test_metrics":metrics, "registry_directory":str(output), "test_adaptive_architecture_change":False, "training_performed":False, "inference_performed":False}


def main() -> int:
    args = parse_args(); config = load_yaml(args.config.resolve()); root = Path(config["project_root"])
    report_dir = resolve(root, config["outputs"]["report_directory"]); record_dir = resolve(root, config["outputs"]["record_directory"]); report_dir.mkdir(parents=True, exist_ok=True); record_dir.mkdir(parents=True, exist_ok=True)
    logger = logger_for(record_dir / "step124.log")
    try:
        required = [resolve(root, value) for value in config["inputs"].values()] + [resolve(root, config["model"][key]) for key in ("scaler", "training_config", "model_source", "inference_source", "utci_source")]
        required.extend(resolve(root, config["model"]["checkpoint_root"]) / f"seed_{seed}" / f"{config['model']['checkpoint_kind']}.pt" for seed in config["model"]["official_ensemble_seeds"])
        missing = [str(path) for path in required if not path.exists()]
        if missing: raise FileNotFoundError("Missing finalization inputs: " + "; ".join(missing))
        if args.mode == "check":
            result = {"status":"PASS", "model_id":config["model"]["model_id"], "required_file_count":len(required), "official_checkpoint_count":len(config["model"]["official_ensemble_seeds"]), "no_registry_written":True}
            atomic_json(record_dir / "preflight.json", result); print(json.dumps(result, ensure_ascii=True, indent=2)); return 0
        result = run(config, root, args.overwrite); atomic_json(record_dir / "summary.json", result); atomic_csv(record_dir / "failed_files.csv", [])
        report = f"# Step124：多日期正式模型冻结与登记\n\n状态：**PASS**。\n\n正式模型为`{result['model_id']}`五种子完整模型集成。Step123消融仅用于科学解释，没有在Test后更换架构、重选checkpoint或重拟合Scaler。\n\n关键交付位于`{result['registry_directory']}`：模型卡、推理交接、checkpoint SHA-256登记、证据索引与机器可读模型注册表。\n"
        atomic_text(report_dir / "STEP124_MULTIDATE_MODEL_FINALIZATION.md", report); logger.info("Step124 PASS | model=%s", result["model_id"]); print(json.dumps(result, ensure_ascii=True, indent=2)); return 0
    except Exception as error:
        logger.exception("Step124 failed"); atomic_json(record_dir / "failure.json", {"status":"FAIL", "error":str(error), "traceback":traceback.format_exc()}); return 1


if __name__ == "__main__": raise SystemExit(main())
