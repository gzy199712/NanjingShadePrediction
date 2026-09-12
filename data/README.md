# 数据目录

本目录只包含两类数据：

- `raw/`：不可修改的原始输入。
- `training_data/`：由正式 Pipeline 生成、模型训练必须使用的关键结果。

日志、QC、preview、ArcGIS 临时结果和其他中间文件统一放在 `artifacts/intermediate/`；模型 checkpoint 位于 `training/checkpoints/`。
