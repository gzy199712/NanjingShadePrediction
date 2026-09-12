# 数据流程

该目录只保留当前多天气训练表流程：晴热日筛选、SOLWEIG GPU 计算、多日期训练表构建和时空划分预检。

```powershell
python -m workflows.data_pipeline --stages 112 113 114 115
python -m workflows.data_pipeline --stages 112 113 114 115 --execute
```

不带 `--execute` 时只检查脚本和输出目录。输入、输出及质量门槛位于 `data_pipeline/configs/multiweather/`。
