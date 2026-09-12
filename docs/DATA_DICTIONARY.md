# 数据字典

- 自动生成时间：2026-07-24T14:43:17+08:00
- 字段清单直接读取正式CSV表头；类型由正式文件首行推断。

## dinov2_window_index.csv

- 来源：`data\training_data\static\dinov2_window_index.csv`
- 行数：8975

| 字段 | 类型 | 单位 | 含义 | 模型输入 | 预测目标 | 仅QC/验证 |
|---|---|---|---|---:|---:|---:|
| row_index | integer |  | row index | 否 | 否 | 否 |
| point_id | integer |  | point id | 否 | 否 | 否 |
| panorama_path | string |  | panorama path | 否 | 否 | 否 |
| window_count | integer |  | window count | 否 | 否 | 否 |
| embedding_dim | integer |  | embedding dim | 否 | 否 | 否 |
| azimuth_0 | float | degree or encoded unitless | azimuth 0 | 否 | 否 | 否 |
| azimuth_1 | float | degree or encoded unitless | azimuth 1 | 否 | 否 | 否 |
| azimuth_2 | float | degree or encoded unitless | azimuth 2 | 否 | 否 | 否 |
| azimuth_3 | float | degree or encoded unitless | azimuth 3 | 否 | 否 | 否 |
| azimuth_4 | float | degree or encoded unitless | azimuth 4 | 否 | 否 | 否 |
| azimuth_5 | float | degree or encoded unitless | azimuth 5 | 否 | 否 | 否 |
| azimuth_6 | float | degree or encoded unitless | azimuth 6 | 否 | 否 | 否 |
| azimuth_7 | float | degree or encoded unitless | azimuth 7 | 否 | 否 | 否 |

## main_sample_metadata.csv

- 来源：`data\training_data\spatial\main_sample_metadata.csv`
- 行数：8975

| 字段 | 类型 | 单位 | 含义 | 模型输入 | 预测目标 | 仅QC/验证 |
|---|---|---|---|---:|---:|---:|
| point_id | integer |  | point id | 否 | 否 | 否 |
| x | float | degree | x | 否 | 否 | 否 |
| y | float | degree | y | 否 | 否 | 否 |
| year | integer |  | year | 否 | 否 | 否 |
| month | integer |  | month | 否 | 否 | 否 |

## point_hour_labels_20240729.csv

- 来源：`data\training_data\dynamic\point_hour_labels_20240729.csv`
- 行数：116675

| 字段 | 类型 | 单位 | 含义 | 模型输入 | 预测目标 | 仅QC/验证 |
|---|---|---|---|---:|---:|---:|
| point_id | integer |  | point id | 否 | 否 | 否 |
| date | string |  | date | 否 | 否 | 否 |
| hour | integer |  | hour | 否 | 否 | 否 |
| datetime_local | string |  | datetime local | 否 | 否 | 否 |
| shade_rate | integer |  | shade rate | 否 | 是 | 否 |
| shade_nonshade_rate | integer |  | shade nonshade rate | 否 | 否 | 否 |
| shade_mean_raw | integer |  | shade mean raw | 否 | 否 | 否 |
| shade_valid_pixels | integer |  | shade valid pixels | 否 | 否 | 否 |
| shade_nodata_pixels | integer |  | shade nodata pixels | 否 | 否 | 否 |
| shade_total_candidate_pixels | integer |  | shade total candidate pixels | 否 | 否 | 否 |
| shade_valid_ratio | integer | 0–1 | shade valid ratio | 否 | 否 | 否 |
| tmrt_mean | float | °C | tmrt mean | 否 | 是 | 否 |
| tmrt_std | float | °C | tmrt std | 否 | 否 | 否 |
| tmrt_min | float | °C | tmrt min | 否 | 否 | 否 |
| tmrt_max | float | °C | tmrt max | 否 | 否 | 否 |
| tmrt_p10 | float | °C | tmrt p10 | 否 | 否 | 否 |
| tmrt_p90 | float | °C | tmrt p90 | 否 | 否 | 否 |
| tmrt_valid_pixels | integer | °C | tmrt valid pixels | 否 | 否 | 否 |
| tmrt_nodata_pixels | integer | °C | tmrt nodata pixels | 否 | 否 | 否 |
| tmrt_total_candidate_pixels | integer | °C | tmrt total candidate pixels | 否 | 否 | 否 |
| tmrt_valid_ratio | integer | 0–1 | tmrt valid ratio | 否 | 否 | 否 |
| utci_mean | float | °C | utci mean | 否 | 否 | 是（验证） |
| utci_std | float | °C | utci std | 否 | 否 | 否 |
| utci_min | float | °C | utci min | 否 | 否 | 否 |
| utci_max | float | °C | utci max | 否 | 否 | 否 |
| utci_p10 | float | °C | utci p10 | 否 | 否 | 否 |
| utci_p90 | float | °C | utci p90 | 否 | 否 | 否 |
| utci_valid_pixels | integer | °C | utci valid pixels | 否 | 否 | 否 |
| utci_nodata_pixels | integer | °C | utci nodata pixels | 否 | 否 | 否 |
| utci_total_candidate_pixels | integer | °C | utci total candidate pixels | 否 | 否 | 否 |
| utci_valid_ratio | integer | 0–1 | utci valid ratio | 否 | 否 | 否 |

## dynamic_conditions_20240729.csv

- 来源：`data\training_data\dynamic\dynamic_conditions_20240729.csv`
- 行数：13

| 字段 | 类型 | 单位 | 含义 | 模型输入 | 预测目标 | 仅QC/验证 |
|---|---|---|---|---:|---:|---:|
| date | string |  | date | 否 | 否 | 否 |
| hour | integer |  | hour | 否 | 否 | 否 |
| datetime_local | string |  | datetime local | 否 | 否 | 否 |
| representative_longitude | float |  | representative longitude | 否 | 否 | 否 |
| representative_latitude | float |  | representative latitude | 否 | 否 | 否 |
| solar_altitude | float | degree or encoded unitless | solar altitude | 是 | 否 | 否 |
| solar_azimuth | float | degree or encoded unitless | solar azimuth | 否 | 否 | 否 |
| sin_solar_azimuth | float | degree or encoded unitless | sin solar azimuth | 是 | 否 | 否 |
| cos_solar_azimuth | float | degree or encoded unitless | cos solar azimuth | 是 | 否 | 否 |
| sin_solar_altitude | float | degree or encoded unitless | sin solar altitude | 是 | 否 | 否 |
| cos_solar_altitude | float | degree or encoded unitless | cos solar altitude | 是 | 否 | 否 |
| air_temperature | float | °C | air temperature | 是 | 否 | 否 |
| relative_humidity | float |  | relative humidity | 是 | 否 | 否 |
| wind_speed | float | m/s | wind speed | 是 | 否 | 否 |
| wind_direction | float | degree or encoded unitless | wind direction | 否 | 否 | 否 |
| sin_wind_direction | float | degree or encoded unitless | sin wind direction | 否 | 否 | 否 |
| cos_wind_direction | float | degree or encoded unitless | cos wind direction | 否 | 否 | 否 |
| air_pressure_kpa | float | kPa | air pressure kpa | 否 | 否 | 否 |
| rainfall | float |  | rainfall | 否 | 否 | 否 |
| global_shortwave_radiation | float | W/m² | global shortwave radiation | 是 | 否 | 否 |
| direct_shortwave_radiation | float | W/m² | direct shortwave radiation | 是 | 否 | 否 |
| diffuse_shortwave_radiation | float | W/m² | diffuse shortwave radiation | 是 | 否 | 否 |

## semantic_features_8975.csv

- 来源：`data\training_data\static\semantic_features_8975.csv`
- 行数：8975

| 字段 | 类型 | 单位 | 含义 | 模型输入 | 预测目标 | 仅QC/验证 |
|---|---|---|---|---:|---:|---:|
| point_id | integer |  | point id | 否 | 否 | 否 |
| class_0_ratio | float | 0–1 | class 0 ratio | 是 | 否 | 否 |
| class_1_ratio | float | 0–1 | class 1 ratio | 是 | 否 | 否 |
| class_2_ratio | float | 0–1 | class 2 ratio | 是 | 否 | 否 |
| class_3_ratio | float | 0–1 | class 3 ratio | 是 | 否 | 否 |
| class_4_ratio | float | 0–1 | class 4 ratio | 是 | 否 | 否 |
| class_5_ratio | float | 0–1 | class 5 ratio | 是 | 否 | 否 |
| class_6_ratio | float | 0–1 | class 6 ratio | 是 | 否 | 否 |
| class_7_ratio | float | 0–1 | class 7 ratio | 是 | 否 | 否 |
| class_8_ratio | float | 0–1 | class 8 ratio | 是 | 否 | 否 |
| class_9_ratio | float | 0–1 | class 9 ratio | 是 | 否 | 否 |
| class_10_ratio | float | 0–1 | class 10 ratio | 是 | 否 | 否 |
| class_11_ratio | float | 0–1 | class 11 ratio | 是 | 否 | 否 |
| class_12_ratio | float | 0–1 | class 12 ratio | 是 | 否 | 否 |
| class_13_ratio | float | 0–1 | class 13 ratio | 是 | 否 | 否 |
| class_14_ratio | float | 0–1 | class 14 ratio | 是 | 否 | 否 |
| class_15_ratio | integer | 0–1 | class 15 ratio | 是 | 否 | 否 |
| class_16_ratio | integer | 0–1 | class 16 ratio | 是 | 否 | 否 |
| class_17_ratio | integer | 0–1 | class 17 ratio | 是 | 否 | 否 |
| class_18_ratio | integer | 0–1 | class 18 ratio | 是 | 否 | 否 |
| SVF | float | 0–1 | SVF | 是 | 否 | 否 |
| BVI | float | 0–1 | BVI | 是 | 否 | 否 |
| GVI | float | 0–1 | GVI | 是 | 否 | 否 |
| vegetation_patch_number | integer |  | vegetation patch number | 是 | 否 | 否 |
| largest_vegetation_patch_ratio | float | 0–1 | largest vegetation patch ratio | 是 | 否 | 否 |
| vegetation_fragmentation | float |  | vegetation fragmentation | 是 | 否 | 否 |
| sky_building_boundary_ratio | float | 0–1 | sky building boundary ratio | 是 | 否 | 否 |
| sky_vegetation_boundary_ratio | float | 0–1 | sky vegetation boundary ratio | 是 | 否 | 否 |

## dinov2_mean_features_8975.csv

- 来源：`data\training_data\static\dinov2_mean_features_8975.csv`
- 行数：8975

| 字段 | 类型 | 单位 | 含义 | 模型输入 | 预测目标 | 仅QC/验证 |
|---|---|---|---|---:|---:|---:|
| point_id | integer |  | point id | 否 | 否 | 否 |
| feature_001 | float |  | feature 001 | 是 | 否 | 否 |
| feature_002 | float |  | feature 002 | 是 | 否 | 否 |
| feature_003 | float |  | feature 003 | 是 | 否 | 否 |
| feature_004 | float |  | feature 004 | 是 | 否 | 否 |
| feature_005 | float |  | feature 005 | 是 | 否 | 否 |
| feature_006 | float |  | feature 006 | 是 | 否 | 否 |
| feature_007 | float |  | feature 007 | 是 | 否 | 否 |
| feature_008 | float |  | feature 008 | 是 | 否 | 否 |
| feature_009 | float |  | feature 009 | 是 | 否 | 否 |
| feature_010 | float |  | feature 010 | 是 | 否 | 否 |
| feature_011 | float |  | feature 011 | 是 | 否 | 否 |
| feature_012 | float |  | feature 012 | 是 | 否 | 否 |
| feature_013 | float |  | feature 013 | 是 | 否 | 否 |
| feature_014 | float |  | feature 014 | 是 | 否 | 否 |
| feature_015 | float |  | feature 015 | 是 | 否 | 否 |
| feature_016 | float |  | feature 016 | 是 | 否 | 否 |
| feature_017 | float |  | feature 017 | 是 | 否 | 否 |
| feature_018 | float |  | feature 018 | 是 | 否 | 否 |
| feature_019 | float |  | feature 019 | 是 | 否 | 否 |
| feature_020 | float |  | feature 020 | 是 | 否 | 否 |
| feature_021 | float |  | feature 021 | 是 | 否 | 否 |
| feature_022 | float |  | feature 022 | 是 | 否 | 否 |
| feature_023 | float |  | feature 023 | 是 | 否 | 否 |
| feature_024 | float |  | feature 024 | 是 | 否 | 否 |
| feature_025 | float |  | feature 025 | 是 | 否 | 否 |
| feature_026 | float |  | feature 026 | 是 | 否 | 否 |
| feature_027 | float |  | feature 027 | 是 | 否 | 否 |
| feature_028 | float |  | feature 028 | 是 | 否 | 否 |
| feature_029 | float |  | feature 029 | 是 | 否 | 否 |
| feature_030 | float |  | feature 030 | 是 | 否 | 否 |
| feature_031 | float |  | feature 031 | 是 | 否 | 否 |
| feature_032 | float |  | feature 032 | 是 | 否 | 否 |
| feature_033 | float |  | feature 033 | 是 | 否 | 否 |
| feature_034 | float |  | feature 034 | 是 | 否 | 否 |
| feature_035 | float |  | feature 035 | 是 | 否 | 否 |
| feature_036 | float |  | feature 036 | 是 | 否 | 否 |
| feature_037 | float |  | feature 037 | 是 | 否 | 否 |
| feature_038 | float |  | feature 038 | 是 | 否 | 否 |
| feature_039 | float |  | feature 039 | 是 | 否 | 否 |
| feature_040 | float |  | feature 040 | 是 | 否 | 否 |
| feature_041 | float |  | feature 041 | 是 | 否 | 否 |
| feature_042 | float |  | feature 042 | 是 | 否 | 否 |
| feature_043 | float |  | feature 043 | 是 | 否 | 否 |
| feature_044 | float |  | feature 044 | 是 | 否 | 否 |
| feature_045 | float |  | feature 045 | 是 | 否 | 否 |
| feature_046 | float |  | feature 046 | 是 | 否 | 否 |
| feature_047 | float |  | feature 047 | 是 | 否 | 否 |
| feature_048 | float |  | feature 048 | 是 | 否 | 否 |
| feature_049 | float |  | feature 049 | 是 | 否 | 否 |
| feature_050 | float |  | feature 050 | 是 | 否 | 否 |
| feature_051 | float |  | feature 051 | 是 | 否 | 否 |
| feature_052 | float |  | feature 052 | 是 | 否 | 否 |
| feature_053 | float |  | feature 053 | 是 | 否 | 否 |
| feature_054 | float |  | feature 054 | 是 | 否 | 否 |
| feature_055 | float |  | feature 055 | 是 | 否 | 否 |
| feature_056 | float |  | feature 056 | 是 | 否 | 否 |
| feature_057 | float |  | feature 057 | 是 | 否 | 否 |
| feature_058 | float |  | feature 058 | 是 | 否 | 否 |
| feature_059 | float |  | feature 059 | 是 | 否 | 否 |
| feature_060 | float |  | feature 060 | 是 | 否 | 否 |
| feature_061 | float |  | feature 061 | 是 | 否 | 否 |
| feature_062 | float |  | feature 062 | 是 | 否 | 否 |
| feature_063 | float |  | feature 063 | 是 | 否 | 否 |
| feature_064 | float |  | feature 064 | 是 | 否 | 否 |
| feature_065 | float |  | feature 065 | 是 | 否 | 否 |
| feature_066 | float |  | feature 066 | 是 | 否 | 否 |
| feature_067 | float |  | feature 067 | 是 | 否 | 否 |
| feature_068 | float |  | feature 068 | 是 | 否 | 否 |
| feature_069 | float |  | feature 069 | 是 | 否 | 否 |
| feature_070 | float |  | feature 070 | 是 | 否 | 否 |
| feature_071 | float |  | feature 071 | 是 | 否 | 否 |
| feature_072 | float |  | feature 072 | 是 | 否 | 否 |
| feature_073 | float |  | feature 073 | 是 | 否 | 否 |
| feature_074 | float |  | feature 074 | 是 | 否 | 否 |
| feature_075 | float |  | feature 075 | 是 | 否 | 否 |
| feature_076 | float |  | feature 076 | 是 | 否 | 否 |
| feature_077 | float |  | feature 077 | 是 | 否 | 否 |
| feature_078 | float |  | feature 078 | 是 | 否 | 否 |
| feature_079 | float |  | feature 079 | 是 | 否 | 否 |
| feature_080 | float |  | feature 080 | 是 | 否 | 否 |
| feature_081 | float |  | feature 081 | 是 | 否 | 否 |
| feature_082 | float |  | feature 082 | 是 | 否 | 否 |
| feature_083 | float |  | feature 083 | 是 | 否 | 否 |
| feature_084 | float |  | feature 084 | 是 | 否 | 否 |
| feature_085 | float |  | feature 085 | 是 | 否 | 否 |
| feature_086 | float |  | feature 086 | 是 | 否 | 否 |
| feature_087 | float |  | feature 087 | 是 | 否 | 否 |
| feature_088 | float |  | feature 088 | 是 | 否 | 否 |
| feature_089 | float |  | feature 089 | 是 | 否 | 否 |
| feature_090 | float |  | feature 090 | 是 | 否 | 否 |
| feature_091 | float |  | feature 091 | 是 | 否 | 否 |
| feature_092 | float |  | feature 092 | 是 | 否 | 否 |
| feature_093 | float |  | feature 093 | 是 | 否 | 否 |
| feature_094 | float |  | feature 094 | 是 | 否 | 否 |
| feature_095 | float |  | feature 095 | 是 | 否 | 否 |
| feature_096 | float |  | feature 096 | 是 | 否 | 否 |
| feature_097 | float |  | feature 097 | 是 | 否 | 否 |
| feature_098 | float |  | feature 098 | 是 | 否 | 否 |
| feature_099 | float |  | feature 099 | 是 | 否 | 否 |
| feature_100 | float |  | feature 100 | 是 | 否 | 否 |
| feature_101 | float |  | feature 101 | 是 | 否 | 否 |
| feature_102 | float |  | feature 102 | 是 | 否 | 否 |
| feature_103 | float |  | feature 103 | 是 | 否 | 否 |
| feature_104 | float |  | feature 104 | 是 | 否 | 否 |
| feature_105 | float |  | feature 105 | 是 | 否 | 否 |
| feature_106 | float |  | feature 106 | 是 | 否 | 否 |
| feature_107 | float |  | feature 107 | 是 | 否 | 否 |
| feature_108 | float |  | feature 108 | 是 | 否 | 否 |
| feature_109 | float |  | feature 109 | 是 | 否 | 否 |
| feature_110 | float |  | feature 110 | 是 | 否 | 否 |
| feature_111 | float |  | feature 111 | 是 | 否 | 否 |
| feature_112 | float |  | feature 112 | 是 | 否 | 否 |
| feature_113 | float |  | feature 113 | 是 | 否 | 否 |
| feature_114 | float |  | feature 114 | 是 | 否 | 否 |
| feature_115 | float |  | feature 115 | 是 | 否 | 否 |
| feature_116 | float |  | feature 116 | 是 | 否 | 否 |
| feature_117 | float |  | feature 117 | 是 | 否 | 否 |
| feature_118 | float |  | feature 118 | 是 | 否 | 否 |
| feature_119 | float |  | feature 119 | 是 | 否 | 否 |
| feature_120 | float |  | feature 120 | 是 | 否 | 否 |
| feature_121 | float |  | feature 121 | 是 | 否 | 否 |
| feature_122 | float |  | feature 122 | 是 | 否 | 否 |
| feature_123 | float |  | feature 123 | 是 | 否 | 否 |
| feature_124 | float |  | feature 124 | 是 | 否 | 否 |
| feature_125 | float |  | feature 125 | 是 | 否 | 否 |
| feature_126 | float |  | feature 126 | 是 | 否 | 否 |
| feature_127 | float |  | feature 127 | 是 | 否 | 否 |
| feature_128 | float |  | feature 128 | 是 | 否 | 否 |
| feature_129 | float |  | feature 129 | 是 | 否 | 否 |
| feature_130 | float |  | feature 130 | 是 | 否 | 否 |
| feature_131 | float |  | feature 131 | 是 | 否 | 否 |
| feature_132 | float |  | feature 132 | 是 | 否 | 否 |
| feature_133 | float |  | feature 133 | 是 | 否 | 否 |
| feature_134 | float |  | feature 134 | 是 | 否 | 否 |
| feature_135 | float |  | feature 135 | 是 | 否 | 否 |
| feature_136 | float |  | feature 136 | 是 | 否 | 否 |
| feature_137 | float |  | feature 137 | 是 | 否 | 否 |
| feature_138 | float |  | feature 138 | 是 | 否 | 否 |
| feature_139 | float |  | feature 139 | 是 | 否 | 否 |
| feature_140 | float |  | feature 140 | 是 | 否 | 否 |
| feature_141 | float |  | feature 141 | 是 | 否 | 否 |
| feature_142 | float |  | feature 142 | 是 | 否 | 否 |
| feature_143 | float |  | feature 143 | 是 | 否 | 否 |
| feature_144 | float |  | feature 144 | 是 | 否 | 否 |
| feature_145 | float |  | feature 145 | 是 | 否 | 否 |
| feature_146 | float |  | feature 146 | 是 | 否 | 否 |
| feature_147 | float |  | feature 147 | 是 | 否 | 否 |
| feature_148 | float |  | feature 148 | 是 | 否 | 否 |
| feature_149 | float |  | feature 149 | 是 | 否 | 否 |
| feature_150 | float |  | feature 150 | 是 | 否 | 否 |
| feature_151 | float |  | feature 151 | 是 | 否 | 否 |
| feature_152 | float |  | feature 152 | 是 | 否 | 否 |
| feature_153 | float |  | feature 153 | 是 | 否 | 否 |
| feature_154 | float |  | feature 154 | 是 | 否 | 否 |
| feature_155 | float |  | feature 155 | 是 | 否 | 否 |
| feature_156 | float |  | feature 156 | 是 | 否 | 否 |
| feature_157 | float |  | feature 157 | 是 | 否 | 否 |
| feature_158 | float |  | feature 158 | 是 | 否 | 否 |
| feature_159 | float |  | feature 159 | 是 | 否 | 否 |
| feature_160 | float |  | feature 160 | 是 | 否 | 否 |
| feature_161 | float |  | feature 161 | 是 | 否 | 否 |
| feature_162 | float |  | feature 162 | 是 | 否 | 否 |
| feature_163 | float |  | feature 163 | 是 | 否 | 否 |
| feature_164 | float |  | feature 164 | 是 | 否 | 否 |
| feature_165 | float |  | feature 165 | 是 | 否 | 否 |
| feature_166 | float |  | feature 166 | 是 | 否 | 否 |
| feature_167 | float |  | feature 167 | 是 | 否 | 否 |
| feature_168 | float |  | feature 168 | 是 | 否 | 否 |
| feature_169 | float |  | feature 169 | 是 | 否 | 否 |
| feature_170 | float |  | feature 170 | 是 | 否 | 否 |
| feature_171 | float |  | feature 171 | 是 | 否 | 否 |
| feature_172 | float |  | feature 172 | 是 | 否 | 否 |
| feature_173 | float |  | feature 173 | 是 | 否 | 否 |
| feature_174 | float |  | feature 174 | 是 | 否 | 否 |
| feature_175 | float |  | feature 175 | 是 | 否 | 否 |
| feature_176 | float |  | feature 176 | 是 | 否 | 否 |
| feature_177 | float |  | feature 177 | 是 | 否 | 否 |
| feature_178 | float |  | feature 178 | 是 | 否 | 否 |
| feature_179 | float |  | feature 179 | 是 | 否 | 否 |
| feature_180 | float |  | feature 180 | 是 | 否 | 否 |
| feature_181 | float |  | feature 181 | 是 | 否 | 否 |
| feature_182 | float |  | feature 182 | 是 | 否 | 否 |
| feature_183 | float |  | feature 183 | 是 | 否 | 否 |
| feature_184 | float |  | feature 184 | 是 | 否 | 否 |
| feature_185 | float |  | feature 185 | 是 | 否 | 否 |
| feature_186 | float |  | feature 186 | 是 | 否 | 否 |
| feature_187 | float |  | feature 187 | 是 | 否 | 否 |
| feature_188 | float |  | feature 188 | 是 | 否 | 否 |
| feature_189 | float |  | feature 189 | 是 | 否 | 否 |
| feature_190 | float |  | feature 190 | 是 | 否 | 否 |
| feature_191 | float |  | feature 191 | 是 | 否 | 否 |
| feature_192 | float |  | feature 192 | 是 | 否 | 否 |
| feature_193 | float |  | feature 193 | 是 | 否 | 否 |
| feature_194 | float |  | feature 194 | 是 | 否 | 否 |
| feature_195 | float |  | feature 195 | 是 | 否 | 否 |
| feature_196 | float |  | feature 196 | 是 | 否 | 否 |
| feature_197 | float |  | feature 197 | 是 | 否 | 否 |
| feature_198 | float |  | feature 198 | 是 | 否 | 否 |
| feature_199 | float |  | feature 199 | 是 | 否 | 否 |
| feature_200 | float |  | feature 200 | 是 | 否 | 否 |
| feature_201 | float |  | feature 201 | 是 | 否 | 否 |
| feature_202 | float |  | feature 202 | 是 | 否 | 否 |
| feature_203 | float |  | feature 203 | 是 | 否 | 否 |
| feature_204 | float |  | feature 204 | 是 | 否 | 否 |
| feature_205 | float |  | feature 205 | 是 | 否 | 否 |
| feature_206 | float |  | feature 206 | 是 | 否 | 否 |
| feature_207 | float |  | feature 207 | 是 | 否 | 否 |
| feature_208 | float |  | feature 208 | 是 | 否 | 否 |
| feature_209 | float |  | feature 209 | 是 | 否 | 否 |
| feature_210 | float |  | feature 210 | 是 | 否 | 否 |
| feature_211 | float |  | feature 211 | 是 | 否 | 否 |
| feature_212 | float |  | feature 212 | 是 | 否 | 否 |
| feature_213 | float |  | feature 213 | 是 | 否 | 否 |
| feature_214 | float |  | feature 214 | 是 | 否 | 否 |
| feature_215 | float |  | feature 215 | 是 | 否 | 否 |
| feature_216 | float |  | feature 216 | 是 | 否 | 否 |
| feature_217 | float |  | feature 217 | 是 | 否 | 否 |
| feature_218 | float |  | feature 218 | 是 | 否 | 否 |
| feature_219 | float |  | feature 219 | 是 | 否 | 否 |
| feature_220 | float |  | feature 220 | 是 | 否 | 否 |
| feature_221 | float |  | feature 221 | 是 | 否 | 否 |
| feature_222 | float |  | feature 222 | 是 | 否 | 否 |
| feature_223 | float |  | feature 223 | 是 | 否 | 否 |
| feature_224 | float |  | feature 224 | 是 | 否 | 否 |
| feature_225 | float |  | feature 225 | 是 | 否 | 否 |
| feature_226 | float |  | feature 226 | 是 | 否 | 否 |
| feature_227 | float |  | feature 227 | 是 | 否 | 否 |
| feature_228 | float |  | feature 228 | 是 | 否 | 否 |
| feature_229 | float |  | feature 229 | 是 | 否 | 否 |
| feature_230 | float |  | feature 230 | 是 | 否 | 否 |
| feature_231 | float |  | feature 231 | 是 | 否 | 否 |
| feature_232 | float |  | feature 232 | 是 | 否 | 否 |
| feature_233 | float |  | feature 233 | 是 | 否 | 否 |
| feature_234 | float |  | feature 234 | 是 | 否 | 否 |
| feature_235 | float |  | feature 235 | 是 | 否 | 否 |
| feature_236 | float |  | feature 236 | 是 | 否 | 否 |
| feature_237 | float |  | feature 237 | 是 | 否 | 否 |
| feature_238 | float |  | feature 238 | 是 | 否 | 否 |
| feature_239 | float |  | feature 239 | 是 | 否 | 否 |
| feature_240 | float |  | feature 240 | 是 | 否 | 否 |
| feature_241 | float |  | feature 241 | 是 | 否 | 否 |
| feature_242 | float |  | feature 242 | 是 | 否 | 否 |
| feature_243 | float |  | feature 243 | 是 | 否 | 否 |
| feature_244 | float |  | feature 244 | 是 | 否 | 否 |
| feature_245 | float |  | feature 245 | 是 | 否 | 否 |
| feature_246 | float |  | feature 246 | 是 | 否 | 否 |
| feature_247 | float |  | feature 247 | 是 | 否 | 否 |
| feature_248 | float |  | feature 248 | 是 | 否 | 否 |
| feature_249 | float |  | feature 249 | 是 | 否 | 否 |
| feature_250 | float |  | feature 250 | 是 | 否 | 否 |
| feature_251 | float |  | feature 251 | 是 | 否 | 否 |
| feature_252 | float |  | feature 252 | 是 | 否 | 否 |
| feature_253 | float |  | feature 253 | 是 | 否 | 否 |
| feature_254 | float |  | feature 254 | 是 | 否 | 否 |
| feature_255 | float |  | feature 255 | 是 | 否 | 否 |
| feature_256 | float |  | feature 256 | 是 | 否 | 否 |
| feature_257 | float |  | feature 257 | 是 | 否 | 否 |
| feature_258 | float |  | feature 258 | 是 | 否 | 否 |
| feature_259 | float |  | feature 259 | 是 | 否 | 否 |
| feature_260 | float |  | feature 260 | 是 | 否 | 否 |
| feature_261 | float |  | feature 261 | 是 | 否 | 否 |
| feature_262 | float |  | feature 262 | 是 | 否 | 否 |
| feature_263 | float |  | feature 263 | 是 | 否 | 否 |
| feature_264 | float |  | feature 264 | 是 | 否 | 否 |
| feature_265 | float |  | feature 265 | 是 | 否 | 否 |
| feature_266 | float |  | feature 266 | 是 | 否 | 否 |
| feature_267 | float |  | feature 267 | 是 | 否 | 否 |
| feature_268 | float |  | feature 268 | 是 | 否 | 否 |
| feature_269 | float |  | feature 269 | 是 | 否 | 否 |
| feature_270 | float |  | feature 270 | 是 | 否 | 否 |
| feature_271 | float |  | feature 271 | 是 | 否 | 否 |
| feature_272 | float |  | feature 272 | 是 | 否 | 否 |
| feature_273 | float |  | feature 273 | 是 | 否 | 否 |
| feature_274 | float |  | feature 274 | 是 | 否 | 否 |
| feature_275 | float |  | feature 275 | 是 | 否 | 否 |
| feature_276 | float |  | feature 276 | 是 | 否 | 否 |
| feature_277 | float |  | feature 277 | 是 | 否 | 否 |
| feature_278 | float |  | feature 278 | 是 | 否 | 否 |
| feature_279 | float |  | feature 279 | 是 | 否 | 否 |
| feature_280 | float |  | feature 280 | 是 | 否 | 否 |
| feature_281 | float |  | feature 281 | 是 | 否 | 否 |
| feature_282 | float |  | feature 282 | 是 | 否 | 否 |
| feature_283 | float |  | feature 283 | 是 | 否 | 否 |
| feature_284 | float |  | feature 284 | 是 | 否 | 否 |
| feature_285 | float |  | feature 285 | 是 | 否 | 否 |
| feature_286 | float |  | feature 286 | 是 | 否 | 否 |
| feature_287 | float |  | feature 287 | 是 | 否 | 否 |
| feature_288 | float |  | feature 288 | 是 | 否 | 否 |
| feature_289 | float |  | feature 289 | 是 | 否 | 否 |
| feature_290 | float |  | feature 290 | 是 | 否 | 否 |
| feature_291 | float |  | feature 291 | 是 | 否 | 否 |
| feature_292 | float |  | feature 292 | 是 | 否 | 否 |
| feature_293 | float |  | feature 293 | 是 | 否 | 否 |
| feature_294 | float |  | feature 294 | 是 | 否 | 否 |
| feature_295 | float |  | feature 295 | 是 | 否 | 否 |
| feature_296 | float |  | feature 296 | 是 | 否 | 否 |
| feature_297 | float |  | feature 297 | 是 | 否 | 否 |
| feature_298 | float |  | feature 298 | 是 | 否 | 否 |
| feature_299 | float |  | feature 299 | 是 | 否 | 否 |
| feature_300 | float |  | feature 300 | 是 | 否 | 否 |
| feature_301 | float |  | feature 301 | 是 | 否 | 否 |
| feature_302 | float |  | feature 302 | 是 | 否 | 否 |
| feature_303 | float |  | feature 303 | 是 | 否 | 否 |
| feature_304 | float |  | feature 304 | 是 | 否 | 否 |
| feature_305 | float |  | feature 305 | 是 | 否 | 否 |
| feature_306 | float |  | feature 306 | 是 | 否 | 否 |
| feature_307 | float |  | feature 307 | 是 | 否 | 否 |
| feature_308 | float |  | feature 308 | 是 | 否 | 否 |
| feature_309 | float |  | feature 309 | 是 | 否 | 否 |
| feature_310 | float |  | feature 310 | 是 | 否 | 否 |
| feature_311 | float |  | feature 311 | 是 | 否 | 否 |
| feature_312 | float |  | feature 312 | 是 | 否 | 否 |
| feature_313 | float |  | feature 313 | 是 | 否 | 否 |
| feature_314 | float |  | feature 314 | 是 | 否 | 否 |
| feature_315 | float |  | feature 315 | 是 | 否 | 否 |
| feature_316 | float |  | feature 316 | 是 | 否 | 否 |
| feature_317 | float |  | feature 317 | 是 | 否 | 否 |
| feature_318 | float |  | feature 318 | 是 | 否 | 否 |
| feature_319 | float |  | feature 319 | 是 | 否 | 否 |
| feature_320 | float |  | feature 320 | 是 | 否 | 否 |
| feature_321 | float |  | feature 321 | 是 | 否 | 否 |
| feature_322 | float |  | feature 322 | 是 | 否 | 否 |
| feature_323 | float |  | feature 323 | 是 | 否 | 否 |
| feature_324 | float |  | feature 324 | 是 | 否 | 否 |
| feature_325 | float |  | feature 325 | 是 | 否 | 否 |
| feature_326 | float |  | feature 326 | 是 | 否 | 否 |
| feature_327 | float |  | feature 327 | 是 | 否 | 否 |
| feature_328 | float |  | feature 328 | 是 | 否 | 否 |
| feature_329 | float |  | feature 329 | 是 | 否 | 否 |
| feature_330 | float |  | feature 330 | 是 | 否 | 否 |
| feature_331 | float |  | feature 331 | 是 | 否 | 否 |
| feature_332 | float |  | feature 332 | 是 | 否 | 否 |
| feature_333 | float |  | feature 333 | 是 | 否 | 否 |
| feature_334 | float |  | feature 334 | 是 | 否 | 否 |
| feature_335 | float |  | feature 335 | 是 | 否 | 否 |
| feature_336 | float |  | feature 336 | 是 | 否 | 否 |
| feature_337 | float |  | feature 337 | 是 | 否 | 否 |
| feature_338 | float |  | feature 338 | 是 | 否 | 否 |
| feature_339 | float |  | feature 339 | 是 | 否 | 否 |
| feature_340 | float |  | feature 340 | 是 | 否 | 否 |
| feature_341 | float |  | feature 341 | 是 | 否 | 否 |
| feature_342 | float |  | feature 342 | 是 | 否 | 否 |
| feature_343 | float |  | feature 343 | 是 | 否 | 否 |
| feature_344 | float |  | feature 344 | 是 | 否 | 否 |
| feature_345 | float |  | feature 345 | 是 | 否 | 否 |
| feature_346 | float |  | feature 346 | 是 | 否 | 否 |
| feature_347 | float |  | feature 347 | 是 | 否 | 否 |
| feature_348 | float |  | feature 348 | 是 | 否 | 否 |
| feature_349 | float |  | feature 349 | 是 | 否 | 否 |
| feature_350 | float |  | feature 350 | 是 | 否 | 否 |
| feature_351 | float |  | feature 351 | 是 | 否 | 否 |
| feature_352 | float |  | feature 352 | 是 | 否 | 否 |
| feature_353 | float |  | feature 353 | 是 | 否 | 否 |
| feature_354 | float |  | feature 354 | 是 | 否 | 否 |
| feature_355 | float |  | feature 355 | 是 | 否 | 否 |
| feature_356 | float |  | feature 356 | 是 | 否 | 否 |
| feature_357 | float |  | feature 357 | 是 | 否 | 否 |
| feature_358 | float |  | feature 358 | 是 | 否 | 否 |
| feature_359 | float |  | feature 359 | 是 | 否 | 否 |
| feature_360 | float |  | feature 360 | 是 | 否 | 否 |
| feature_361 | float |  | feature 361 | 是 | 否 | 否 |
| feature_362 | float |  | feature 362 | 是 | 否 | 否 |
| feature_363 | float |  | feature 363 | 是 | 否 | 否 |
| feature_364 | float |  | feature 364 | 是 | 否 | 否 |
| feature_365 | float |  | feature 365 | 是 | 否 | 否 |
| feature_366 | float |  | feature 366 | 是 | 否 | 否 |
| feature_367 | float |  | feature 367 | 是 | 否 | 否 |
| feature_368 | float |  | feature 368 | 是 | 否 | 否 |
| feature_369 | float |  | feature 369 | 是 | 否 | 否 |
| feature_370 | float |  | feature 370 | 是 | 否 | 否 |
| feature_371 | float |  | feature 371 | 是 | 否 | 否 |
| feature_372 | float |  | feature 372 | 是 | 否 | 否 |
| feature_373 | float |  | feature 373 | 是 | 否 | 否 |
| feature_374 | float |  | feature 374 | 是 | 否 | 否 |
| feature_375 | float |  | feature 375 | 是 | 否 | 否 |
| feature_376 | float |  | feature 376 | 是 | 否 | 否 |
| feature_377 | float |  | feature 377 | 是 | 否 | 否 |
| feature_378 | float |  | feature 378 | 是 | 否 | 否 |
| feature_379 | float |  | feature 379 | 是 | 否 | 否 |
| feature_380 | float |  | feature 380 | 是 | 否 | 否 |
| feature_381 | float |  | feature 381 | 是 | 否 | 否 |
| feature_382 | float |  | feature 382 | 是 | 否 | 否 |
| feature_383 | float |  | feature 383 | 是 | 否 | 否 |
| feature_384 | float |  | feature 384 | 是 | 否 | 否 |
| feature_385 | float |  | feature 385 | 是 | 否 | 否 |
| feature_386 | float |  | feature 386 | 是 | 否 | 否 |
| feature_387 | float |  | feature 387 | 是 | 否 | 否 |
| feature_388 | float |  | feature 388 | 是 | 否 | 否 |
| feature_389 | float |  | feature 389 | 是 | 否 | 否 |
| feature_390 | float |  | feature 390 | 是 | 否 | 否 |
| feature_391 | float |  | feature 391 | 是 | 否 | 否 |
| feature_392 | float |  | feature 392 | 是 | 否 | 否 |
| feature_393 | float |  | feature 393 | 是 | 否 | 否 |
| feature_394 | float |  | feature 394 | 是 | 否 | 否 |
| feature_395 | float |  | feature 395 | 是 | 否 | 否 |
| feature_396 | float |  | feature 396 | 是 | 否 | 否 |
| feature_397 | float |  | feature 397 | 是 | 否 | 否 |
| feature_398 | float |  | feature 398 | 是 | 否 | 否 |
| feature_399 | float |  | feature 399 | 是 | 否 | 否 |
| feature_400 | float |  | feature 400 | 是 | 否 | 否 |
| feature_401 | float |  | feature 401 | 是 | 否 | 否 |
| feature_402 | float |  | feature 402 | 是 | 否 | 否 |
| feature_403 | float |  | feature 403 | 是 | 否 | 否 |
| feature_404 | float |  | feature 404 | 是 | 否 | 否 |
| feature_405 | float |  | feature 405 | 是 | 否 | 否 |
| feature_406 | float |  | feature 406 | 是 | 否 | 否 |
| feature_407 | float |  | feature 407 | 是 | 否 | 否 |
| feature_408 | float |  | feature 408 | 是 | 否 | 否 |
| feature_409 | float |  | feature 409 | 是 | 否 | 否 |
| feature_410 | float |  | feature 410 | 是 | 否 | 否 |
| feature_411 | float |  | feature 411 | 是 | 否 | 否 |
| feature_412 | float |  | feature 412 | 是 | 否 | 否 |
| feature_413 | float |  | feature 413 | 是 | 否 | 否 |
| feature_414 | float |  | feature 414 | 是 | 否 | 否 |
| feature_415 | float |  | feature 415 | 是 | 否 | 否 |
| feature_416 | float |  | feature 416 | 是 | 否 | 否 |
| feature_417 | float |  | feature 417 | 是 | 否 | 否 |
| feature_418 | float |  | feature 418 | 是 | 否 | 否 |
| feature_419 | float |  | feature 419 | 是 | 否 | 否 |
| feature_420 | float |  | feature 420 | 是 | 否 | 否 |
| feature_421 | float |  | feature 421 | 是 | 否 | 否 |
| feature_422 | float |  | feature 422 | 是 | 否 | 否 |
| feature_423 | float |  | feature 423 | 是 | 否 | 否 |
| feature_424 | float |  | feature 424 | 是 | 否 | 否 |
| feature_425 | float |  | feature 425 | 是 | 否 | 否 |
| feature_426 | float |  | feature 426 | 是 | 否 | 否 |
| feature_427 | float |  | feature 427 | 是 | 否 | 否 |
| feature_428 | float |  | feature 428 | 是 | 否 | 否 |
| feature_429 | float |  | feature 429 | 是 | 否 | 否 |
| feature_430 | float |  | feature 430 | 是 | 否 | 否 |
| feature_431 | float |  | feature 431 | 是 | 否 | 否 |
| feature_432 | float |  | feature 432 | 是 | 否 | 否 |
| feature_433 | float |  | feature 433 | 是 | 否 | 否 |
| feature_434 | float |  | feature 434 | 是 | 否 | 否 |
| feature_435 | float |  | feature 435 | 是 | 否 | 否 |
| feature_436 | float |  | feature 436 | 是 | 否 | 否 |
| feature_437 | float |  | feature 437 | 是 | 否 | 否 |
| feature_438 | float |  | feature 438 | 是 | 否 | 否 |
| feature_439 | float |  | feature 439 | 是 | 否 | 否 |
| feature_440 | float |  | feature 440 | 是 | 否 | 否 |
| feature_441 | float |  | feature 441 | 是 | 否 | 否 |
| feature_442 | float |  | feature 442 | 是 | 否 | 否 |
| feature_443 | float |  | feature 443 | 是 | 否 | 否 |
| feature_444 | float |  | feature 444 | 是 | 否 | 否 |
| feature_445 | float |  | feature 445 | 是 | 否 | 否 |
| feature_446 | float |  | feature 446 | 是 | 否 | 否 |
| feature_447 | float |  | feature 447 | 是 | 否 | 否 |
| feature_448 | float |  | feature 448 | 是 | 否 | 否 |
| feature_449 | float |  | feature 449 | 是 | 否 | 否 |
| feature_450 | float |  | feature 450 | 是 | 否 | 否 |
| feature_451 | float |  | feature 451 | 是 | 否 | 否 |
| feature_452 | float |  | feature 452 | 是 | 否 | 否 |
| feature_453 | float |  | feature 453 | 是 | 否 | 否 |
| feature_454 | float |  | feature 454 | 是 | 否 | 否 |
| feature_455 | float |  | feature 455 | 是 | 否 | 否 |
| feature_456 | float |  | feature 456 | 是 | 否 | 否 |
| feature_457 | float |  | feature 457 | 是 | 否 | 否 |
| feature_458 | float |  | feature 458 | 是 | 否 | 否 |
| feature_459 | float |  | feature 459 | 是 | 否 | 否 |
| feature_460 | float |  | feature 460 | 是 | 否 | 否 |
| feature_461 | float |  | feature 461 | 是 | 否 | 否 |
| feature_462 | float |  | feature 462 | 是 | 否 | 否 |
| feature_463 | float |  | feature 463 | 是 | 否 | 否 |
| feature_464 | float |  | feature 464 | 是 | 否 | 否 |
| feature_465 | float |  | feature 465 | 是 | 否 | 否 |
| feature_466 | float |  | feature 466 | 是 | 否 | 否 |
| feature_467 | float |  | feature 467 | 是 | 否 | 否 |
| feature_468 | float |  | feature 468 | 是 | 否 | 否 |
| feature_469 | float |  | feature 469 | 是 | 否 | 否 |
| feature_470 | float |  | feature 470 | 是 | 否 | 否 |
| feature_471 | float |  | feature 471 | 是 | 否 | 否 |
| feature_472 | float |  | feature 472 | 是 | 否 | 否 |
| feature_473 | float |  | feature 473 | 是 | 否 | 否 |
| feature_474 | float |  | feature 474 | 是 | 否 | 否 |
| feature_475 | float |  | feature 475 | 是 | 否 | 否 |
| feature_476 | float |  | feature 476 | 是 | 否 | 否 |
| feature_477 | float |  | feature 477 | 是 | 否 | 否 |
| feature_478 | float |  | feature 478 | 是 | 否 | 否 |
| feature_479 | float |  | feature 479 | 是 | 否 | 否 |
| feature_480 | float |  | feature 480 | 是 | 否 | 否 |
| feature_481 | float |  | feature 481 | 是 | 否 | 否 |
| feature_482 | float |  | feature 482 | 是 | 否 | 否 |
| feature_483 | float |  | feature 483 | 是 | 否 | 否 |
| feature_484 | float |  | feature 484 | 是 | 否 | 否 |
| feature_485 | float |  | feature 485 | 是 | 否 | 否 |
| feature_486 | float |  | feature 486 | 是 | 否 | 否 |
| feature_487 | float |  | feature 487 | 是 | 否 | 否 |
| feature_488 | float |  | feature 488 | 是 | 否 | 否 |
| feature_489 | float |  | feature 489 | 是 | 否 | 否 |
| feature_490 | float |  | feature 490 | 是 | 否 | 否 |
| feature_491 | float |  | feature 491 | 是 | 否 | 否 |
| feature_492 | float |  | feature 492 | 是 | 否 | 否 |
| feature_493 | float |  | feature 493 | 是 | 否 | 否 |
| feature_494 | float |  | feature 494 | 是 | 否 | 否 |
| feature_495 | float |  | feature 495 | 是 | 否 | 否 |
| feature_496 | float |  | feature 496 | 是 | 否 | 否 |
| feature_497 | float |  | feature 497 | 是 | 否 | 否 |
| feature_498 | float |  | feature 498 | 是 | 否 | 否 |
| feature_499 | float |  | feature 499 | 是 | 否 | 否 |
| feature_500 | float |  | feature 500 | 是 | 否 | 否 |
| feature_501 | float |  | feature 501 | 是 | 否 | 否 |
| feature_502 | float |  | feature 502 | 是 | 否 | 否 |
| feature_503 | float |  | feature 503 | 是 | 否 | 否 |
| feature_504 | float |  | feature 504 | 是 | 否 | 否 |
| feature_505 | float |  | feature 505 | 是 | 否 | 否 |
| feature_506 | float |  | feature 506 | 是 | 否 | 否 |
| feature_507 | float |  | feature 507 | 是 | 否 | 否 |
| feature_508 | float |  | feature 508 | 是 | 否 | 否 |
| feature_509 | float |  | feature 509 | 是 | 否 | 否 |
| feature_510 | float |  | feature 510 | 是 | 否 | 否 |
| feature_511 | float |  | feature 511 | 是 | 否 | 否 |
| feature_512 | float |  | feature 512 | 是 | 否 | 否 |
| feature_513 | float |  | feature 513 | 是 | 否 | 否 |
| feature_514 | float |  | feature 514 | 是 | 否 | 否 |
| feature_515 | float |  | feature 515 | 是 | 否 | 否 |
| feature_516 | float |  | feature 516 | 是 | 否 | 否 |
| feature_517 | float |  | feature 517 | 是 | 否 | 否 |
| feature_518 | float |  | feature 518 | 是 | 否 | 否 |
| feature_519 | float |  | feature 519 | 是 | 否 | 否 |
| feature_520 | float |  | feature 520 | 是 | 否 | 否 |
| feature_521 | float |  | feature 521 | 是 | 否 | 否 |
| feature_522 | float |  | feature 522 | 是 | 否 | 否 |
| feature_523 | float |  | feature 523 | 是 | 否 | 否 |
| feature_524 | float |  | feature 524 | 是 | 否 | 否 |
| feature_525 | float |  | feature 525 | 是 | 否 | 否 |
| feature_526 | float |  | feature 526 | 是 | 否 | 否 |
| feature_527 | float |  | feature 527 | 是 | 否 | 否 |
| feature_528 | float |  | feature 528 | 是 | 否 | 否 |
| feature_529 | float |  | feature 529 | 是 | 否 | 否 |
| feature_530 | float |  | feature 530 | 是 | 否 | 否 |
| feature_531 | float |  | feature 531 | 是 | 否 | 否 |
| feature_532 | float |  | feature 532 | 是 | 否 | 否 |
| feature_533 | float |  | feature 533 | 是 | 否 | 否 |
| feature_534 | float |  | feature 534 | 是 | 否 | 否 |
| feature_535 | float |  | feature 535 | 是 | 否 | 否 |
| feature_536 | float |  | feature 536 | 是 | 否 | 否 |
| feature_537 | float |  | feature 537 | 是 | 否 | 否 |
| feature_538 | float |  | feature 538 | 是 | 否 | 否 |
| feature_539 | float |  | feature 539 | 是 | 否 | 否 |
| feature_540 | float |  | feature 540 | 是 | 否 | 否 |
| feature_541 | float |  | feature 541 | 是 | 否 | 否 |
| feature_542 | float |  | feature 542 | 是 | 否 | 否 |
| feature_543 | float |  | feature 543 | 是 | 否 | 否 |
| feature_544 | float |  | feature 544 | 是 | 否 | 否 |
| feature_545 | float |  | feature 545 | 是 | 否 | 否 |
| feature_546 | float |  | feature 546 | 是 | 否 | 否 |
| feature_547 | float |  | feature 547 | 是 | 否 | 否 |
| feature_548 | float |  | feature 548 | 是 | 否 | 否 |
| feature_549 | float |  | feature 549 | 是 | 否 | 否 |
| feature_550 | float |  | feature 550 | 是 | 否 | 否 |
| feature_551 | float |  | feature 551 | 是 | 否 | 否 |
| feature_552 | float |  | feature 552 | 是 | 否 | 否 |
| feature_553 | float |  | feature 553 | 是 | 否 | 否 |
| feature_554 | float |  | feature 554 | 是 | 否 | 否 |
| feature_555 | float |  | feature 555 | 是 | 否 | 否 |
| feature_556 | float |  | feature 556 | 是 | 否 | 否 |
| feature_557 | float |  | feature 557 | 是 | 否 | 否 |
| feature_558 | float |  | feature 558 | 是 | 否 | 否 |
| feature_559 | float |  | feature 559 | 是 | 否 | 否 |
| feature_560 | float |  | feature 560 | 是 | 否 | 否 |
| feature_561 | float |  | feature 561 | 是 | 否 | 否 |
| feature_562 | float |  | feature 562 | 是 | 否 | 否 |
| feature_563 | float |  | feature 563 | 是 | 否 | 否 |
| feature_564 | float |  | feature 564 | 是 | 否 | 否 |
| feature_565 | float |  | feature 565 | 是 | 否 | 否 |
| feature_566 | float |  | feature 566 | 是 | 否 | 否 |
| feature_567 | float |  | feature 567 | 是 | 否 | 否 |
| feature_568 | float |  | feature 568 | 是 | 否 | 否 |
| feature_569 | float |  | feature 569 | 是 | 否 | 否 |
| feature_570 | float |  | feature 570 | 是 | 否 | 否 |
| feature_571 | float |  | feature 571 | 是 | 否 | 否 |
| feature_572 | float |  | feature 572 | 是 | 否 | 否 |
| feature_573 | float |  | feature 573 | 是 | 否 | 否 |
| feature_574 | float |  | feature 574 | 是 | 否 | 否 |
| feature_575 | float |  | feature 575 | 是 | 否 | 否 |
| feature_576 | float |  | feature 576 | 是 | 否 | 否 |
| feature_577 | float |  | feature 577 | 是 | 否 | 否 |
| feature_578 | float |  | feature 578 | 是 | 否 | 否 |
| feature_579 | float |  | feature 579 | 是 | 否 | 否 |
| feature_580 | float |  | feature 580 | 是 | 否 | 否 |
| feature_581 | float |  | feature 581 | 是 | 否 | 否 |
| feature_582 | float |  | feature 582 | 是 | 否 | 否 |
| feature_583 | float |  | feature 583 | 是 | 否 | 否 |
| feature_584 | float |  | feature 584 | 是 | 否 | 否 |
| feature_585 | float |  | feature 585 | 是 | 否 | 否 |
| feature_586 | float |  | feature 586 | 是 | 否 | 否 |
| feature_587 | float |  | feature 587 | 是 | 否 | 否 |
| feature_588 | float |  | feature 588 | 是 | 否 | 否 |
| feature_589 | float |  | feature 589 | 是 | 否 | 否 |
| feature_590 | float |  | feature 590 | 是 | 否 | 否 |
| feature_591 | float |  | feature 591 | 是 | 否 | 否 |
| feature_592 | float |  | feature 592 | 是 | 否 | 否 |
| feature_593 | float |  | feature 593 | 是 | 否 | 否 |
| feature_594 | float |  | feature 594 | 是 | 否 | 否 |
| feature_595 | float |  | feature 595 | 是 | 否 | 否 |
| feature_596 | float |  | feature 596 | 是 | 否 | 否 |
| feature_597 | float |  | feature 597 | 是 | 否 | 否 |
| feature_598 | float |  | feature 598 | 是 | 否 | 否 |
| feature_599 | float |  | feature 599 | 是 | 否 | 否 |
| feature_600 | float |  | feature 600 | 是 | 否 | 否 |
| feature_601 | float |  | feature 601 | 是 | 否 | 否 |
| feature_602 | float |  | feature 602 | 是 | 否 | 否 |
| feature_603 | float |  | feature 603 | 是 | 否 | 否 |
| feature_604 | float |  | feature 604 | 是 | 否 | 否 |
| feature_605 | float |  | feature 605 | 是 | 否 | 否 |
| feature_606 | float |  | feature 606 | 是 | 否 | 否 |
| feature_607 | float |  | feature 607 | 是 | 否 | 否 |
| feature_608 | float |  | feature 608 | 是 | 否 | 否 |
| feature_609 | float |  | feature 609 | 是 | 否 | 否 |
| feature_610 | float |  | feature 610 | 是 | 否 | 否 |
| feature_611 | float |  | feature 611 | 是 | 否 | 否 |
| feature_612 | float |  | feature 612 | 是 | 否 | 否 |
| feature_613 | float |  | feature 613 | 是 | 否 | 否 |
| feature_614 | float |  | feature 614 | 是 | 否 | 否 |
| feature_615 | float |  | feature 615 | 是 | 否 | 否 |
| feature_616 | float |  | feature 616 | 是 | 否 | 否 |
| feature_617 | float |  | feature 617 | 是 | 否 | 否 |
| feature_618 | float |  | feature 618 | 是 | 否 | 否 |
| feature_619 | float |  | feature 619 | 是 | 否 | 否 |
| feature_620 | float |  | feature 620 | 是 | 否 | 否 |
| feature_621 | float |  | feature 621 | 是 | 否 | 否 |
| feature_622 | float |  | feature 622 | 是 | 否 | 否 |
| feature_623 | float |  | feature 623 | 是 | 否 | 否 |
| feature_624 | float |  | feature 624 | 是 | 否 | 否 |
| feature_625 | float |  | feature 625 | 是 | 否 | 否 |
| feature_626 | float |  | feature 626 | 是 | 否 | 否 |
| feature_627 | float |  | feature 627 | 是 | 否 | 否 |
| feature_628 | float |  | feature 628 | 是 | 否 | 否 |
| feature_629 | float |  | feature 629 | 是 | 否 | 否 |
| feature_630 | float |  | feature 630 | 是 | 否 | 否 |
| feature_631 | float |  | feature 631 | 是 | 否 | 否 |
| feature_632 | float |  | feature 632 | 是 | 否 | 否 |
| feature_633 | float |  | feature 633 | 是 | 否 | 否 |
| feature_634 | float |  | feature 634 | 是 | 否 | 否 |
| feature_635 | float |  | feature 635 | 是 | 否 | 否 |
| feature_636 | float |  | feature 636 | 是 | 否 | 否 |
| feature_637 | float |  | feature 637 | 是 | 否 | 否 |
| feature_638 | float |  | feature 638 | 是 | 否 | 否 |
| feature_639 | float |  | feature 639 | 是 | 否 | 否 |
| feature_640 | float |  | feature 640 | 是 | 否 | 否 |
| feature_641 | float |  | feature 641 | 是 | 否 | 否 |
| feature_642 | float |  | feature 642 | 是 | 否 | 否 |
| feature_643 | float |  | feature 643 | 是 | 否 | 否 |
| feature_644 | float |  | feature 644 | 是 | 否 | 否 |
| feature_645 | float |  | feature 645 | 是 | 否 | 否 |
| feature_646 | float |  | feature 646 | 是 | 否 | 否 |
| feature_647 | float |  | feature 647 | 是 | 否 | 否 |
| feature_648 | float |  | feature 648 | 是 | 否 | 否 |
| feature_649 | float |  | feature 649 | 是 | 否 | 否 |
| feature_650 | float |  | feature 650 | 是 | 否 | 否 |
| feature_651 | float |  | feature 651 | 是 | 否 | 否 |
| feature_652 | float |  | feature 652 | 是 | 否 | 否 |
| feature_653 | float |  | feature 653 | 是 | 否 | 否 |
| feature_654 | float |  | feature 654 | 是 | 否 | 否 |
| feature_655 | float |  | feature 655 | 是 | 否 | 否 |
| feature_656 | float |  | feature 656 | 是 | 否 | 否 |
| feature_657 | float |  | feature 657 | 是 | 否 | 否 |
| feature_658 | float |  | feature 658 | 是 | 否 | 否 |
| feature_659 | float |  | feature 659 | 是 | 否 | 否 |
| feature_660 | float |  | feature 660 | 是 | 否 | 否 |
| feature_661 | float |  | feature 661 | 是 | 否 | 否 |
| feature_662 | float |  | feature 662 | 是 | 否 | 否 |
| feature_663 | float |  | feature 663 | 是 | 否 | 否 |
| feature_664 | float |  | feature 664 | 是 | 否 | 否 |
| feature_665 | float |  | feature 665 | 是 | 否 | 否 |
| feature_666 | float |  | feature 666 | 是 | 否 | 否 |
| feature_667 | float |  | feature 667 | 是 | 否 | 否 |
| feature_668 | float |  | feature 668 | 是 | 否 | 否 |
| feature_669 | float |  | feature 669 | 是 | 否 | 否 |
| feature_670 | float |  | feature 670 | 是 | 否 | 否 |
| feature_671 | float |  | feature 671 | 是 | 否 | 否 |
| feature_672 | float |  | feature 672 | 是 | 否 | 否 |
| feature_673 | float |  | feature 673 | 是 | 否 | 否 |
| feature_674 | float |  | feature 674 | 是 | 否 | 否 |
| feature_675 | float |  | feature 675 | 是 | 否 | 否 |
| feature_676 | float |  | feature 676 | 是 | 否 | 否 |
| feature_677 | float |  | feature 677 | 是 | 否 | 否 |
| feature_678 | float |  | feature 678 | 是 | 否 | 否 |
| feature_679 | float |  | feature 679 | 是 | 否 | 否 |
| feature_680 | float |  | feature 680 | 是 | 否 | 否 |
| feature_681 | float |  | feature 681 | 是 | 否 | 否 |
| feature_682 | float |  | feature 682 | 是 | 否 | 否 |
| feature_683 | float |  | feature 683 | 是 | 否 | 否 |
| feature_684 | float |  | feature 684 | 是 | 否 | 否 |
| feature_685 | float |  | feature 685 | 是 | 否 | 否 |
| feature_686 | float |  | feature 686 | 是 | 否 | 否 |
| feature_687 | float |  | feature 687 | 是 | 否 | 否 |
| feature_688 | float |  | feature 688 | 是 | 否 | 否 |
| feature_689 | float |  | feature 689 | 是 | 否 | 否 |
| feature_690 | float |  | feature 690 | 是 | 否 | 否 |
| feature_691 | float |  | feature 691 | 是 | 否 | 否 |
| feature_692 | float |  | feature 692 | 是 | 否 | 否 |
| feature_693 | float |  | feature 693 | 是 | 否 | 否 |
| feature_694 | float |  | feature 694 | 是 | 否 | 否 |
| feature_695 | float |  | feature 695 | 是 | 否 | 否 |
| feature_696 | float |  | feature 696 | 是 | 否 | 否 |
| feature_697 | float |  | feature 697 | 是 | 否 | 否 |
| feature_698 | float |  | feature 698 | 是 | 否 | 否 |
| feature_699 | float |  | feature 699 | 是 | 否 | 否 |
| feature_700 | float |  | feature 700 | 是 | 否 | 否 |
| feature_701 | float |  | feature 701 | 是 | 否 | 否 |
| feature_702 | float |  | feature 702 | 是 | 否 | 否 |
| feature_703 | float |  | feature 703 | 是 | 否 | 否 |
| feature_704 | float |  | feature 704 | 是 | 否 | 否 |
| feature_705 | float |  | feature 705 | 是 | 否 | 否 |
| feature_706 | float |  | feature 706 | 是 | 否 | 否 |
| feature_707 | float |  | feature 707 | 是 | 否 | 否 |
| feature_708 | float |  | feature 708 | 是 | 否 | 否 |
| feature_709 | float |  | feature 709 | 是 | 否 | 否 |
| feature_710 | float |  | feature 710 | 是 | 否 | 否 |
| feature_711 | float |  | feature 711 | 是 | 否 | 否 |
| feature_712 | float |  | feature 712 | 是 | 否 | 否 |
| feature_713 | float |  | feature 713 | 是 | 否 | 否 |
| feature_714 | float |  | feature 714 | 是 | 否 | 否 |
| feature_715 | float |  | feature 715 | 是 | 否 | 否 |
| feature_716 | float |  | feature 716 | 是 | 否 | 否 |
| feature_717 | float |  | feature 717 | 是 | 否 | 否 |
| feature_718 | float |  | feature 718 | 是 | 否 | 否 |
| feature_719 | float |  | feature 719 | 是 | 否 | 否 |
| feature_720 | float |  | feature 720 | 是 | 否 | 否 |
| feature_721 | float |  | feature 721 | 是 | 否 | 否 |
| feature_722 | float |  | feature 722 | 是 | 否 | 否 |
| feature_723 | float |  | feature 723 | 是 | 否 | 否 |
| feature_724 | float |  | feature 724 | 是 | 否 | 否 |
| feature_725 | float |  | feature 725 | 是 | 否 | 否 |
| feature_726 | float |  | feature 726 | 是 | 否 | 否 |
| feature_727 | float |  | feature 727 | 是 | 否 | 否 |
| feature_728 | float |  | feature 728 | 是 | 否 | 否 |
| feature_729 | float |  | feature 729 | 是 | 否 | 否 |
| feature_730 | float |  | feature 730 | 是 | 否 | 否 |
| feature_731 | float |  | feature 731 | 是 | 否 | 否 |
| feature_732 | float |  | feature 732 | 是 | 否 | 否 |
| feature_733 | float |  | feature 733 | 是 | 否 | 否 |
| feature_734 | float |  | feature 734 | 是 | 否 | 否 |
| feature_735 | float |  | feature 735 | 是 | 否 | 否 |
| feature_736 | float |  | feature 736 | 是 | 否 | 否 |
| feature_737 | float |  | feature 737 | 是 | 否 | 否 |
| feature_738 | float |  | feature 738 | 是 | 否 | 否 |
| feature_739 | float |  | feature 739 | 是 | 否 | 否 |
| feature_740 | float |  | feature 740 | 是 | 否 | 否 |
| feature_741 | float |  | feature 741 | 是 | 否 | 否 |
| feature_742 | float |  | feature 742 | 是 | 否 | 否 |
| feature_743 | float |  | feature 743 | 是 | 否 | 否 |
| feature_744 | float |  | feature 744 | 是 | 否 | 否 |
| feature_745 | float |  | feature 745 | 是 | 否 | 否 |
| feature_746 | float |  | feature 746 | 是 | 否 | 否 |
| feature_747 | float |  | feature 747 | 是 | 否 | 否 |
| feature_748 | float |  | feature 748 | 是 | 否 | 否 |
| feature_749 | float |  | feature 749 | 是 | 否 | 否 |
| feature_750 | float |  | feature 750 | 是 | 否 | 否 |
| feature_751 | float |  | feature 751 | 是 | 否 | 否 |
| feature_752 | float |  | feature 752 | 是 | 否 | 否 |
| feature_753 | float |  | feature 753 | 是 | 否 | 否 |
| feature_754 | float |  | feature 754 | 是 | 否 | 否 |
| feature_755 | float |  | feature 755 | 是 | 否 | 否 |
| feature_756 | float |  | feature 756 | 是 | 否 | 否 |
| feature_757 | float |  | feature 757 | 是 | 否 | 否 |
| feature_758 | float |  | feature 758 | 是 | 否 | 否 |
| feature_759 | float |  | feature 759 | 是 | 否 | 否 |
| feature_760 | float |  | feature 760 | 是 | 否 | 否 |
| feature_761 | float |  | feature 761 | 是 | 否 | 否 |
| feature_762 | float |  | feature 762 | 是 | 否 | 否 |
| feature_763 | float |  | feature 763 | 是 | 否 | 否 |
| feature_764 | float |  | feature 764 | 是 | 否 | 否 |
| feature_765 | float |  | feature 765 | 是 | 否 | 否 |
| feature_766 | float |  | feature 766 | 是 | 否 | 否 |
| feature_767 | float |  | feature 767 | 是 | 否 | 否 |
| feature_768 | float |  | feature 768 | 是 | 否 | 否 |

## point_static_metadata_8975.csv

- 来源：`data\training_data\static\point_static_metadata_8975.csv`
- 行数：8975

| 字段 | 类型 | 单位 | 含义 | 模型输入 | 预测目标 | 仅QC/验证 |
|---|---|---|---|---:|---:|---:|
| point_id | integer |  | point id | 否 | 否 | 否 |
| x | float | degree | x | 否 | 否 | 否 |
| y | float | degree | y | 否 | 否 | 否 |
| year | integer |  | year | 否 | 否 | 否 |
| month | integer |  | month | 否 | 否 | 否 |
| dino_window_row | integer |  | dino window row | 否 | 否 | 否 |

## spatial_split_assignments.csv

- 来源：`data\training_data\splits\spatial_split_assignments.csv`
- 行数：8975

| 字段 | 类型 | 单位 | 含义 | 模型输入 | 预测目标 | 仅QC/验证 |
|---|---|---|---|---:|---:|---:|
| point_id | integer |  | point id | 否 | 否 | 否 |
| spatial_block_id | string |  | spatial block id | 否 | 否 | 否 |
| overlap_component_id | string |  | overlap component id | 否 | 否 | 否 |
| dataset_split | string |  | dataset split | 否 | 否 | 否 |
| split_seed | integer |  | split seed | 否 | 否 | 否 |

## training_manifest.csv

- 来源：`data\training_data\manifest\training_manifest.csv`
- 行数：116675

| 字段 | 类型 | 单位 | 含义 | 模型输入 | 预测目标 | 仅QC/验证 |
|---|---|---|---|---:|---:|---:|
| sample_id | integer |  | sample id | 否 | 否 | 否 |
| point_id | integer |  | point id | 否 | 否 | 否 |
| date | string |  | date | 否 | 否 | 否 |
| hour | integer |  | hour | 否 | 否 | 否 |
| static_feature_row | integer |  | static feature row | 否 | 否 | 否 |
| dino_mean_row | integer |  | dino mean row | 否 | 否 | 否 |
| dino_window_row | integer |  | dino window row | 否 | 否 | 否 |
| dynamic_condition_row | integer |  | dynamic condition row | 否 | 否 | 否 |
| label_row | integer |  | label row | 否 | 否 | 否 |
| spatial_block_id | string |  | spatial block id | 否 | 否 | 否 |
| overlap_component_id | string |  | overlap component id | 否 | 否 | 否 |
| dataset_split | string |  | dataset split | 否 | 否 | 否 |

## dinov2_window_features.npy

- 来源：`artifacts\intermediate\pipeline_records\steps\step5b_dinov2_windows\dinov2_window_features.npy`
- shape：`[8975, 8, 768]`
- dtype：`float32`
- 维度含义：点位 × 8个方位窗口 × 768维DINOv2 embedding。
- 通过`point_static_metadata_8975.csv`中的`dino_window_row`关联。
- 模型输入：是；预测目标：否。
