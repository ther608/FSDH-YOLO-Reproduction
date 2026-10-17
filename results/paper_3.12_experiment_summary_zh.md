# Paper 3.12 experiment ledger (original Chinese text)

> This is the original Chinese data ledger used for the coverage audit. It is retained for provenance; the English audit is in `docs/01_coverage_audit.md`.
> Absolute workspace paths were replaced with placeholders.

---

# 论文3.12 第三章实验结果数据汇总（均值±标准差 / 逐种子）

> 本文档为**数据汇总**，不构成正文修改。所有数值均自源文件重新读取并复算（样本标准差 ddof=1），未手工转录。
> 结构：正文 8 个实验 + 附录 B 4 个实验；其中 B.4 只给汇总，其余 11 个均给「均值±SD」与「逐种子」两种形式。

## 0. 计算协议（全部实验共用，务必先读）

| 项 | 设置 |
|---|---|
| 精度评价器 | 统一 **Common Evaluator**（与训练框架解耦；**不是** Ultralytics 原生 val/test 指标） |
| 输入尺寸 | 640×640 方形推理 |
| 预测置信度下限 | 0.001（保留构建完整 PR 曲线所需的低置信度预测） |
| NMS | 类别感知，IoU=0.70；RT-DETR 等无 NMS 模型保留原生 decoder/top-k，不额外施加 |
| 每图最大预测框 | 300 |
| AP 协议 | COCO 101 点插值；IoU 自 0.50 至 0.95、步长 0.05 |
| P / R 工作点 | 类别平均 F1 最大时的置信度点；类别感知匹配 IoU≥0.50；宏平均 |
| 误差口径 | 三个随机种子（seed=0,1,2）的算术平均值 ± **样本标准差（ddof=1）** |
| 训练 | 每配置独立训练 3 个 seed；best.pt 按 Ultralytics 默认 fitness（验证集 mAP50-95）选定 |
| 测试集 | 独立测试集，不参与训练、checkpoint 选择与超参优化 |
| **FPS** | **另行测速，非训练产出**，与精度评价不同批。CUDA FP32、batch=1、640×640、预热 100 次、计时 500 次、每次前后 CUDA 同步；计时范围＝前向+预测解码+类别感知 NMS（**不含**图像读取与输入预处理）；FPS = 1000 / 平均单图总时延；三个 seed 的模型同协议独立测速后取 mean±SD。文中 NMS 统一用 conf=0.25、IoU=0.70（仅用于统一推理负载，与精度评价阶段的 conf 下限 0.001 不同） |
| Params | 推理态**融合后**参数量（Ultralytics `summary (fused)` 口径） |

> ⚠️ **不可混引**：同一 run 的 Ultralytics 原生指标与本文口径不同。
> 例：`runs/detect/paper_groupaware_clean_v2_reruns/dsA_groupaware_clean_aug_v2__ablation__shab__seed0/final_summary.txt`
> 原生 val 为 mAP50=0.845 / mAP50-95=0.669，而论文表内为 0.811 / 0.662（Common Evaluator）。

> 命名说明：论文正文用 **SHEB**，代码、配置与 checkpoint 中仍为 **SHAB**，二者同一模块。下表统一标注 SHEB。

---

# 一、正文实验（8 个）

## 1.1 多尺度特征增强模块对比实验（3.3 / 表 3）

**三随机种子均值 ± 样本标准差**

| Model | P/% | R/% | mAP50/% | mAP50-95/% |
|---|---|---|---|---|
| YOLO11n | 78.007±1.588 | 78.174±2.110 | 78.984±0.959 | 64.429±0.603 |
| ASPP | 77.293±1.624 | 78.412±1.554 | 80.538±1.545 | 65.501±1.010 |
| SPP | 77.567±0.687 | 79.809±1.065 | 81.255±1.202 | 66.419±0.984 |
| DilatedConv | 76.359±0.187 | 80.754±0.819 | 81.155±0.338 | 66.214±0.283 |
| RepConv | 77.033±1.232 | 79.455±1.479 | 80.402±0.704 | 65.646±0.278 |
| MultiScaleDWConv | 77.451±0.902 | 79.800±0.468 | 81.022±0.682 | 66.357±0.328 |
| RFB | 77.072±0.618 | 79.493±1.751 | 79.770±2.469 | 65.631±1.540 |
| SHEB | 78.342±0.196 | 78.854±1.244 | 81.092±2.012 | 66.215±1.142 |

**FPS + Params（三 seed 均值 ± 样本标准差；forward+NMS）**

| Model | Params/M | FPS | 逐 seed (s0/s1/s2) | 来源文件 |
|---|---|---|---|---|
| YOLO11n | 2.583 | 146.69±4.87 | 150.57 148.28 141.23 | `runs/quality/paper16_non_fsd_uniform_latency/uniform_latency_results.csv` |
| ASPP | 3.419 | 123.68±2.20 | 122.46 122.35 126.22 | `runs/quality/shab_comparison_uniform_latency_fp32_b1_640/uniform_latency_results.csv` |
| SPP | 2.829 | 136.13±3.85 | 135.93 132.38 140.07 | `runs/quality/shab_comparison_uniform_latency_fp32_b1_640/uniform_latency_results.csv` |
| DilatedConv | 2.688 | 138.53±1.18 | 137.68 139.89 138.04 | `runs/quality/shab_comparison_uniform_latency_fp32_b1_640/uniform_latency_results.csv` |
| RepConv | 3.468 | 143.79±4.31 | 143.03 148.44 139.92 | `runs/quality/shab_comparison_uniform_latency_fp32_b1_640/uniform_latency_results.csv` |
| MultiScaleDWConv | 2.742 | 132.33±1.55 | 130.56 132.93 133.48 | `runs/quality/shab_comparison_uniform_latency_fp32_b1_640/uniform_latency_results.csv` |
| RFB | 2.947 | 123.97±0.35 | 124.21 124.12 123.57 | `runs/quality/shab_comparison_uniform_latency_fp32_b1_640/uniform_latency_results.csv` |
| SHEB | 2.996 | 122.01±1.17 | 122.57 122.80 120.67 | `runs/quality/paper16_non_fsd_uniform_latency/uniform_latency_results.csv` |

**逐种子结果**

| Model | seed | P/% | R/% | mAP50/% | mAP50-95/% | FPS |
|---|---|---|---|---|---|---|
| YOLO11n | 0 | 77.117 | 80.436 | 80.068 | 65.120 | 150.57 |
| YOLO11n | 1 | 79.841 | 76.257 | 78.246 | 64.151 | 148.28 |
| YOLO11n | 2 | 77.063 | 77.829 | 78.637 | 64.016 | 141.23 |
| ASPP | 0 | 77.103 | 79.754 | 80.793 | 65.918 | 122.46 |
| ASPP | 1 | 75.772 | 76.710 | 78.881 | 64.349 | 122.35 |
| ASPP | 2 | 79.004 | 78.772 | 81.939 | 66.235 | 126.22 |
| SPP | 0 | 78.360 | 79.066 | 80.615 | 66.164 | 135.93 |
| SPP | 1 | 77.198 | 79.333 | 80.508 | 65.587 | 132.38 |
| SPP | 2 | 77.144 | 81.029 | 82.641 | 67.505 | 140.07 |
| DilatedConv | 0 | 76.499 | 80.966 | 81.543 | 66.357 | 137.68 |
| DilatedConv | 1 | 76.432 | 79.850 | 80.927 | 65.889 | 139.89 |
| DilatedConv | 2 | 76.146 | 81.447 | 80.995 | 66.397 | 138.04 |
| RepConv | 0 | 77.828 | 79.842 | 81.065 | 65.703 | 143.03 |
| RepConv | 1 | 75.614 | 80.702 | 80.478 | 65.892 | 148.44 |
| RepConv | 2 | 77.658 | 77.821 | 79.662 | 65.344 | 139.92 |
| MultiScaleDWConv | 0 | 78.327 | 79.506 | 80.934 | 66.452 | 130.56 |
| MultiScaleDWConv | 1 | 77.500 | 80.340 | 81.745 | 66.628 | 132.93 |
| MultiScaleDWConv | 2 | 76.525 | 79.555 | 80.389 | 65.992 | 133.48 |
| RFB | 0 | 77.499 | 81.230 | 82.507 | 67.348 | 124.21 |
| RFB | 1 | 76.363 | 79.522 | 79.093 | 65.174 | 124.12 |
| RFB | 2 | 77.355 | 77.728 | 77.711 | 64.372 | 123.57 |
| SHEB | 0 | 78.173 | 79.970 | 83.031 | 67.388 | 122.57 |
| SHEB | 1 | 78.297 | 79.081 | 79.013 | 65.106 | 122.80 |
| SHEB | 2 | 78.556 | 77.513 | 81.233 | 66.152 | 120.67 |

<details><summary>数据来源路径</summary>

| 内容 | 路径 |
|---|---|
| 逐 seed 精度（7 模块） | `analysis/shab_comparison_common_protocol/summary.json → per_seed_accuracy_provenance[]` |
| 均值±SD（7 模块） | `analysis/shab_comparison_common_protocol/summary.json → results[]` |
| 可读版汇总 | `analysis/shab_comparison_common_protocol/seven_modules_per_seed_and_mean_common_evaluator.md` |
| 逐 seed 精度（YOLO11n / SHEB） | `analysis/paper16_unified_common_protocol/per_seed.csv（key: yolo11n, shab）` |
| Common Evaluator 指标 JSON（7 模块） | `runs/common_eval/paper16_unified_protocol/shab_comparison_{aspp,spp,dilatedconv,repconv,multiscale_dwconv,rfb}/seed{0,1,2}__test_metrics_common.json；SHEB 在 .../shab/` |
| Common Evaluator 指标 JSON（YOLO11n） | `runs/common_eval/paper16_unified_protocol/yolo11n/seed{0,1,2}__test_metrics_common.json` |
| FPS（ASPP/SPP/DilatedConv/RepConv/MultiScaleDWConv/RFB） | `runs/quality/shab_comparison_uniform_latency_fp32_b1_640/uniform_latency_results.csv（+ uniform_latency_three_seed_mean_sd.csv）` |
| FPS（YOLO11n、SHEB） | `runs/quality/paper16_non_fsd_uniform_latency/uniform_latency_results.csv（+ uniform_latency_three_seed_mean_sd.csv）` |
| Params | `上述 latency CSV 的 params_fused_M / params_M 列（= fused 计数；SHEB = 2.995793）` |

</details>

> **FPS 口径提示（已按统一口径取值）**：本表 FPS 为独立于精度评价的另行测速。
> SHEB 统一取 `runs/quality/paper16_non_fsd_uniform_latency`（该轮行名为 SHAB）的 **122.01±1.17**，逐 seed 122.57 / 122.80 / 120.67，
> 与正文表 4、附录 B.3 的 SHEB 完全一致。
> 另注：若把 SHEB 换用本表其余 6 个模块所在的 `..._fp32_b1_640` 批次，其值为 `122.48±1.80`（逐 seed 124.29 / 122.46 / 120.69）；
> 两轮为同机不同时间测速，相差 0.47 FPS。本表**不采用**该值，以保证全篇 SHEB 只有一个 FPS。

## 1.2 消融实验（3.4 / 表 4）

**三随机种子均值 ± 样本标准差**

| Model | P/% | R/% | mAP50/% | mAP50-95/% |
|---|---|---|---|---|
| YOLO11n | 78.007±1.588 | 78.174±2.110 | 78.984±0.959 | 64.429±0.603 |
| YOLO11n-PM | 79.323±1.656 | 78.289±0.729 | 80.940±1.793 | 66.872±1.140 |
| YOLO11n +FSDConv | 76.927±1.346 | 79.188±3.066 | 80.563±1.522 | 66.292±0.971 |
| YOLO11n +SHEB | 78.342±0.196 | 78.854±1.244 | 81.092±2.012 | 66.215±1.142 |
| FSDH-YOLO | 79.989±2.168 | 78.443±1.127 | 81.766±0.994 | 67.297±0.718 |

**FPS + Params（三 seed 均值 ± 样本标准差；forward+NMS）**

| Model | Params/M | FPS | 逐 seed (s0/s1/s2) | 来源文件 |
|---|---|---|---|---|
| YOLO11n | 2.583 | 146.69±4.87 | 150.57 148.28 141.23 | `runs/quality/paper16_non_fsd_uniform_latency/uniform_latency_results.csv` |
| YOLO11n-PM | 3.014 | 161.18±3.15 | 164.69 158.62 160.22 | `runs/latency/paper16_shab_branch_subset_and_param_match_v1/dsA_groupaware_clean_aug_v2__capacity_control__yolo11n_pm__seed{0,1,2}/uniform_latency_results.json` |
| YOLO11n +FSDConv | 2.584 | 148.99±0.45 | 148.49 149.34 149.14 | `runs/quality/paper16_spo_fsd_uniform_latency/uniform_latency_results.csv` |
| YOLO11n +SHEB | 2.996 | 122.01±1.17 | 122.57 122.80 120.67 | `runs/quality/paper16_non_fsd_uniform_latency/uniform_latency_results.csv` |
| FSDH-YOLO | 2.997 | 123.39±2.94 | 124.34 120.09 125.74 | `runs/quality/paper16_spo_fsd_uniform_latency/uniform_latency_results.csv` |

**逐种子结果**

| Model | seed | P/% | R/% | mAP50/% | mAP50-95/% | FPS |
|---|---|---|---|---|---|---|
| YOLO11n | 0 | 77.117 | 80.436 | 80.068 | 65.120 | 150.57 |
| YOLO11n | 1 | 79.841 | 76.257 | 78.246 | 64.151 | 148.28 |
| YOLO11n | 2 | 77.063 | 77.829 | 78.637 | 64.016 | 141.23 |
| YOLO11n-PM | 0 | 79.060 | 78.937 | 80.820 | 66.931 | 164.69 |
| YOLO11n-PM | 1 | 77.814 | 78.430 | 79.209 | 65.703 | 158.62 |
| YOLO11n-PM | 2 | 81.095 | 77.499 | 82.790 | 67.981 | 160.22 |
| YOLO11n +FSDConv | 0 | 75.375 | 82.672 | 81.487 | 66.541 | 148.49 |
| YOLO11n +FSDConv | 1 | 77.780 | 77.989 | 81.395 | 67.115 | 149.34 |
| YOLO11n +FSDConv | 2 | 77.626 | 76.902 | 78.806 | 65.221 | 149.14 |
| YOLO11n +SHEB | 0 | 78.173 | 79.970 | 83.031 | 67.388 | 122.57 |
| YOLO11n +SHEB | 1 | 78.297 | 79.081 | 79.013 | 65.106 | 122.80 |
| YOLO11n +SHEB | 2 | 78.556 | 77.513 | 81.233 | 66.152 | 120.67 |
| FSDH-YOLO | 0 | 78.581 | 78.364 | 80.931 | 66.670 | 124.34 |
| FSDH-YOLO | 1 | 78.900 | 79.608 | 81.500 | 67.140 | 120.09 |
| FSDH-YOLO | 2 | 82.485 | 77.358 | 82.866 | 68.080 | 125.74 |

> YOLO11n-PM 为容量匹配对照：宽度系数 0.25→0.27、最大通道 1024→992，深度系数 0.50 与模块层数不变；
> 参数量 3.014 M，与 FSDH-YOLO 相差 0.017 M。

<details><summary>数据来源路径</summary>

| 内容 | 路径 |
|---|---|
| 逐 seed 精度 | `analysis/paper16_unified_common_protocol/per_seed.csv（key: yolo11n, fsdconv, shab, fsdconv_shab）` |
| YOLO11n-PM 逐 seed 精度 | `runs/common_eval/paper16_shab_branch_subset_and_param_match_v1/YOLO11N_PM/seed{0,1,2}__test_metrics_common.json` |
| Common Evaluator 指标 JSON | `runs/common_eval/paper16_unified_protocol/{fsdconv,shab,fsdconv_shab}/；YOLO11n 在 .../yolo11n/` |
| FPS（YOLO11n、+SHEB） | `runs/quality/paper16_non_fsd_uniform_latency/uniform_latency_results.csv` |
| FPS（+FSDConv、FSDH-YOLO） | `runs/quality/paper16_spo_fsd_uniform_latency/uniform_latency_results.csv` |
| FPS（YOLO11n-PM） | `runs/latency/paper16_shab_branch_subset_and_param_match_v1/dsA_groupaware_clean_aug_v2__capacity_control__yolo11n_pm__seed{0,1,2}/uniform_latency_results.json` |

</details>


## 1.3 小目标检测性能分析实验（3.5.1 / 表 5）

面积口径：`AP_small(640)` = 原图面积 ∈ [0, 4096) px² —— 由 COCO 阈值 32²=1024 px² 按「原图约 1280 → 输入 640」的 2:1 线性缩放、4:1 面积缩放换算得到；`AP_non-small` = 原图面积 ∈ [4096, 1e10) px²。两项均按 COCOeval 的 area-range **ignore** 语义计算（评价某区间时，区间外 GT 标记为 ignore，而非从评价数据中删除）。指标口径为 AP50-95。

**三随机种子均值 ± 样本标准差**

| Target | GT small | YOLO11n AP_small(640)/% | FSDH-YOLO AP_small(640)/% | Δ/pp | GT non-small | YOLO11n AP_non-small/% | FSDH-YOLO AP_non-small/% | Δ/pp |
|---|---|---|---|---|---|---|---|---|
| All | 889 | 55.204±0.616 | 58.065±1.405 | +2.861 | 1886 | 68.443±0.736 | 71.375±0.666 | +2.931 |
| unripe | 457 | 62.011±0.623 | 62.702±0.446 | +0.691 | 500 | 77.171±1.029 | 79.174±0.312 | +2.002 |
| semi-ripe | 145 | 41.777±2.455 | 46.437±2.666 | +4.660 | 508 | 59.896±2.832 | 61.511±0.847 | +1.615 |
| ripe | 287 | 61.824±0.773 | 65.057±1.308 | +3.233 | 878 | 68.263±1.271 | 73.439±1.893 | +5.176 |

**逐种子结果（AP50-95）**

| Target | 面积区间 | GT 数 | YOLO11n s0/s1/s2 | FSDH-YOLO s0/s1/s2 |
|---|---|---|---|---|
| All | small(640) | 889 | 55.914 / 54.824 / 54.873 | 57.577 / 56.969 / 59.649 |
| All | non-small | 1886 | 69.293 / 68.014 / 68.022 | 70.685 / 71.423 / 72.016 |
| unripe | small(640) | 457 | 62.098 / 61.349 / 62.586 | 62.782 / 62.221 / 63.102 |
| unripe | non-small | 500 | 77.682 / 75.987 / 77.845 | 79.136 / 79.503 / 78.882 |
| semi-ripe | small(640) | 145 | 44.605 / 40.538 / 40.188 | 44.827 / 44.969 / 49.515 |
| semi-ripe | non-small | 508 | 61.972 / 61.046 / 56.669 | 60.650 / 62.342 / 61.542 |
| ripe | small(640) | 287 | 61.040 / 62.585 / 61.846 | 65.122 / 63.717 / 66.330 |
| ripe | non-small | 878 | 68.227 / 67.010 / 69.552 | 72.270 / 72.423 / 75.623 |

**两模型总体指标（P / R / mAP50 / mAP50-95 / FPS，逐种子）**

| Model | seed | P/% | R/% | mAP50/% | mAP50-95/% | FPS |
|---|---|---|---|---|---|---|
| YOLO11n | 0 | 77.117 | 80.436 | 80.068 | 65.120 | 150.57 |
| YOLO11n | 1 | 79.841 | 76.257 | 78.246 | 64.151 | 148.28 |
| YOLO11n | 2 | 77.063 | 77.829 | 78.637 | 64.016 | 141.23 |
| FSDH-YOLO | 0 | 78.581 | 78.364 | 80.931 | 66.670 | 124.34 |
| FSDH-YOLO | 1 | 78.900 | 79.608 | 81.500 | 67.140 | 120.09 |
| FSDH-YOLO | 2 | 82.485 | 77.358 | 82.866 | 68.080 | 125.74 |

<details><summary>数据来源路径</summary>

| 内容 | 路径 |
|---|---|
| 面积分层逐 seed + GT 数 + 面积定义 | `analysis/paper27_statistical_supplement/results.json → datasets.dataset_a.area_ap[] 与 protocol` |
| 均值±SD 表 | `analysis/paper27_statistical_supplement/dataset_a_area_ap.csv` |
| 复算脚本 | `scripts/paper27_bootstrap_confusion_area.py` |
| 底层预测 / 冻结标注 | `runs/common_eval/paper16_unified_protocol/{yolo11n,fsdconv_shab}/seed{0,1,2}__test_predictions_coco.json；external_configs/rtdetr_coco_annotations/instances_test.json` |

</details>

> 补充（标准 COCO 小目标，原图面积 < 1024 px²、仅 37 个 GT）：三 seed 均值 YOLO11n 38.581±5.158 / FSDH-YOLO 37.146±4.610，差值 −1.435 pp，95% CI [−6.264,+4.244] 跨 0。来源：`bootstrap_ci_extended.csv` 行 `COCO AP_s`。
> 当前测试集中不存在面积恰为 4096 px² 的 GT，故边界包含规则不影响结果。

## 1.4 定位实验（3.5.2-1 / 表 6）

指标 `AP75`：IoU=0.75 单阈值下的 AP（AP50-95 口径平均值），对边界位置偏差更敏感。

| Model | AP75/% | 相对 YOLO11n/pp | 逐 seed (s0/s1/s2) |
|---|---|---|---|
| YOLO11n | 74.441±1.029 | — | 75.559 / 73.533 / 74.231 |
| FSDH-YOLO | 77.195±0.998 | +2.754 | 76.193 / 77.205 / 78.188 |

<details><summary>数据来源路径</summary>

| 内容 | 路径 |
|---|---|
| 逐 seed AP75 | `analysis/paper_3_2_ap75_shab/AP75_逐种子.csv（key: yolo11n, fsdh；列 AP75_percent）` |
| 均值±SD | `analysis/paper_3_2_ap75_shab/AP75_三种子汇总.csv` |
| Common Evaluator 指标 JSON | `runs/common_eval/paper16_unified_protocol/{yolo11n,fsdconv_shab}/seed{0,1,2}__test_metrics_common.json（字段 overall.ap75）` |

</details>

> AP75 差值的逐图像配对 Bootstrap 95% CI = [+1.169, +4.331] pp，未跨 0。
> 注意：主流对比中 YOLOv7-tiny 为 77.756%、YOLO26n 为 77.236%，均值略高于 FSDH-YOLO 的 77.195%，故不宜表述为「对全部模型定位最优」。

## 1.5 边界框误差分析实验（3.5.2-2 / 表 7）

协议：`conf=0.25`、类别一致、按置信度降序一对一贪心匹配、最低匹配 IoU=0.50；**仅比较同一 seed 下两模型都匹配到的共同 GT**，以隔离漏检与分类差异。IoU 定位误差 = `mean(1−IoU)×100%`，**越低越好**。

| seed | 共同 GT 数 | YOLO11n/% | FSDH-YOLO/% | 误差降低/pp |
|---|---|---|---|---|
| 0 | 2166 | 9.729 | 9.314 | 0.415 |
| 1 | 2120 | 9.319 | 9.279 | 0.041 |
| 2 | 2211 | 9.684 | 9.589 | 0.094 |
| **均值±SD** | — | **9.577±0.224** | **9.394±0.170** | **0.183±0.202** |

配对图像 Bootstrap 95% CI = [-0.320, -0.049] pp（10,000 次逐图像配对重采样；三 seed 与两模型共用同一重采样索引；百分位法）。

<details><summary>数据来源路径</summary>

| 内容 | 路径 |
|---|---|
| 逐 seed + 均值±SD + Bootstrap CI + 协议 | `analysis/paper_3_4_review/paper35_extended_statistics/bootstrap_results_extended.json → metrics["IoU localization error"]` |
| 复算脚本 | `scripts/paper35_extended_statistics.py（class_aware_iou_error_per_image / common_match_errors）` |
| 底层预测 / 冻结标注 | `bootstrap_results_extended.json 顶层 inputs / annotations 字段记录的绝对路径` |

</details>

> 该指标条件化于「两模型均已检出」，须与 Miss/FN 或 AP75 并列解读，不能单独当作全部 GT 上的定位性能。

## 1.6 成熟度类别性能分析实验（3.5.3 / 表 8）

**三随机种子均值 ± 样本标准差**

| Class | YOLO11n AP50/% | FSDH-YOLO AP50/% | Δ/pp | YOLO11n AP50-95/% | FSDH-YOLO AP50-95/% | Δ/pp |
|---|---|---|---|---|---|---|
| unripe | 87.291±1.106 | 89.231±0.577 | +1.940 | 69.915±0.668 | 71.557±0.052 | +1.642 |
| semi-ripe | 68.926±3.093 | 70.174±0.919 | +1.248 | 56.482±2.641 | 58.705±0.807 | +2.223 |
| ripe | 80.734±1.364 | 85.892±1.903 | +5.158 | 66.890±1.139 | 71.629±1.616 | +4.740 |

**逐种子结果**

| Class | Model | seed | AP50/% | AP50-95/% |
|---|---|---|---|---|
| unripe | YOLO11n | 0 | 87.738 | 70.253 |
| unripe | YOLO11n | 1 | 86.032 | 69.146 |
| unripe | YOLO11n | 2 | 88.103 | 70.346 |
| unripe | FSDH-YOLO | 0 | 88.985 | 71.518 |
| unripe | FSDH-YOLO | 1 | 88.818 | 71.537 |
| unripe | FSDH-YOLO | 2 | 89.890 | 71.615 |
| semi-ripe | YOLO11n | 0 | 71.820 | 58.612 |
| semi-ripe | YOLO11n | 1 | 69.291 | 57.307 |
| semi-ripe | YOLO11n | 2 | 65.668 | 53.527 |
| semi-ripe | FSDH-YOLO | 0 | 69.118 | 57.775 |
| semi-ripe | FSDH-YOLO | 1 | 70.783 | 59.211 |
| semi-ripe | FSDH-YOLO | 2 | 70.623 | 59.131 |
| ripe | YOLO11n | 0 | 80.646 | 66.496 |
| ripe | YOLO11n | 1 | 79.416 | 66.000 |
| ripe | YOLO11n | 2 | 82.139 | 68.174 |
| ripe | FSDH-YOLO | 0 | 84.691 | 70.719 |
| ripe | FSDH-YOLO | 1 | 84.899 | 70.673 |
| ripe | FSDH-YOLO | 2 | 88.086 | 73.495 |

**类别差值（Δ = FSDH − YOLO11n）的逐图像配对 Bootstrap 95% CI**

| 指标 | Δ/pp | 95% CI/pp | 判断 |
|---|---|---|---|
| unripe AP50 | +1.940 | [+0.593, +3.339] | 支持正向提升 |
| unripe AP50-95 | +1.642 | [+0.386, +2.959] | 支持正向提升 |
| semi-ripe AP50 | +1.248 | [-1.753, +4.284] | 区间跨 0 |
| semi-ripe AP50-95 | +2.223 | [-0.355, +4.651] | 区间跨 0 |
| ripe AP50 | +5.158 | [+3.322, +7.594] | 支持正向提升 |
| ripe AP50-95 | +4.740 | [+3.264, +6.532] | 支持正向提升 |

<details><summary>数据来源路径</summary>

| 内容 | 路径 |
|---|---|
| 类别 AP 逐 seed | `runs/common_eval/paper16_unified_protocol/{yolo11n,fsdconv_shab}/seed{0,1,2}__test_metrics_common.json → per_class.<类>.{ap50,ap50_95}` |
| 均值±SD / Δ / Bootstrap CI | `analysis/paper_3_4_review/paper35_extended_statistics/bootstrap_ci_extended.csv（行 unripe/semi-ripe/ripe 的 AP50 与 AP50-95）` |
| 复算脚本 | `scripts/paper35_extended_statistics.py` |

</details>

> 固定工作点混淆矩阵属**可视化分析**（3.6），不属本实验：协议为 conf=0.25、匹配 IoU=0.50、每 seed 按真实类别行归一化后三 seed 取均值，末列为 Miss/FN，未匹配预测不入矩阵。数值见 `analysis/paper27_statistical_supplement/confusion_matrices.csv`，图件为 `confusion_dataset_a_yolo11n.png` / `confusion_dataset_a_fsdh_yolo.png`。

## 1.7 模型综合对比实验（3.7）

**三随机种子均值 ± 样本标准差**

| Model | P/% | R/% | mAP50/% | mAP50-95/% |
|---|---|---|---|---|
| YOLO11n | 78.007±1.588 | 78.174±2.110 | 78.984±0.959 | 64.429±0.603 |
| YOLOv5nu | 77.389±2.690 | 77.770±1.384 | 79.943±1.104 | 64.953±1.170 |
| YOLOv7-tiny | 77.260±1.808 | 78.866±2.704 | 84.259±0.154 | 66.341±0.061 |
| YOLOv8n | 78.121±1.052 | 78.997±0.507 | 79.699±1.193 | 65.083±0.519 |
| YOLO26n | 77.584±0.330 | 77.132±1.061 | 82.247±0.921 | 66.938±0.632 |
| RT-DETRv2-S | 74.626±1.406 | 77.007±2.800 | 81.443±0.588 | 65.984±1.003 |
| PEANet-YOLOv5-D | 77.535±1.458 | 79.522±1.544 | 80.596±1.239 | 66.120±0.626 |
| YOLOv8-FPDW | 78.810±1.021 | 78.194±0.138 | 80.127±1.101 | 65.900±1.042 |
| Edge-YOLOv11 | 79.157±1.545 | 77.902±1.156 | 79.807±2.130 | 65.728±1.608 |
| GPC-YOLO | 79.060±0.855 | 78.330±1.412 | 81.061±2.225 | 66.459±1.484 |
| FSDH-YOLO | 79.989±2.168 | 78.443±1.127 | 81.766±0.994 | 67.297±0.718 |

**FPS + Params（三 seed 均值 ± 样本标准差；forward+NMS）**

| Model | Params/M | FPS | 逐 seed (s0/s1/s2) | 来源文件 |
|---|---|---|---|---|
| YOLO11n | 2.583 | 146.69±4.87 | 150.57 148.28 141.23 | `runs/quality/paper16_non_fsd_uniform_latency/uniform_latency_results.csv` |
| YOLOv5nu | 2.504 | 166.74±4.88 | 161.33 168.08 170.82 | `runs/quality/paper16_non_fsd_uniform_latency/uniform_latency_results.csv` |
| YOLOv7-tiny | 6.013 | 189.48±0.82 | 188.92 190.42 189.09 | `None` |
| YOLOv8n | 3.006 | 183.43±4.69 | 178.05 186.64 185.60 | `runs/quality/paper16_non_fsd_uniform_latency/uniform_latency_results.csv` |
| YOLO26n | 2.375 | 136.66±1.39 | 136.45 138.15 135.39 | `runs/quality/paper16_non_fsd_uniform_latency/uniform_latency_results.csv` |
| RT-DETRv2-S | 21.857 | 68.26±11.12 | 60.63 63.12 81.02 | `None` |
| PEANet-YOLOv5-D | 9.156 | 139.99±2.08 | 137.78 140.29 141.90 | `runs/quality/paper16_pending_uniform_latency/dsA_groupaware_clean_aug_v2__literature_repro__peanet_yolov5d__seed{0,1,2}/uniform_latency_results.json` |
| YOLOv8-FPDW | 2.952 | 149.21±6.20 | 142.38 154.47 150.79 | `runs/quality/paper16_pending_uniform_latency/dsA_groupaware_clean_aug_v2__literature_repro__yolov8_fpdw_wioufix_v2__seed{0,1,2}/uniform_latency_results.json` |
| Edge-YOLOv11 | 6.336 | 67.40±1.03 | 67.16 68.53 66.51 | `runs/quality/paper16_pending_uniform_latency/dsA_groupaware_clean_aug_v2__literature_repro__edge_yolov11s__seed{0,1,2}/uniform_latency_results.json` |
| GPC-YOLO | 1.204 | 173.38±9.56 | 173.08 163.96 183.09 | `runs/quality/paper16_pending_uniform_latency/dsA_groupaware_clean_aug_v2__literature_repro__gpc_yolov8n__seed{0,1,2}/uniform_latency_results.json` |
| FSDH-YOLO | 2.997 | 123.39±2.94 | 124.34 120.09 125.74 | `runs/quality/paper16_spo_fsd_uniform_latency/uniform_latency_results.csv` |

**逐种子结果**

| Model | seed | P/% | R/% | mAP50/% | mAP50-95/% | FPS |
|---|---|---|---|---|---|---|
| YOLO11n | 0 | 77.117 | 80.436 | 80.068 | 65.120 | 150.57 |
| YOLO11n | 1 | 79.841 | 76.257 | 78.246 | 64.151 | 148.28 |
| YOLO11n | 2 | 77.063 | 77.829 | 78.637 | 64.016 | 141.23 |
| YOLOv5nu | 0 | 76.997 | 77.039 | 78.671 | 63.617 | 161.33 |
| YOLOv5nu | 1 | 74.916 | 79.366 | 80.506 | 65.445 | 168.08 |
| YOLOv5nu | 2 | 80.253 | 76.904 | 80.650 | 65.796 | 170.82 |
| YOLOv7-tiny | 0 | 78.446 | 77.212 | 84.414 | 66.301 | 188.92 |
| YOLOv7-tiny | 1 | 78.155 | 77.399 | 84.257 | 66.412 | 190.42 |
| YOLOv7-tiny | 2 | 75.179 | 81.986 | 84.106 | 66.311 | 189.09 |
| YOLOv8n | 0 | 77.052 | 79.376 | 79.107 | 64.740 | 178.05 |
| YOLOv8n | 1 | 78.157 | 78.421 | 78.918 | 64.830 | 186.64 |
| YOLOv8n | 2 | 79.155 | 79.192 | 81.072 | 65.680 | 185.60 |
| YOLO26n | 0 | 77.406 | 75.954 | 81.579 | 67.001 | 136.45 |
| YOLO26n | 1 | 77.381 | 78.013 | 83.297 | 67.536 | 138.15 |
| YOLO26n | 2 | 77.964 | 77.427 | 81.866 | 66.277 | 135.39 |
| RT-DETRv2-S | 0 | 73.708 | 78.814 | 82.062 | 67.131 | 60.63 |
| RT-DETRv2-S | 1 | 76.244 | 73.781 | 80.892 | 65.272 | 63.12 |
| RT-DETRv2-S | 2 | 73.925 | 78.425 | 81.375 | 65.549 | 81.02 |
| PEANet-YOLOv5-D | 0 | 78.442 | 78.593 | 81.795 | 66.207 | 137.78 |
| PEANet-YOLOv5-D | 1 | 78.310 | 78.669 | 79.320 | 65.455 | 140.29 |
| PEANet-YOLOv5-D | 2 | 75.853 | 81.305 | 80.672 | 66.699 | 141.90 |
| YOLOv8-FPDW | 0 | 79.714 | 78.201 | 80.291 | 66.133 | 142.38 |
| YOLOv8-FPDW | 1 | 79.012 | 78.329 | 81.137 | 66.805 | 154.47 |
| YOLOv8-FPDW | 2 | 77.704 | 78.053 | 78.953 | 64.760 | 150.79 |
| Edge-YOLOv11 | 0 | 80.186 | 78.275 | 81.142 | 66.704 | 67.16 |
| Edge-YOLOv11 | 1 | 77.380 | 78.825 | 77.350 | 63.872 | 68.53 |
| Edge-YOLOv11 | 2 | 79.903 | 76.605 | 80.927 | 66.609 | 66.51 |
| GPC-YOLO | 0 | 78.378 | 77.989 | 79.202 | 64.925 | 173.08 |
| GPC-YOLO | 1 | 78.783 | 77.120 | 80.455 | 66.563 | 163.96 |
| GPC-YOLO | 2 | 80.020 | 79.880 | 83.527 | 67.888 | 183.09 |
| FSDH-YOLO | 0 | 78.581 | 78.364 | 80.931 | 66.670 | 124.34 |
| FSDH-YOLO | 1 | 78.900 | 79.608 | 81.500 | 67.140 | 120.09 |
| FSDH-YOLO | 2 | 82.485 | 77.358 | 82.866 | 68.080 | 125.74 |

> 所有模型均在相同数据划分、训练策略与评价协议下重新训练与测试；不与原文报告结果横向比较。
> 复现模型：Edge-YOLOv11 依公开结构复现，PEANet-YOLOv5-D / YOLOv8-FPDW / GPC-YOLO 依原文描述与公开组件重构；
> YOLOv8-FPDW 使用 `YOLOv8_FPDW_FIXED`（WIoU fix v2）。

<details><summary>数据来源路径</summary>

| 内容 | 路径 |
|---|---|
| 逐 seed（统一协议 5 模型） | `analysis/paper16_unified_common_protocol/per_seed.csv（key: yolo11n, yolov5nu, yolov8n, yolo26n, yolov7_tiny, rtdetrv2_s, fsdconv_shab）` |
| 逐 seed AP75 索引 | `analysis/paper_3_2_ap75_shab/AP75_逐种子.csv（含 31 个 key）` |
| Common Evaluator 指标 JSON（统一协议/外部框架） | `runs/common_eval/paper16_unified_protocol/{...}/ 与 runs/external_eval/dsA_groupaware_clean_aug_v2__comparison__{yolov7_tiny,rtdetrv2_s}/seed{0,1,2}__test_metrics_common.json` |
| Common Evaluator 指标 JSON（复现模型） | `runs/common_eval/paper16_pending_experiments/{PEANet_YOLOv5_D,Edge_YOLOv11,YOLOv8_FPDW_FIXED,GPC_YOLO}/seed{0,1,2}__test_metrics_common.json` |
| 外部框架权重（仓库外） | `<data-root>/paper/tomato/yolov7-main/runs/train/...；<data-root>/paper/general journal/RT-DETR/rtdetrv2_pytorch/output/...` |
| FPS（统一协议） | `runs/quality/paper16_non_fsd_uniform_latency/ 与 runs/quality/paper16_spo_fsd_uniform_latency/uniform_latency_results.csv` |
| FPS（复现模型） | `runs/quality/paper16_pending_uniform_latency/<配置>__seed{0,1,2}/uniform_latency_results.json` |
| FPS（外部框架） | `runs/external_eval/dsA_groupaware_clean_aug_v2__comparison__{yolov7_tiny,rtdetrv2_s}/seed{0,1,2}__latency_fp32_b1_640.json（+ three_seed_mean_sd.csv）` |

</details>


## 1.8 跨数据集适用性实验（3.8）

Dataset B = lichi-maturity：4 个原始成熟类别 **0 green / 1 half / 2 red / 3 young**（未合并或重映射）；训练 1,071 / 验证 212 / 测试 212 原图，仅训练集做三类独立变体离线增强（增强后 4,284 张）。
面积口径：Dataset B 原始图像即为 640×640 尺度，原图→输入无缩放，故 `AP_small(640)` 采用 COCO 原版阈值（原图面积 < 1024 px²），与标准 COCO AP_s 等价。来源文件的 `area_definition` 字段记作 `original-image area < 1024 pixels`。

**三随机种子均值 ± 样本标准差**

| Model | P/% | R/% | mAP50/% | mAP50-95/% | AP75/% | AP_small(640)/% |
|---|---|---|---|---|---|---|
| YOLO11n | 81.097±1.471 | 82.504±2.222 | 84.902±0.748 | 61.238±0.374 | 70.577±0.443 | 32.332±1.812 |
| FSDH-YOLO-n | 81.073±2.128 | 82.820±0.674 | 84.492±0.707 | 60.210±0.621 | 70.294±0.630 | 32.524±1.329 |

**逐种子结果**

| Model | seed | P/% | R/% | mAP50/% | mAP50-95/% | AP75/% | AP_small(640)/% |
|---|---|---|---|---|---|---|---|
| YOLO11n | 0 | 80.060 | 84.385 | 85.729 | 61.466 | 70.069 | 30.335 |
| YOLO11n | 1 | 82.781 | 80.053 | 84.706 | 61.442 | 70.882 | 32.790 |
| YOLO11n | 2 | 80.451 | 83.075 | 84.272 | 60.807 | 70.779 | 33.870 |
| FSDH-YOLO-n | 0 | 79.807 | 83.488 | 83.993 | 59.683 | 69.845 | 32.795 |
| FSDH-YOLO-n | 1 | 83.530 | 82.141 | 85.301 | 60.895 | 71.015 | 33.697 |
| FSDH-YOLO-n | 2 | 79.882 | 82.831 | 84.181 | 60.051 | 70.023 | 31.080 |

**分类别 AP50 / AP50-95**

| Class | Model | AP50/% (均值±SD) | AP50 逐 seed | AP50-95/% (均值±SD) | AP50-95 逐 seed |
|---|---|---|---|---|---|
| green | YOLO11n | 89.155±0.334 | 89.267 / 88.780 / 89.419 | 64.945±0.498 | 64.526 / 64.815 / 65.495 |
| green | FSDH-YOLO-n | 90.947±0.674 | 90.268 / 91.615 / 90.959 | 66.475±0.336 | 66.244 / 66.861 / 66.320 |
| half | YOLO11n | 70.062±2.834 | 73.180 / 69.367 / 67.641 | 57.935±2.923 | 61.218 / 56.976 / 55.612 |
| half | FSDH-YOLO-n | 66.015±1.698 | 64.912 / 67.970 / 65.163 | 53.660±0.842 | 52.726 / 54.360 / 53.894 |
| red | YOLO11n | 87.199±0.344 | 87.401 / 86.802 / 87.395 | 57.507±1.198 | 56.132 / 58.060 / 58.328 |
| red | FSDH-YOLO-n | 86.815±0.352 | 86.717 / 87.206 / 86.522 | 55.831±0.861 | 55.308 / 56.825 / 55.359 |
| young | YOLO11n | 93.192±0.631 | 93.067 / 93.876 / 92.632 | 64.565±1.175 | 63.987 / 65.916 / 63.791 |
| young | FSDH-YOLO-n | 94.190±0.192 | 94.077 / 94.413 / 94.082 | 64.873±0.580 | 64.453 / 65.534 / 64.632 |

**配对差值（FSDH-YOLO-n − YOLO11n）**

| 指标 | Δ/pp | 逐 seed 差值 |
|---|---|---|
| P | -0.025 | -0.254 / +0.748 / -0.568 |
| R | +0.316 | -0.897 / +2.089 / -0.244 |
| mAP50 | -0.410 | -1.735 / +0.595 / -0.090 |
| mAP50_95 | -1.028 | -1.783 / -0.547 / -0.755 |
| AP75 | -0.282 | -0.224 / +0.133 / -0.756 |
| AP_small | +0.192 | +2.460 / +0.907 / -2.790 |
| green_AP50 | +1.792 | +1.001 / +2.836 / +1.541 |
| green_AP | +1.529 | +1.718 / +2.045 / +0.825 |
| half_AP50 | -4.047 | -8.268 / -1.396 / -2.478 |
| half_AP | -4.275 | -8.492 / -2.615 / -1.719 |
| red_AP50 | -0.385 | -0.684 / +0.404 / -0.874 |
| red_AP | -1.676 | -0.825 / -1.235 / -2.968 |
| young_AP50 | +0.999 | +1.010 / +0.537 / +1.450 |
| young_AP | +0.308 | +0.466 / -0.382 / +0.841 |

<details><summary>数据来源路径</summary>

| 内容 | 路径 |
|---|---|
| 本实验唯一权威源（含协议、GT、逐 seed、配对差值） | `analysis/paper_dataset_b_groupaware_clean_v4_aug_common_v2__main_results.json` |
| 人读版 | `analysis/paper_dataset_b_groupaware_clean_v4_aug_common_v2__main_results.md` |
| AP75 专项 CSV | `analysis/paper_dataset_b_groupaware_clean_v4_aug_common_v2__main_results_task_ap75.csv` |
| AP_small 专项 CSV | `analysis/paper_dataset_b_groupaware_clean_v4_aug_common_v2__main_results_task_apsmall_coco.csv` |
| 类别 AP 专项 CSV | `analysis/paper_dataset_b_groupaware_clean_v4_aug_common_v2__main_results_task_maturity_ap.csv` |
| Common Evaluator 指标 JSON | `runs/common_eval/paper_dataset_b_groupaware_clean_v4_aug_common_v2/{DatasetB4_YOLO11n,DatasetB4_FSDH_YOLO}/seed{0,1,2}__test_metrics_common.json` |
| 冻结标注 | `external_configs/dataset_b_native4_v4/instances_test.json` |
| 训练/测速 | `runs/detect/paper_dataset_b_groupaware_clean_v4_aug_common_v2/；runs/latency/paper_dataset_b_groupaware_clean_v4_aug_comparison_v1/` |
| Dataset B 构建与增强 | `<data-root>/data/lichi-maturity/lichi-maturity/lichi_701515_groupaware_clean_v4_raw/build_summary.json；..._v4_aug/augmentation_summary.json；两个 .yaml` |
| 独立复核 | `analysis/dataset_b_v4_verification.json；analysis/dataset_b_v4_protocol_audit/` |

</details>

> 论文 3.8 表格**不含 Params 与 FPS 列**，故本实验不列 FPS。
> 附（非论文表格列，仅供备查）：同源测速 YOLO11n `165.56±3.10`、FSDH-YOLO-n `140.06±2.34`（s0/s1/s2：168.85 162.69 165.14｜138.80 138.63 142.76）。

---

# 二、附录 B 实验（4 个）

## B.1 YOLO11 模型规模选择分析实验

**三随机种子均值 ± 样本标准差**

| Model | P/% | R/% | mAP50/% | mAP50-95/% |
|---|---|---|---|---|
| YOLO11n | 78.007±1.588 | 78.174±2.110 | 78.984±0.959 | 64.429±0.603 |
| YOLO11s | 76.002±0.901 | 77.081±1.601 | 74.890±2.703 | 61.400±1.821 |
| YOLO11m | 77.540±0.691 | 75.779±2.693 | 77.169±4.401 | 63.015±3.452 |

**FPS + Params（三 seed 均值 ± 样本标准差）**

| Model | Params/M | FPS | 逐 seed (s0/s1/s2) | 来源文件 |
|---|---|---|---|---|
| YOLO11n | 2.583 | 146.69±4.87 | 150.57 148.28 141.23 | `runs/quality/paper16_non_fsd_uniform_latency/uniform_latency_results.csv` |
| YOLO11s | 9.414 | 132.15±3.42 | 131.76 135.74 128.94 | `runs/quality/paper16_non_fsd_uniform_latency/uniform_latency_results.csv` |
| YOLO11m | 20.032 | 103.40±0.81 | 102.85 104.33 103.02 | `runs/quality/paper16_non_fsd_uniform_latency/uniform_latency_results.csv` |

**逐种子结果**

| Model | seed | P/% | R/% | mAP50/% | mAP50-95/% | FPS |
|---|---|---|---|---|---|---|
| YOLO11n | 0 | 77.117 | 80.436 | 80.068 | 65.120 | 150.57 |
| YOLO11n | 1 | 79.841 | 76.257 | 78.246 | 64.151 | 148.28 |
| YOLO11n | 2 | 77.063 | 77.829 | 78.637 | 64.016 | 141.23 |
| YOLO11s | 0 | 76.909 | 76.774 | 75.708 | 61.927 | 131.76 |
| YOLO11s | 1 | 75.107 | 78.814 | 77.089 | 62.899 | 135.74 |
| YOLO11s | 2 | 75.990 | 75.657 | 71.872 | 59.375 | 128.94 |
| YOLO11m | 0 | 78.007 | 75.290 | 76.765 | 62.837 | 102.85 |
| YOLO11m | 1 | 77.866 | 73.364 | 72.985 | 59.656 | 104.33 |
| YOLO11m | 2 | 76.746 | 78.684 | 81.758 | 66.552 | 103.02 |

<details><summary>数据来源路径</summary>

| 内容 | 路径 |
|---|---|
| 逐 seed 精度 | `analysis/paper16_unified_common_protocol/per_seed.csv（key: yolo11n, yolo11s, yolo11m）` |
| Common Evaluator 指标 JSON | `runs/common_eval/paper16_unified_protocol/{yolo11n,yolo11s,yolo11m}/seed{0,1,2}__test_metrics_common.json` |
| FPS + Params | `runs/quality/paper16_non_fsd_uniform_latency/uniform_latency_three_seed_mean_sd.csv（+ uniform_latency_results.csv）` |

</details>

> 目的：确定与当前数据规模和任务复杂度匹配的模型容量，为结构改进提供适配基准。

## B.2 SPDConv 与 FSDConv 模块对比实验

> ⚠️ **来源边界（重要）**：本实验的 Common Evaluator 预测/指标 JSON **在仓库中不存在**（`runs/common_eval/` 下无任何 spd 条目；亦无对应 `seedN__test_predictions_coco.json`）。
> 下表 SPDConv 与 SPDConv+SHEB 的逐 seed 精度取自论文既有记录（清单 12.2 节），已复算与其汇总行自洽；
> 其余三行来自标准 Common Evaluator 留档。训练权重存在，可据此补算并补存 JSON。

**三随机种子均值 ± 样本标准差**

| Model | P/% | R/% | mAP50/% | mAP50-95/% | Params/M | FPS |
|---|---|---|---|---|---|---|
| YOLO11n | 78.007±1.588 | 78.174±2.110 | 78.984±0.959 | 64.429±0.603 | 2.583 | 146.69±4.87 |
| SPDConv | 77.338±1.265 | 78.315±1.055 | 78.539±2.198 | 64.709±1.669 | 2.584 | 147.50±3.14 |
| FSDConv | 76.927±1.346 | 79.188±3.066 | 80.563±1.522 | 66.292±0.971 | 2.584 | 148.99±0.45 |
| SPDConv+SHEB | 78.058±1.128 | 79.519±1.138 | 80.773±0.241 | 66.649±0.335 | 2.997 | 125.45±2.16 |
| FSDH-YOLO（FSDConv+SHEB） | 79.989±2.168 | 78.443±1.127 | 81.766±0.994 | 67.297±0.718 | 2.997 | 123.39±2.94 |

**逐种子结果**

| Model | seed | P/% | R/% | mAP50/% | mAP50-95/% | FPS |
|---|---|---|---|---|---|---|
| YOLO11n | 0 | 77.117 | 80.436 | 80.068 | 65.120 | 150.57 |
| YOLO11n | 1 | 79.841 | 76.257 | 78.246 | 64.151 | 148.28 |
| YOLO11n | 2 | 77.063 | 77.829 | 78.637 | 64.016 | 141.23 |
| SPDConv | 0 | 76.294 | 77.168 | 76.121 | 62.912 | 147.97 |
| SPDConv | 1 | 76.975 | 79.244 | 79.081 | 65.005 | 150.38 |
| SPDConv | 2 | 78.744 | 78.534 | 80.415 | 66.211 | 144.15 |
| FSDConv | 0 | 75.375 | 82.672 | 81.487 | 66.541 | 148.49 |
| FSDConv | 1 | 77.780 | 77.989 | 81.395 | 67.115 | 149.34 |
| FSDConv | 2 | 77.626 | 76.902 | 78.806 | 65.221 | 149.14 |
| SPDConv+SHEB | 0 | 76.774 | 80.833 | 80.527 | 66.701 | 125.36 |
| SPDConv+SHEB | 1 | 78.886 | 78.882 | 80.783 | 66.291 | 127.65 |
| SPDConv+SHEB | 2 | 78.515 | 78.842 | 81.009 | 66.954 | 123.33 |
| FSDH-YOLO（FSDConv+SHEB） | 0 | 78.581 | 78.364 | 80.931 | 66.670 | 124.34 |
| FSDH-YOLO（FSDConv+SHEB） | 1 | 78.900 | 79.608 | 81.500 | 67.140 | 120.09 |
| FSDH-YOLO（FSDConv+SHEB） | 2 | 82.485 | 77.358 | 82.866 | 68.080 | 125.74 |

<details><summary>数据来源路径</summary>

| 内容 | 路径 |
|---|---|
| 逐 seed 精度（SPDConv / SPDConv+SHEB） | `analysis/paper_3_4_review/论文3.4_实验章节架构与参数结果清单.md → 12.2 节（论文既有记录；无 Common Evaluator JSON）` |
| 三 seed 汇总（论文记录） | `同上清单 → 12.3 节` |
| 逐 seed 精度（YOLO11n / FSDConv / FSDH-YOLO） | `analysis/paper16_unified_common_protocol/per_seed.csv（key: yolo11n, fsdconv, fsdconv_shab）` |
| FPS（SPDConv / SPDConv+SHEB） | `runs/quality/paper_groupaware_clean_v2_reruns/uniform_latency/uniform_latency_results.csv（+ three_seed_mean_sd.csv）` |
| FPS（其余） | `runs/quality/paper16_non_fsd_uniform_latency/ 与 runs/quality/paper16_spo_fsd_uniform_latency/uniform_latency_results.csv` |
| 既有权重（可据此补算 CE） | `runs/detect/paper_groupaware_clean_v2_reruns/dsA_groupaware_clean_aug_v2__spdcontrol__{spdconv,spdconv_shab}__seed{0,1,2}/weights/best.pt` |

</details>

> 两层对比：第一层仅换输入端首个下采样模块（SPDConv vs FSDConv）；第二层保持完整 SHEB 不变（SPDConv+SHEB vs FSDH-YOLO）。
> 两种实现输出尺度、通道数与参数量一致，故差异不能归因于容量。

## B.3 SHEB 分支消融实验

**三随机种子均值 ± 样本标准差**

| Model | P/% | R/% | mAP50/% | mAP50-95/% |
|---|---|---|---|---|
| YOLO11n | 78.007±1.588 | 78.174±2.110 | 78.984±0.959 | 64.429±0.603 |
| SHEB-L（仅局部） | 77.538±0.772 | 78.734±0.584 | 79.500±2.351 | 65.161±1.584 |
| SHEB-M（仅中程） | 78.527±0.247 | 80.109±0.311 | 81.856±0.847 | 67.089±0.567 |
| SHEB-G（仅全局） | 75.768±0.300 | 80.206±1.870 | 80.752±1.278 | 66.104±0.910 |
| SHEB-LM | 76.952±0.807 | 78.858±0.697 | 80.397±0.443 | 65.773±0.246 |
| SHEB-LG | 76.489±0.694 | 79.966±0.681 | 79.998±1.697 | 65.584±1.258 |
| SHEB-MG | 76.854±1.118 | 79.904±2.080 | 80.209±1.279 | 65.984±0.739 |
| SHEB(-LMG) | 78.342±0.196 | 78.854±1.244 | 81.092±2.012 | 66.215±1.142 |
| FSDConv+SHEB-M | 78.352±2.425 | 78.986±1.084 | 80.173±1.393 | 65.947±1.007 |
| FSDH-YOLO | 79.989±2.168 | 78.443±1.127 | 81.766±0.994 | 67.297±0.718 |

**FPS + Params（三 seed 均值 ± 样本标准差）**

| Model | Params/M | FPS | 逐 seed (s0/s1/s2) | 来源文件 |
|---|---|---|---|---|
| YOLO11n | 2.583 | 146.69±4.87 | 150.57 148.28 141.23 | `runs/quality/paper16_non_fsd_uniform_latency/uniform_latency_results.csv` |
| SHEB-L（仅局部） | 2.711 | 147.14±4.01 | 151.77 144.69 144.96 | `runs/latency/paper16_shab_branch_subset_and_param_match_v1/dsA_groupaware_clean_aug_v2__shab_branch_subset__L__seed{0,1,2}/uniform_latency_results.json` |
| SHEB-M（仅中程） | 2.720 | 151.38±3.70 | 150.11 155.55 148.48 | `runs/latency/paper16_shab_branch_subset_and_param_match_v1/dsA_groupaware_clean_aug_v2__shab_branch_subset__M__seed{0,1,2}/uniform_latency_results.json` |
| SHEB-G（仅全局） | 2.731 | 146.02±2.53 | 145.00 144.17 148.91 | `runs/latency/paper16_shab_branch_subset_and_param_match_v1/dsA_groupaware_clean_aug_v2__shab_branch_subset__G__seed{0,1,2}/uniform_latency_results.json` |
| SHEB-LM | 2.848 | 141.97±4.29 | 137.34 142.77 145.81 | `runs/latency/paper16_shab_branch_subset_and_param_match_v1/dsA_groupaware_clean_aug_v2__shab_branch_subset__LM__seed{0,1,2}/uniform_latency_results.json` |
| SHEB-LG | 2.859 | 139.31±6.05 | 145.71 138.53 133.68 | `runs/latency/paper16_shab_branch_subset_and_param_match_v1/dsA_groupaware_clean_aug_v2__shab_branch_subset__LG__seed{0,1,2}/uniform_latency_results.json` |
| SHEB-MG | 2.868 | 142.35±5.32 | 136.96 147.61 142.49 | `runs/latency/paper16_shab_branch_subset_and_param_match_v1/dsA_groupaware_clean_aug_v2__shab_branch_subset__MG__seed{0,1,2}/uniform_latency_results.json` |
| SHEB(-LMG) | 2.996 | 122.01±1.17 | 122.57 122.80 120.67 | `runs/quality/paper16_non_fsd_uniform_latency/uniform_latency_results.csv` |
| FSDConv+SHEB-M | 2.721 | 148.81±3.00 | 149.84 151.17 145.43 | `runs/latency/paper16_fsdconv_shab_m_confirmation_v1/dsA_groupaware_clean_aug_v2__fsd_shab_m_confirmation__fsdconv_shab_m__seed{0,1,2}/uniform_latency_results.json` |
| FSDH-YOLO | 2.997 | 123.39±2.94 | 124.34 120.09 125.74 | `runs/quality/paper16_spo_fsd_uniform_latency/uniform_latency_results.csv` |

**逐种子结果**

| Model | seed | P/% | R/% | mAP50/% | mAP50-95/% | FPS |
|---|---|---|---|---|---|---|
| YOLO11n | 0 | 77.117 | 80.436 | 80.068 | 65.120 | 150.57 |
| YOLO11n | 1 | 79.841 | 76.257 | 78.246 | 64.151 | 148.28 |
| YOLO11n | 2 | 77.063 | 77.829 | 78.637 | 64.016 | 141.23 |
| SHEB-L（仅局部） | 0 | 76.646 | 78.846 | 76.910 | 63.457 | 151.77 |
| SHEB-L（仅局部） | 1 | 77.984 | 79.254 | 81.500 | 66.589 | 144.69 |
| SHEB-L（仅局部） | 2 | 77.984 | 78.101 | 80.088 | 65.435 | 144.96 |
| SHEB-M（仅中程） | 0 | 78.351 | 80.461 | 82.794 | 67.724 | 150.11 |
| SHEB-M（仅中程） | 1 | 78.421 | 79.997 | 81.148 | 66.630 | 155.55 |
| SHEB-M（仅中程） | 2 | 78.809 | 79.870 | 81.627 | 66.914 | 148.48 |
| SHEB-G（仅全局） | 0 | 75.933 | 78.154 | 79.301 | 65.057 | 145.00 |
| SHEB-G（仅全局） | 1 | 75.948 | 81.812 | 81.710 | 66.705 | 144.17 |
| SHEB-G（仅全局） | 2 | 75.422 | 80.653 | 81.245 | 66.551 | 148.91 |
| SHEB-LM | 0 | 77.882 | 79.049 | 80.326 | 65.700 | 137.34 |
| SHEB-LM | 1 | 76.544 | 79.439 | 80.871 | 66.048 | 142.77 |
| SHEB-LM | 2 | 76.431 | 78.085 | 79.993 | 65.573 | 145.81 |
| SHEB-LG | 0 | 75.739 | 80.089 | 78.173 | 64.350 | 145.71 |
| SHEB-LG | 1 | 77.108 | 80.578 | 81.530 | 66.865 | 138.53 |
| SHEB-LG | 2 | 76.619 | 79.232 | 80.292 | 65.537 | 133.68 |
| SHEB-MG | 0 | 78.033 | 77.647 | 79.558 | 65.673 | 136.96 |
| SHEB-MG | 1 | 76.719 | 80.323 | 79.386 | 65.452 | 147.61 |
| SHEB-MG | 2 | 75.810 | 81.742 | 81.683 | 66.828 | 142.49 |
| SHEB(-LMG) | 0 | 78.173 | 79.970 | 83.031 | 67.388 | 122.57 |
| SHEB(-LMG) | 1 | 78.297 | 79.081 | 79.013 | 65.106 | 122.80 |
| SHEB(-LMG) | 2 | 78.556 | 77.513 | 81.233 | 66.152 | 120.67 |
| FSDConv+SHEB-M | 0 | 75.832 | 80.153 | 79.147 | 65.081 | 149.84 |
| FSDConv+SHEB-M | 1 | 80.670 | 78.011 | 79.613 | 65.708 | 151.17 |
| FSDConv+SHEB-M | 2 | 78.553 | 78.794 | 81.759 | 67.052 | 145.43 |
| FSDH-YOLO | 0 | 78.581 | 78.364 | 80.931 | 66.670 | 124.34 |
| FSDH-YOLO | 1 | 78.900 | 79.608 | 81.500 | 67.140 | 120.09 |
| FSDH-YOLO | 2 | 82.485 | 77.358 | 82.866 | 68.080 | 125.74 |

<details><summary>数据来源路径</summary>

| 内容 | 路径 |
|---|---|
| 逐 seed 精度（6 个分支） | `runs/common_eval/paper16_shab_branch_subset_and_param_match_v1/SHAB_{L,M,G,LM,LG,MG}/seed{0,1,2}__test_metrics_common.json` |
| 逐 seed 精度（YOLO11n / SHEB(-LMG) / FSDH-YOLO） | `analysis/paper16_unified_common_protocol/per_seed.csv（key: yolo11n, shab, fsdconv_shab）` |
| 逐 seed 精度（FSDConv+SHEB-M） | `runs/common_eval/paper16_fsdconv_shab_m_confirmation_v1/FSDCONV_SHAB_M/seed{0,1,2}__test_metrics_common.json` |
| FPS（6 个分支 + YOLO11n） | `runs/latency/paper16_shab_branch_subset_and_param_match_v1/dsA_groupaware_clean_aug_v2__shab_branch_subset__<分支>__seed{0,1,2}/uniform_latency_results.json` |
| FPS（FSDConv+SHEB-M） | `runs/latency/paper16_fsdconv_shab_m_confirmation_v1/dsA_groupaware_clean_aug_v2__fsd_shab_m_confirmation__fsdconv_shab_m__seed{0,1,2}/uniform_latency_results.json` |
| FPS（YOLO11n / SHEB(-LMG) / FSDH-YOLO） | `runs/quality/paper16_non_fsd_uniform_latency/ 与 runs/quality/paper16_spo_fsd_uniform_latency/uniform_latency_results.csv` |

</details>


## B.4 Bootstrap 稳健性分析（仅汇总，无逐种子要求）

逐图像配对 Bootstrap：从 126 张测试图像中有放回抽样、重建同规模测试样本，重复 **10,000** 次；三 seed 与全部待比较模型在每次重采样中**共用同一图像索引**；先在同一 seed 内计算模型差值，再对三 seed 求平均；95% CI 采用百分位法。该区间仅量化**测试图像抽样**带来的不确定性，训练随机性由三 seed 的 mean±SD 与同 seed 配对方向一致性另行判断。

| 指标 | YOLO11n/% | FSDH-YOLO/% | Δ/pp | Bootstrap 95% CI/pp | 判断 |
|---|---|---|---|---|---|
| mAP50-95 | 64.429±0.603 | 67.297±0.718 | +2.868 | [+1.506, +4.215] | 支持正向提升 |
| mAP50 | 78.984±0.959 | 81.766±0.994 | +2.782 | [+1.179, +4.513] | 支持正向提升 |
| AP75 | 74.441±1.029 | 77.195±0.998 | +2.754 | [+1.169, +4.331] | 支持正向提升 |
| AP_small@640eq | 55.204±0.616 | 58.065±1.405 | +2.861 | [+1.176, +4.697] | 支持正向提升 |
| COCO AP_s | 38.581±5.158 | 37.146±4.610 | -1.435 | [-6.264, +4.244] | 区间跨 0，不支持 |
| unripe AP50 | 87.291±1.106 | 89.231±0.577 | +1.940 | [+0.593, +3.339] | 支持正向提升 |
| unripe AP50-95 | 69.915±0.668 | 71.557±0.052 | +1.642 | [+0.386, +2.959] | 支持正向提升 |
| unripe AP_small@640eq | 62.011±0.623 | 62.702±0.446 | +0.691 | [-1.125, +3.190] | 区间跨 0，不支持 |
| unripe AP_non-small | 77.171±1.029 | 79.174±0.312 | +2.002 | [+0.353, +3.558] | 支持正向提升 |
| semi-ripe AP50 | 68.926±3.093 | 70.174±0.919 | +1.248 | [-1.753, +4.284] | 区间跨 0，不支持 |
| semi-ripe AP50-95 | 56.482±2.641 | 58.705±0.807 | +2.223 | [-0.355, +4.651] | 区间跨 0，不支持 |
| semi-ripe AP_small@640eq | 41.777±2.455 | 46.437±2.666 | +4.660 | [+0.905, +8.335] | 支持正向提升 |
| semi-ripe AP_non-small | 59.896±2.832 | 61.511±0.847 | +1.615 | [-1.428, +4.381] | 区间跨 0，不支持 |
| ripe AP50 | 80.734±1.364 | 85.892±1.903 | +5.158 | [+3.322, +7.594] | 支持正向提升 |
| ripe AP50-95 | 66.890±1.139 | 71.629±1.616 | +4.740 | [+3.264, +6.532] | 支持正向提升 |
| ripe AP_small@640eq | 61.824±0.773 | 65.057±1.308 | +3.233 | [+1.220, +5.450] | 支持正向提升 |
| ripe AP_non-small | 68.263±1.271 | 73.439±1.893 | +5.176 | [+3.326, +7.407] | 支持正向提升 |
| IoU localization error | 9.577±0.224 | 9.394±0.170 | -0.183 | [-0.320, -0.049] | 支持误差下降 |

<details><summary>数据来源路径</summary>

| 内容 | 路径 |
|---|---|
| Bootstrap 全部指标（CSV） | `analysis/paper_3_4_review/paper35_extended_statistics/bootstrap_ci_extended.csv` |
| 协议 / 逐 seed / 输入绝对路径（JSON） | `analysis/paper_3_4_review/paper35_extended_statistics/bootstrap_results_extended.json` |
| 复算脚本 | `scripts/paper35_extended_statistics.py` |

</details>

> 解读边界：`AP_small(640)`（889 个 GT）与标准 `COCO AP_s`（37 个 GT）口径与样本量不同，须分别标注，不可混称。
> 三级类别 AP50-95 均值均上升，但仅 unripe 与 ripe 的区间未跨 0；semi-ripe 区间跨 0，故不宜表述为「过渡类别已稳定改善」。

---

# 三、来源路径总索引

| 用途 | 路径（相对仓库根 `<original-workspace>`） |
|---|---|
| 逐 seed 精度总索引（30 key / 90 行，含 metrics_json + checkpoint 两列） | `analysis/paper16_unified_common_protocol/per_seed.csv` |
| 三 seed 均值±SD 总索引 | `analysis/paper16_unified_common_protocol/three_seed_mean_sd.csv` |
| per-seed AP75 / mAP 索引（31 key / 93 行） | `analysis/paper_3_2_ap75_shab/AP75_逐种子.csv` |
| 七模块 Common Evaluator 汇总 + 逐 seed provenance | `analysis/shab_comparison_common_protocol/summary.json` |
| 面积分层 AP（Dataset A） | `analysis/paper27_statistical_supplement/dataset_a_area_ap.csv` |
| 面积分层 + 混淆矩阵 + Bootstrap 复算结果 | `analysis/paper27_statistical_supplement/results.json` |
| IoU 定位误差 / Bootstrap 扩展 | `analysis/paper_3_4_review/paper35_extended_statistics/bootstrap_results_extended.json` |
| Dataset B 权威结果 | `analysis/paper_dataset_b_groupaware_clean_v4_aug_common_v2__main_results.json` |
| Common Evaluator 指标 JSON 根 | `runs/common_eval/` |
| 统一测速（多尺度模块，另跑） | `runs/quality/shab_comparison_uniform_latency_fp32_b1_640/` |
| 统一测速（非 FSD 模型） | `runs/quality/paper16_non_fsd_uniform_latency/` |
| 统一测速（FSD 模型） | `runs/quality/paper16_spo_fsd_uniform_latency/` |
| 统一测速（SHEB 分支 / YOLO11n-PM） | `runs/latency/paper16_shab_branch_subset_and_param_match_v1/` |
| 统一测速（SPDConv 对照） | `runs/quality/paper_groupaware_clean_v2_reruns/uniform_latency/` |
| 统一测速（文献复现模型） | `runs/quality/paper16_pending_uniform_latency/` |
| 统一测速（外部框架 YOLOv7/RT-DETR） | `runs/external_eval/dsA_groupaware_clean_aug_v2__comparison__{yolov7_tiny,rtdetrv2_s}/` |
| 训练权重、超参（args.yaml）与环境记录（final_summary.txt） | `runs/detect/paper_groupaware_clean_v2_reruns/<配置>__seed{0,1,2}/` |
| 模型结构配置 | `ultralytics/cfg/models/11/` |
| Dataset A 构建与增强记录 | `<data-root>/data/An RGB-D Image Dataset for Lychee Detection and Ma/.../lychee_groupaware_clean_{raw,aug}_v2/` |
| Dataset B 构建与增强记录 | `<data-root>/data/lichi-maturity/lichi-maturity/lichi_701515_groupaware_clean_v4_{raw,aug}/` |
| 实验章节架构与参数结果清单（主索引文档） | `analysis/paper_3_4_review/论文3.4_实验章节架构与参数结果清单.md` |

---

## 四、口径差异登记（不改正文，仅备核对）

| # | 项 | 说明 |
|---|---|---|
| 1 | 表 3 的 SHEB FPS 与同表其余 6 行不同轮次（已按统一口径取值） | SHEB 全篇统一取 `paper16_non_fsd_uniform_latency` 的 **122.01±1.17**（与表 4、附录 B.3 一致）；同表 6 个模块取 `..._fp32_b1_640`。两轮为同机不同时间测速，后者中 SHEB 为 122.48±1.80，相差 0.47 FPS。此为**有意选择**，用以保证全篇 SHEB 只有一个 FPS；若改为表内 7 行严格同批，则 SHEB 须用 122.48±1.80，并需连带修改表 4、附录 B.3 及两处正文。 |
| 2 | 附录 RepConv 逐 seed 行 | 源 `summary.json` 中 RepConv 三 seed 为 77.828/79.842/81.065/65.703、75.614/80.702/80.478/65.892、77.658/77.821/79.662/65.344，与 DilatedConv 完全不同；论文附录该三行现与 DilatedConv 逐位相同。表 3 的 RepConv 均值±SD 用的是正确数据。 |
| 3 | 附录汇总行「FSDConv+SHEB」 | 其四项精度等于 SPDConv+SHEB（见 B.2），FPS 亦为 SPDConv+SHEB 的 125.45±2.16，行名与数据不符。 |
| 4 | B.2 无 Common Evaluator 留档 | 投稿归档前需补存 SPDConv / SPDConv+SHEB 的预测与指标 JSON。 |
| 5 | Δ 舍入口径不一致（共 5 处，均为「先各自四舍五入再相减」而非配对统计） | ① 表 8 ripe AP50-95：配对值 +4.7396 → **+4.740**，论文写 +4.739；② Dataset B AP75：配对值 −0.282318 → **0.282**，论文写 0.283；③ green AP50-95：配对值 +1.529448 → **+1.529**，论文写 +1.530；④ red AP50：配对值 −0.384589 → **−0.385**，论文写 −0.384；⑤ young AP50：配对值 +0.998679 → **+0.999**，论文写 +0.998。反向核对：表 5 的 +2.931 与 Dataset B 的 −0.410 / −1.028 / +0.192 / +1.792 / −4.047 / −4.275 / −1.676 / +0.308 均与配对口径一致，无需改动。 |
| 6 | YOLOv7-tiny mAP50-95 | 源值为 66.341（源：`external_eval/dsA_groupaware_clean_aug_v2__comparison__yolov7_tiny/three_seed_mean_sd.csv`），正文某处写作 66.343%。66.343 实为 SHAB+MaturityDetect 的 seed1 值。 |

---

*本文档由一次性脚本自源文件复算生成（抽取 `build.py` + 排版 `render.py`），生成脚本暂存于会话临时目录、**未纳入仓库**；
如需长期复现，可将其移入 `scripts/`（与 `paper35_extended_statistics.py` 等并列）。*

*复核强度：*

*（1）对 `per_seed.csv` / `three_seed_mean_sd.csv` / `summary.json` / Dataset B `main_results.json` / 各 latency CSV 自动交叉核对 **176 项，0 处不一致**；*
*（2）另做一次**独立复核**——绕开全部中间汇总，直接读取 Common Evaluator 的 `seed{0,1,2}__test_metrics_common.json` 重算 **132 项精度指标（含 Dataset B），逐位一致**；*
*（3）本文档引用的 **34 条结果路径 + 18 条 FPS 来源路径全部存在**。*

*全程只读源文件，未修改任何既有文件。*

---

# 五、与《论文3.13.docx》的对应性核对（2026-09-30 复核）

核对方法：从 `论文3.13.docx` 抽取全部 22 张实验表格文本，与本文档逐格比对。

**结论：22 张表中 19 张完全对应；1 张单项数值需回改（表3 的 SHEB FPS）；2 张附录逐种子表整体错位一行。**

| # | 3.13 中的表 | 对应状态 |
|---|---|---|
| 1 | 正文 表3 多尺度特征增强模块对比 | ⚠️ 8 行的 P/R/mAP 全部对应；但 SHEB 的 FPS 为 122.48±1.80，与本文档统一口径 **122.01±1.17** 不符（见 5.3-1） |
| 2 | 正文 表4 消融 | ✅ 5 行对应；SHEB FPS 为 122.01±1.17，与本文档统一口径一致 |
| 3 | 正文 表5 小目标检测性能 | ✅ 对应 |
| 4 | 正文 表6 定位实验 | ✅ 对应 |
| 5 | 正文 表7 边界框误差分析 | ✅ 对应 |
| 6 | 正文 表8 成熟度类别性能 | ✅ 数值对应；Δ 舍入见 5.3-4 |
| 7 | 正文 3.7 模型综合对比 | ✅ 11 行全部对应 |
| 8 | 正文 3.8 总体表 | ✅ 对应 |
| 9 | 正文 3.8 分类别表 | ✅ 数值对应；Δ 舍入见 5.3-4 |
| 10 | 附录 逐种子—多尺度特征增强模块对比 | ❌ **整体错位一行**（见 5.1） |
| 11 | 附录 逐种子—消融实验 | ❌ **整体错位一行**（见 5.2） |
| 12 | 附录 逐种子—小目标检测性能分析 | ✅ 新增表，数值正确 |
| 13 | 附录 逐种子—定位实验 | ✅ 新增表，数值正确 |
| 14 | 附录 逐种子—边框误差分析 | ✅ 新增表，数值正确 |
| 15 | 附录 逐种子—成熟度类别 | ✅ 新增表，数值正确 |
| 16 | 附录 混淆矩阵（YOLO11n / FSDH-YOLO） | ✅ 对应 |
| 17 | 附录 逐种子—模型综合对比 | ✅ 33 行全部对应（含 FPS，逐 seed 一致） |
| 18 | 附录 逐种子—跨数据集适用性 | ✅ 对应 |
| 19 | 附录 逐种子—YOLO11 模型规模选择 | ✅ 对应 |
| 20 | 附录 逐种子—SPDConv 与 FSDConv 对比补充 | ✅ 对应 |
| 21 | 附录 逐种子—SHEB 分支消融 | ✅ 21 行全部对应 |
| 22 | 附录 B.4 Bootstrap 稳健性分析 | ✅ 已升级为 18 行对照表（含判断列），与本文档一致 |

## 5.1 错误一：附录「逐种子结果—多尺度特征增强模块对比」整表错位一行

Model 标签列比数据列少一行（首个标签 `YOLO11n` 丢失），每个标签携带的是**前一个模型**的数据，**RFB 自身数据整行丢失**：

| 3.13 行标签 | 3.13 行内实际数据属于 | 该行**应为** |
|---|---|---|
| ASPP | YOLO11n | 77.103 / 79.754 / 80.793 / 65.918；FPS 122.46 / 122.35 / 126.22 |
| SPP | ASPP | 78.360 / 79.066 / 80.615 / 66.164；FPS 135.93 / 132.38 / 140.07 |
| DilatedConv | SPP | 76.499 / 80.966 / 81.543 / 66.357；FPS 137.68 / 139.89 / 138.04 |
| RepConv | DilatedConv | 77.828 / 79.842 / 81.065 / 65.703；FPS 143.03 / 148.44 / 139.92 |
| MultiScaleDWConv | RepConv | 78.327 / 79.506 / 80.934 / 66.452；FPS 130.56 / 132.93 / 133.48 |
| RFB | MultiScaleDWConv | 77.499 / 81.230 / 82.507 / 67.348；FPS 124.21 / 124.12 / 123.57 |

（六行数据列整体上移一格；完整逐 seed 值与 FPS 见本文档 1.1 节「逐种子结果」表。）

## 5.2 错误二：附录「逐种子结果—消融实验」整表错位一行

| 3.13 行标签 | 3.13 行内实际数据属于 | 该行**应为** |
|---|---|---|
| YOLO11n-PM | YOLO11n | 79.060 / 78.937 / 80.820 / 66.931；FPS 164.69 / 158.62 / 160.22 |
| FSDConv | YOLO11n-PM | 75.375 / 82.672 / 81.487 / 66.541；FPS 148.49 / 149.34 / 149.14 |
| SHEB | FSDConv | 78.173 / 79.970 / 83.031 / 67.388；FPS 122.57 / 122.80 / 120.67 |
| FSDH-YOLO | SHEB | 78.581 / 78.364 / 80.931 / 66.670；FPS 124.34 / 120.09 / 125.74 |

> **两处成因相同**：从本文档的逐种子表迁移时，Model 标签列的**首行（`YOLO11n`）被删除**且标签列上移一格、数据列未动，以致末行数据溢出丢失。修复方式：在每个表最上方补回 `YOLO11n` 行（其数据即本文档 1.1 / 1.2 节首三行），标签列整体下移一格即可复原。

## 5.3 其余待修（均在正文文字，3.13 未改动）

| # | 位置 | 问题 | 应为 |
|---|---|---|---|
| 1 | 表3 表格 | SHEB FPS 现为 122.48±1.80（取自 `..._fp32_b1_640` 批次） | 改回 **122.01±1.17**，与表4、附录 B.3、本文档统一 |
| 2 | 表3 正文 | 「…FPS仍达到122.01」 | ✅ 无需修改；表格改回后即与正文一致 |
| 3 | 表4 表格 + 正文 | 表格 122.01±1.17、「FPS降至122.01」 | ✅ 无需修改，已是统一口径 |
| 4 | 3.7 正文 | 「YOLOv7-tiny…其mAP50-95为66.343%」 | 66.341%（表格已正确） |
| 5 | 3.8 正文 | 「AP75…降低0.283 pp」 | 0.282 pp（配对口径 −0.282318） |
| 6 | 3.8 正文 | 「如表 X 所示」占位符未填 | 补表号 |
| 7 | 表8 ripe AP50-95 Δ | +4.739 | +4.740（配对口径 +4.7396095） |
| 8 | 3.8 分类别 Δ | green AP50-95 +1.530、red AP50 −0.384、young AP50 +0.998 | +1.529、−0.385、+0.999（配对口径） |

> 说明：3.8 的 Δ 与表8 的 Δ 在 3.13 中采用了「先各自四舍五入再相减」，与源文件的**配对统计口径**相差 0.001；两种写法都能自圆其说，但同一篇论文内应统一。若采用配对口径，上表最后两行（表8 ripe Δ、3.8 分类别 Δ）需一并调整；反之则本文档 1.6 / 1.8 节的 Δ 需相应改写。

> **SHEB FPS 的统一口径（2026-09-30 定）**：全篇 SHEB 一律取 `122.01±1.17`（逐 seed 122.57 / 122.80 / 120.67，来源 `runs/quality/paper16_non_fsd_uniform_latency`），涉及**正文表 3、正文表 4、正文两处文字、附录 B.3、附录逐种子—消融表、本文档 1.1 / 1.2 / B.3 节**，共 6 处须一致。表 3 其余 6 个模块仍取 `..._fp32_b1_640` 批次，两批为同机不同时间测速，该批中 SHEB 为 `122.48±1.80`（仅供参考，不采用）。
