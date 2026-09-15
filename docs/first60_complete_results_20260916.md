# 四模型 60 单元：完整结果

**四模型 60 个单元、共 60,000 次完整单元样本评估已完成。CoCoGen 的 PDE 等权平均相对 L2 为 39.19%，FM4PDE 为 19.40%；CoCoGen 在 11/60 个单元上平均误差更低。** 相对 L2 越低越好。

| PDE | CoCoGen | FM4PDE | CoCoGen 均值更低单元 |
|---|---:|---:|---:|
| Darcy | 33.61% | 16.07% | 3/15 |
| Poisson | 40.34% | 20.75% | 2/15 |
| Helmholtz | 42.61% | 23.47% | 4/15 |
| NS | 40.22% | 17.29% | 2/15 |

## 主要结论

- 优势主要集中在完整观测正问题：12 个对应单元中，CoCoGen 有 10 个平均误差更低。
- 全部 12 个完整反问题平均误差均更高。36 个稀疏单元中，只有 Helmholtz rough 联合重建的目标场平均误差略低，而且该单元的解场 u 仍更差。
- 有效观测点全部精确满足，但未观测区域仍有明显重建偏差。观测约束生效不等于完整场准确。
- 这是现有权重和冻结适配的比较，不能视为原论文全部设置的复现，也不能单独归因于训练或步数。单种子、跨任务复用样本，不提供多种子置信区间。

## 实测耗时

- 调度开始：**2026-09-15 11:51:55 CST**；60 单元及原始汇总完成：**2026-09-16 07:13:56 CST**。
- 墙钟 **19.367 小时，约 19 小时 22 分钟**，满足约一天的四模型评估预算。
- 两张 A100 累计纯采样 **38.544 GPU 小时**，是各 GPU 时间之和。
- 独立结果复核于 **07:14:23 CST** 完成。上述墙钟不含此前校准、后续 Burgers 补训及最终归档。

## 计算与复核口径

- 每例在物理空间计算完整目标场相对 L2，再对 1,000 例取均值。正问题只评价 u，反问题只评价 a，联合任务先对 a/u 等权；之后对单元和 PDE 等权。
- 使用相同 FM4PDE 归档编号和有效掩码：Darcy/Poisson/Helmholtz 为 0–999，NS 为 2000–2999。四模型均冻结为 100 步、RePaint 4 次、500 NFE，未从正式成绩重新调参。
- 60 个单元均已有独立 NumPy float64 误差复算，覆盖 1,920 份预测；旧审计与新增审计通过文件哈希及每场结果关联到完整快照。
- 完整收集模式、60 单元完成记录及原始 CSV 均已通过检查；制表脚本另核对字段、单元、PDE、总体各级权重。

## 查看结果

- [全部任务、分布和目标场的表格](execution_20260915/comparison_first60/comparison.md)
- [完整指标 JSON](execution_20260915/metrics_first60.json) · [目标场 CSV](execution_20260915/metrics_first60_targets.csv)
- [Darcy](darcy_complete_results_20260915.md) · [Poisson](poisson_complete_results_20260916.md) · [Helmholtz](helmholtz_complete_results_20260916.md) · [NS](ns_complete_results_20260916.md)
- [20 个固定 ID 示例的图像说明](main_example_comparisons_20260916.md)
- [Darcy 2,000 步 / 增大 RePaint 的独立诊断](darcy_long_sampling_diagnostic_20260916.md)：小幅改善，成本约 38 倍；不改变正式配置。

197 主结果目录：

```text
/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915/
```

详细比较位于 `reports/comparison_first60/`；完整指标为 `reports/metrics_first60.json` 和 `reports/metrics_first60_targets.csv`。

## 后续阶段

Burgers 于 **2026-09-16 07:14 CST** 通过全部前 60 项结果的检查，开始数据准备，07:18 已完成显存测速并启动双 A100 训练，每卡 batch 32。之后还需独立采样校准和六个 1,000 例单元评估。此时尚未取得 Burgers 训练完成或正式质量结果，不能将四模型完成称为五 PDE 全部完成。
