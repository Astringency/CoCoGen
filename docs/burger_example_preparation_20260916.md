# Burgers 图像对照准备

已完成六个固定样本的 **FM4PDE 参考图**，覆盖 ID、smooth、rough 的随机 500 点和五个固定空间传感器两种设置。**这些图尚不包含 CoCoGen 预测，不能作为 Burgers 的 CoCoGen 评估结果。**

- [六页参考图 PDF](execution_20260915/burger_reference_examples/burger_reference_examples.pdf)
- [ID：随机 500 点](execution_20260915/burger_reference_examples/burger_id_random_sample_0.png) · [ID：五个空间传感器](execution_20260915/burger_reference_examples/burger_id_sensor_column_sample_0.png)
- [smooth：随机 500 点](execution_20260915/burger_reference_examples/burger_smooth_random_sample_0.png) · [smooth：五个空间传感器](execution_20260915/burger_reference_examples/burger_smooth_sensor_column_sample_0.png)
- [rough：随机 500 点](execution_20260915/burger_reference_examples/burger_rough_random_sample_0.png) · [rough：五个空间传感器](execution_20260915/burger_reference_examples/burger_rough_sensor_column_sample_0.png)

## 固定的展示口径

每个单元选择正式编号 **0**，在看到 CoCoGen 结果之前固定，没有按重建质量挑选。每张图展示一个样本，不代表 1,000 例平均成绩；同一分布的两种观测设置复用同一样本。

横轴为空间索引，纵轴为时间索引，时间由下向上增加。图中使用物理场值、最近邻显示，六页共享同一对称色标。由于旧 ID 数据没有完整物理时间元数据，坐标采用索引，不把时间常数标成由原文件测得。

只应用归档中实际生效的解场观测 mask。随机设置为 500 个时空点；空间传感器位于索引 **20、22、37、103、121**，每个位置观测全部 128 个时刻，共 640 个值。系数候选 mask 的观测权重为零，不加入图中。完整轴向核查另见 [Burgers 协议](burgers_training_protocol_20260915.md)。

## 本次实际验证

CPU 任务于 **2026-09-16 08:56:24–08:56:30 CST** 正常完成，退出码 0。六个样本的真实归档来源、真值别名一致性、有效观测形状及数量均通过检查；FM4PDE 每例误差按冻结口径重算，与各自历史记录一致。本次只核验六个展示样本，不重算全部 6,000 个 FM4PDE 样本。

六张 PNG 已全部实际查看，标题、坐标轴、观测点和共享色标可读，没有发现文本裁切或重叠。PDF 实际页数为 6；未另外逐页栅格化检查 PDF。输出哈希、来源与视觉检查分别保存在 [provenance.json](execution_20260915/burger_reference_examples/provenance.json) 和 [visual_qa.json](execution_20260915/burger_reference_examples/visual_qa.json)。

197 保存目录：

```text
/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915/reports/figures_burger_reference_20260916/
```

## 正式 CoCoGen 对照仍待生成

绘图入口为 `cocogen_eval/plot_burger_case_comparison.py`，冻结源码提交 `06f35764ae01a84203dc1ea32f724368ab2bffa1`，归档 bundle 为 `source/burger_figures_code.bundle`。

正式模式要求真实完成的 `all66` 指标和六个 Burgers 单元，读取其预测与收据；按实际 NFE 标注预算，核验展示样本的相对 L2 及观测点一致性。正式预测尚未产生，所以本次只实际执行了参考图模式，**尚未验证真实 CoCoGen 图像读取与四列排版**。不能用上述参考图检查代替正式模式的后续运行及视觉检查。

真实结果齐备后，从该冻结检出运行下列命令，并使用新的输出目录：

```bash
study=/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915
CUDA_VISIBLE_DEVICES="" python -m cocogen_eval.plot_burger_case_comparison \
  --root "$study" --original-study "$study" \
  --metrics reports/metrics_all66.json \
  --out "$study/reports/figures_burger_complete"
```

绘图准备没有启动 CoCoGen 采样、改变训练设置或修改冻结的正式评估配置。训练仍继续运行；整个五 PDE 任务尚未完成。
