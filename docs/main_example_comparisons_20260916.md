# 正式评估：同样本图像对照

已生成 **16 页真值 / FM4PDE / CoCoGen 对照图**，用于查看生成场的结构偏差。图中是每个已完成 ID 单元的第一个正式样本，不能作为 1,000 例平均成绩；完整成绩请看各 PDE 报告。

## 查看图像

- [16 页完整 PDF](execution_20260915/main_examples_47/first_id_main_comparisons.pdf)
- Darcy：[全观测正问题](execution_20260915/main_examples_47/darcy_id_full_forward_sample_0.png)、[稀疏联合重建](execution_20260915/main_examples_47/darcy_id_sparse_joint_sample_0.png)。
- Poisson：[稀疏反问题](execution_20260915/main_examples_47/poisson_id_sparse_inverse_sample_0.png)。
- Helmholtz：[稀疏正问题](execution_20260915/main_examples_47/helmholtz_id_sparse_forward_sample_0.png)、[全观测反问题](execution_20260915/main_examples_47/helmholtz_id_full_inverse_sample_0.png)。
- Navier–Stokes：[全观测正问题](execution_20260915/main_examples_47/nsnonbounded_id_full_forward_sample_2000.png)。

范围绑定固定的 **47 单元快照**：Darcy、Poisson、Helmholtz 各五个 ID 任务，NS 仅 ID 全观测正问题。前三者样本编号为 0，NS 为 2000；按编号取第一例，没有按重建质量挑选。原始 PNG 均保留在上述目录中。

## 图中可以看出什么

在 Darcy 第 0 例中，全观测正问题的 CoCoGen 解场误差为 0.67%，稀疏联合任务的系数场误差为 62.96%，后者可见区域边界偏移和零散斑点。Poisson 的稀疏反问题也有明显源项场结构偏差，单例误差为 97.92%。这些示例说明精确满足观测点并不足以保证未观测区域的重建质量；不据此推断所有样本或单一失败原因。

这组图对应冻结的 **100 步 / RePaint=4 / 500 NFE 正式配置**。用户要求的 2,000 步对照使用另一组校准样本，结果另见 [Darcy 长采样诊断](darcy_long_sampling_diagnostic_20260916.md)。两组图的样本编号不同，不能逐例相互比较。

## 显示与验证

每行依次显示有效观测、真值、历史 FM4PDE 预测及 CoCoGen 预测，使用原始物理量。同一 PDE、同一字段在其全部页面上共享色标，最近邻显示。正问题只对 u 标误差，反问题只对 a 标误差，联合任务对两者标误差；非目标场明确标为不计入任务成绩。

全部 16 例均通过来源哈希、样本编号、形状、有限值、CoCoGen 观测一致性与两模型逐例误差核验。FM4PDE 的冻结指标先以 float32 做差、再以 float64 求范数；CoCoGen 指标先转换为 float64 再做差。绘图校验分别复现各自保存指标的计算顺序，未修改既有正式指标或采样。

PDF 实测为 16 页；上述六张 PNG 已实际查看，覆盖四个 PDE 和五种任务，标题、观测数量、字段含义、色标和脚注可读。未逐页目视检查全部 PDF。原始来源记录保留为 [provenance.json](execution_20260915/main_examples_47/provenance.json)，视觉检查另存 [visual_qa.json](execution_20260915/main_examples_47/visual_qa.json)。

绘图源码提交为 `12c330b7847802fe3c2e048c2d4608634d38e3ee`。首轮绘图因历史 FM4PDE 指标计算顺序的末位差异而在创建图像前退出；保留失败日志和退出码，修正后的独立任务正常退出。两个任务都只读取保存预测，不运行模型采样。

## 存储

197 主目录下的 `reports/figures_id_first_47/` 保存全部图像及来源记录：

```text
/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915/reports/figures_id_first_47/
```

绑定快照 SHA256：`c156ad2f3e26f481d09d11927e2aff4664e34b6da812bfe915d703910b25a12b`；PDF SHA256：`2f6ed61125ff9911bce5be9c99545e3f4cf77c4c981f088106adf71f573209cb`。
