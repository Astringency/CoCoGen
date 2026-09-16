# 正式评估：同样本图像对照

已生成 **20 页真值 / FM4PDE / CoCoGen 对照图**，覆盖 Darcy、Poisson、Helmholtz、NS 各五个 ID 任务。图中是各单元按编号取的第一个正式样本，不能作为 1,000 例平均成绩；完整成绩请看各 PDE 报告。

## 查看图像

- [20 页完整 PDF](execution_20260915/main_examples_57/first_id_main_comparisons.pdf)
- Darcy：[全观测正问题](execution_20260915/main_examples_57/darcy_id_full_forward_sample_0.png)、[稀疏联合重建](execution_20260915/main_examples_57/darcy_id_sparse_joint_sample_0.png)。
- Poisson：[稀疏反问题](execution_20260915/main_examples_57/poisson_id_sparse_inverse_sample_0.png)。
- Helmholtz：[稀疏正问题](execution_20260915/main_examples_57/helmholtz_id_sparse_forward_sample_0.png)、[全观测反问题](execution_20260915/main_examples_57/helmholtz_id_full_inverse_sample_0.png)。
- NS：[完整正问题](execution_20260915/main_examples_57/nsnonbounded_id_full_forward_sample_2000.png)、[完整反问题](execution_20260915/main_examples_57/nsnonbounded_id_full_inverse_sample_2000.png)、[稀疏正问题](execution_20260915/main_examples_57/nsnonbounded_id_sparse_forward_sample_2000.png)、[稀疏反问题](execution_20260915/main_examples_57/nsnonbounded_id_sparse_inverse_sample_2000.png)、[稀疏联合重建](execution_20260915/main_examples_57/nsnonbounded_id_sparse_joint_sample_2000.png)。

范围绑定固定的 **57 单元快照**，其中四个 PDE 的全部 ID 五任务已经齐备。前三者样本编号为 0，NS 为 2000；没有按重建质量挑选。此图集不覆盖 rough/smooth 或 Burgers。

## 图中可以看出什么

在 Darcy 第 0 例中，全观测正问题的 CoCoGen 解场误差为 0.67%，稀疏联合任务的系数场误差为 62.96%，后者可见区域边界偏移和零散斑点。Poisson 的稀疏反问题也有明显源项场结构偏差，单例误差为 97.92%。这些示例说明精确满足观测点并不足以保证未观测区域的重建质量；不据此推断所有样本或单一失败原因。

补齐的 NS 样本 2000 中，完整正问题的末态场较接近真值；稀疏正问题的末态场和稀疏反问题的初始场可见幅值偏弱、结构细节缺失。联合任务仍有局部结构偏差和观测点附近的斑点。以下全部是这个**单一样本**的相对 L2，不是正式均值：

| NS 任务与目标场 | CoCoGen | FM4PDE |
|---|---:|---:|
| 完整正问题，u | 3.04% | 8.44% |
| 完整反问题，a | 10.16% | 8.10% |
| 稀疏正问题，u | 51.71% | 9.17% |
| 稀疏反问题，a | 66.12% | 8.72% |
| 稀疏联合，a | 31.95% | 13.47% |
| 稀疏联合，u | 25.32% | 6.85% |

这组图对应冻结的 **100 步 / RePaint=4 / 500 NFE 正式配置**。用户要求的 2,000 步对照使用另一组校准样本，结果另见 [Darcy 长采样诊断](darcy_long_sampling_diagnostic_20260916.md)。两组图的样本编号不同，不能逐例相互比较。

## 显示与验证

每行依次显示有效观测、真值、历史 FM4PDE 预测及 CoCoGen 预测，使用原始物理量。同一 PDE、同一字段在其全部五页上共享色标，最近邻显示。正问题只对 u 标误差，反问题只对 a 标误差，联合任务对两者标误差；非目标场明确标为不计入任务成绩。

全部 20 例均通过来源哈希、样本编号、形状、有限值、CoCoGen 观测一致性与两模型逐例误差核验。FM4PDE 的冻结指标先以 float32 做差、再以 float64 求范数；CoCoGen 指标先转换为 float64 再做差。绘图校验分别复现各自保存指标的计算顺序，未修改既有正式指标或采样。

`pdfinfo` 实测为 20 页。本次实际查看了全部五张 NS PNG；此前已查看的五张 Darcy/Poisson/Helmholtz PNG 与新版本逐字节相同，原视觉检查继续有效。其余页面有来源与数值检查，未逐页目视检查全部 PDF。原始来源记录为 [provenance.json](execution_20260915/main_examples_57/provenance.json)，视觉检查另存 [visual_qa.json](execution_20260915/main_examples_57/visual_qa.json)。

与旧图集相比，Darcy/Poisson/Helmholtz 的 15 张 PNG 全部逐字节相同；先前 16 例的样本、误差和预测来源哈希均未变化。NS 的 a 色标范围扩大，以包含新增任务的场值；因此原完整正问题页也重新显示并目视检查。五页 NS 使用同一新色标，没有针对某一预测单独调整。

绘图源码仍为 `12c330b7847802fe3c2e048c2d4608634d38e3ee`，本次任务于 06:31:21–06:31:51 CST 正常退出，退出码 0。任务只读取保存预测，不运行模型采样。

## 存储与历史版本

197 主目录下的 `reports/figures_id_first_57/` 保存全部图像及来源记录：

```text
/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915/reports/figures_id_first_57/
```

绑定快照 SHA256：`bd6b8cc5b93f7cc32bf85014e82a110b08758ba5e39dc9e2c6ee0f45b0dd4760`；PDF SHA256：`8f0547e627cc4758c98196ed95bcd595c6e44e8abd53128407d802549bcd5262`。

此前绑定 47 单元快照的 [16 页 PDF](execution_20260915/main_examples_47/first_id_main_comparisons.pdf)、PNG、来源与视觉检查记录独立保留在 `main_examples_47/`，服务器对应目录为 `reports/figures_id_first_47/`。首轮因 FM4PDE 指标浮点计算顺序而在输出图像前退出的日志也仍保留。

## Burgers 对照准备

已另行生成并查看 [Burgers 六个固定样本的 FM4PDE 参考图](burger_example_preparation_20260916.md)，明确时空轴及两种有效观测设置。该图集尚不包含 CoCoGen 预测；真实训练、正式采样完成后，再运行专用入口生成和检查四列模型对照，不计入上述已完成的 20 个四模型示例。
