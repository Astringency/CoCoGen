# CoCoGen 归档内复核与迁移恢复

## 当前已验证范围

截至 2026-09-16 02:33 CST，已完成的 **46 个正式单元、46,000 次预测**可以完全依靠主归档中的预测、输入副本、配置和权重标识重新核验。覆盖 Darcy、Poisson、Helmholtz 各 15 个单元，以及 NS 的 rough 稀疏正问题。

该检查确认了已报告重建误差的可重算性。它不代表全部实验完成；剩余 NS、Burgers 训练和六单元评估、完整报告及最终归档仍需完成。

## 为什么需要专门的恢复入口

历史 JSON 为保留来源，包含原服务器的绝对路径。`verify_archived_predictions.py` 保持这些文件内容不变，只在读取时映射路径：

- 原主结果目录内的路径映射到当前归档根目录下的相对位置。
- 外部 FM4PDE 结果、既有模型和配置，使用 `archive/dependencies/manifest.json` 的路径到归档对象映射。
- 未登记的外部路径、越出归档根目录的路径和外部链接不会被用作回退来源。

输入加载器接收内存中的路径映射副本，批次路径和逐例成员路径同时转换；有效掩码仍应用原来冻结的任务规则。Burgers 的单通道数据分支保留，但真实 Burgers 预测尚未产生，当前不声称已验证该分支的完整结果恢复。

## 实际验证

### 46 个已完成单元

2026-09-16 **02:30:45–02:32:28 CST**，CPU 复核正常结束，退出码 0。读取并核验 **4,976 个归档文件，24,869,469,966 字节**，检查包括：

- 样本 ID、原始输入、有效观测掩码与保存的采样请求一致。
- 归一化配置、已选权重和模型配置文件哈希、采样实现及评估源文件标识一致。
- 1,472 份预测文件及其收据一致，实际 NFE 与终点时间符合冻结配置。
- 全部有效观测点精确一致；NumPy float64 重新计算每例完整场相对 L2，与保存误差匹配。
- 目标场均值、与 FM4PDE 的配对差、误差比、逐例胜率及采样时间与固定快照一致。

FM4PDE 参照误差使用已冻结的历史记录；本入口没有重新计算其原始模型预测，也没有重新计算 PDE 残差或训练损失。检查仅使用 CPU，四个线程，低 CPU/I/O 优先级；同时运行的两张 A100 继续正式采样。

证据：`validation/archive_reader_46cells.json`。本地副本为 `docs/execution_20260915/validation_archive_reader_46cells.json`，SHA256：

```text
e5a0da22d55759835dd9bbbce8c8d4db7a43e231abb18f5d784021997ec569c4
```

该记录绑定独立快照 `reports/snapshots/metrics_first60_46cells.json`，快照 SHA256 为 `b750e6066d1bd11024936e053ac7a595fda291524c62b9a423a9af5f999b0b3f`，不会随着最新局部结果更新而改变。

### 独立目录迁移

另外选取 Darcy ID 稀疏联合重建与 NS rough 稀疏正问题，共 **2,000 例**，覆盖两种不同的历史输入布局。

2026-09-16 **02:33:08–02:33:20 CST**，将所需 **224 个文件、1,335,178,039 字节**复制到新的临时目录，使用独立普通文件副本；从归档 Git bundle 恢复新的源码仓库，检查 `git fsck --full` 与无 alternates 依赖。

恢复子进程安装 Python `open` 审计钩子，拒绝访问旧主结果目录和依赖清单中的原文件。两个主动读取旧路径的探测均被拒绝，之后实际复核没有尝试访问旧路径。重算结果和读取文件哈希与原复核逐项一致。Python 环境与系统库仍使用本机安装，此检查不是操作系统级文件访问沙箱或环境镜像恢复。

该迁移测试严格覆盖上述两个单元；46 个单元的完整数据复核是在主归档目录中进行。临时数据副本和恢复仓库已清理，三个验证任务均退出码 0，对应 tmux 会话已自动结束。

证据：`validation/archive_relocation_two_cells.json`，本地副本 SHA256 为 `1f36cd480da3f42d52fc636a12cd0ca80b3f148e70f5974c13996003bdc9ea5c`。其输入为 `validation/archive_reader_two_cells.json`，原始日志和退出码保存在主目录的 `logs/archive_reader_probe.*`、`logs/archive_reader_46.*`、`logs/archive_relocation_probe.*`。

## 恢复后重新核验当前快照

主归档目录为：

```text
/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915/
```

将完整主目录迁移到新位置后，在已有相容 Python/PyTorch/NumPy 环境中，使用下列入口。先把 `recovered_study` 改成新位置；`--original-study` 保留原路径，用于解释历史记录。

```bash
recovered_study=/path/to/cocogen_main_20260915
git init recovered-code
git -C recovered-code fetch "$recovered_study/source/archive_reader_code.bundle" HEAD
git -C recovered-code checkout --detach 5094909b66d22a78ded57dd3a6f161ae51379303
cd recovered-code
CUDA_VISIBLE_DEVICES="" python -m cocogen_eval.verify_archived_predictions \
  --root "$recovered_study" \
  --original-study /research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915 \
  --metrics reports/snapshots/metrics_first60_46cells.json \
  --allow-partial \
  --output "$recovered_study/validation/recovered_46cells_new.json"
```

输出路径必须是新文件。`--allow-partial` 明确声明这是未完成的第一阶段快照。最终 60/66 单元的完整快照应在真实结果齐备后分别运行，不用当前阶段性记录代替。

读取器源码固定提交为 `5094909b66d22a78ded57dd3a6f161ae51379303`，迁移验证器提交为 `85f481e2d1af6146eef7b783c88515b0597ca56e`。源代码分别归档于 `source/archive_reader_code.bundle`、`source/archive_relocation_code.bundle`。两个任务专用检出目录留待最终统一清理，运行中的正式采样与 Burgers 代码版本保持冻结。
