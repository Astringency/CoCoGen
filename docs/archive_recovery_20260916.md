# CoCoGen 归档内复核与迁移恢复

## 当前已验证范围

截至 2026-09-16 07:14 CST，**四模型 60 个正式单元、60,000 次完整单元样本预测**均已完成并通过归档内独立数值复核。覆盖来自早期 46 单元审计与后续新增单元审计的联合，绑定完整快照及全部 1,920 份预测哈希；不是声称早期 46 单元任务单次检查了 60 单元。完整记录见 [validation_report_watch_60cells.json](execution_20260915/validation_report_watch_60cells.json)。

这些检查确认了已报告重建误差的可重算性。四模型报告已完成，Burgers 已于 07:18 启动训练；Burgers 六单元评估、最终五 PDE 报告及全量迁移恢复仍需完成。下面的迁移实测仍只覆盖两个单元，不扩大其声明范围。

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

## 恢复后重新核验历史 46 单元快照

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

## Burgers 训练归档的后续验证范围

2026-09-16 06:10 CST 的源码检查确认，预测读取器可以选择全部 66 个单元，迁移工具会复制所选审计实际读取的文件。不过，这个文件集合由预测复核决定，**不包含完整训练恢复所需的全部数据与状态**。Burgers 尚在等待第一阶段完成，真实训练迁移检查目前不能执行。

`cocogen_burger/data.py` 的训练缓存清单保存 `training_path` 和 `validation_path` 的绝对路径。`require_first60` 还会读取第一阶段完成收据中的 CSV、单元摘要及逐批预测绝对路径。`cocogen_burger/train.py` 会校验缓存清单哈希，并要求恢复检查点中的训练请求与当前请求一致。因此，移动目录后简单改写缓存 JSON 会改变已绑定的哈希；不能把预测读取器通过当作整个训练入口可直接在新路径恢复的证据。

最终归档需保留并用真实产物核验：

- 50,000 例标准化训练缓存、独立检查点验证面板、归一化参数及原始清单。
- best/last 检查点、模型/训练配置、训练请求、历史日志、优化器和各 rank 随机状态。
- 训练前置条件引用的第一阶段收据与文件，以及实际训练源码和环境记录。
- 新根目录中对历史绝对路径的明确映射，保留原始文件内容与哈希，并禁止恢复检查回退读取原目录。

真实检查点尚未生成，本段只记录已经确认的路径依赖和后续验证要求，不声称训练恢复通过，不改变正在等待的 Burgers 源码或执行顺序。届时应将训练产物的恢复证据与全部 66 项预测恢复证据分别保存，再判断完整归档是否可独立使用。

## 真实训练检查点初查（2026-09-16 07:32 CST）

上节 06:10 的等待状态是历史记录。Burgers 已开始真实训练；第 2 轮的恢复检查点在独立 CPU 进程中通过只读检查：128 个模型张量严格加载，17,653,505 个参数有限，86 组 AdamW 状态完整、形状匹配、步数均为 1,564，两个 rank 的 Python/NumPy/CPU/CUDA 随机状态均存在。Python、NumPy 和 CPU 随机状态已实际载入，CUDA 随机状态只检查保存格式。

检查点与其第 2 轮记录、训练请求、缓存清单、归一化及三份训练源文件标识相符。检查从同一个打开的文件描述符计算哈希并加载，避免训练原子替换 last.ckpt 时混入不同轮次。对应检查点 SHA256 为 `5c36e1c2295c68d5cadd8785aa1cd06ac2f2dbd4065d4bde9347f2459780949f`。

记录为 [validation_burger_checkpoint_initial.json](execution_20260915/validation_burger_checkpoint_initial.json)，审计源码提交 `da51d46be2f81135782ec81d73e4ae8dff9023ad`；任务正常退出，未改变正在训练的进程。last.ckpt 仍按轮次滚动更新，本记录描述读取当时的第 2 轮状态，不把该动态路径当作永久不变的 checkpoint。

该检查不执行训练更新、不重新哈希完整训练缓存、不验证 CUDA 续训或新目录迁移，也不证明收敛。最终检查仍需使用训练完成后的真实 best/last 及全部训练资产。
