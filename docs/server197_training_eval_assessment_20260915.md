# CoCoGen 使用 server197 数据训练并对齐 FM4PDE 主实验

核查日期：2026-09-15。核查范围：本地 CoCoGen、相邻 FM4PDE 仓库、server197 的代码、数据文件元信息、历史训练日志和 checkpoint 元信息。未启动训练、GPU 性能测试或正式采样，未修改服务器文件。

## 结论

可以执行。建议直接在 server197 使用已有 PDEdata，并在本地完成适配后通过 Git 同步到独立检出目录。当前本地版本不能仅修改数据路径就用于正式比较。

主论文范围是五个 PDE 的 66 个评估单元，而不是 FM4PDE 当前框架支持的全部 11 个 PDE。每个 PDE 训练一个联合先验，通常可用于该 PDE 的所有方向和观测任务，不需要按 66 个单元训练 66 个模型。

优先做一个 Poisson 小规模端到端验证，然后再决定：从头训练五个模型，或核查并复用服务器现有四个模型、补训 Burgers。旧权重的复用需要恢复训练归一化并验证数据身份；现阶段不能保证它们的最终精度或完整训练来源可追溯。

## 1. 服务器和已有材料

访问指南实际位于 `/home/tat512/share/SSH_REMOTE_WORK.md`，用户给出的 `/home/tats512/...` 在本机不存在。

2026-09-15 09:31 CST 只读资源检查：

| 项目 | 结果 |
| --- | --- |
| SSH | `server197`，用户 `zhangxifeng` |
| GPU | 2 × A100-SXM4-80GB，各占用 14 MiB，利用率 0%，无计算进程 |
| CPU / RAM | load average 0.25 / 0.31 / 0.29，可用内存约 924 GiB |
| 存储 | `/research_data` 可用约 3.8 TiB，`/large_storage` 可用约 16 TiB |
| 数据根目录 | `/large_storage/zhangxf/PDEdata/` |
| FM4PDE | `/research_data/users/zhangxifeng/C01Python/FM4PDE/` |
| 旧 CoCoGen | `/research_data/users/zhangxifeng/C01Python/CoCoGen/`；该目录没有 Git 仓库 |
| 已有环境 | `/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python` |
| 环境元信息 | torch 2.8.0、torchvision 0.23.0、pytorch-lightning 2.5.5、OmegaConf 2.3.0、h5py 3.15.0 |

这是检查时的快照，正式运行前需重新检查资源。环境这里只核查了包元信息，尚未完成 CoCoGen 的端到端兼容性测试。本地 environment.yml 锁定的是 Python 3.11 / torch 2.0.1 / Lightning 2.0.6，且包含作者机器的 prefix，不宜原样覆盖服务器现有环境。

旧 CoCoGen 有 `data/load.py`、128 分辨率网络修正和额外物理算子，可以作为适配参考。`output/` 约 9.3 GiB，包含基础模型和 ControlNet 阶段权重。四个基础模型的 `last.ckpt` 经 CPU 读取确认：

| PDE | output 下的目录 | checkpoint epoch（从 0 开始） | global_step |
| --- | --- | ---: | ---: |
| Poisson | `2025-11-14T18-14-05_cocogen4poisson` | 1624 | 1,270,750 |
| Darcy | `2025-11-15T17-57-32_cocogen4darcy` | 1129 | 883,660 |
| Helmholtz | `2025-11-15T18-01-40_cocogen4helmholtz` | 1184 | 926,670 |
| NS | `2025-11-15T18-08-44_cocogen4nsnonbounded` | 1184 | 926,670 |

它们有模型与优化器状态，但顶层没有保存 normalizer。旧训练代码在全训练数据上计算每通道均值和标准差，需按原规则恢复；文件路径相同不能证明文件内容从训练以来没有变化。现有旧目录中未发现 Burgers 模型。

## 2. 数据接口

五个 PDE 的五个训练分片以及 ID / Smooth / Rough 测试文件均存在。每个训练分片标称并在首分片形状核查中确认 10,000 条，完整训练池为每 PDE 50,000 条。当前检查验证了分片清单和代表文件的结构，没有扫描全部数值检查 NaN、重复或重算全文件哈希。

| PDE | 子目录 / 文件模式 | 数据字段与语义 |
| --- | --- | --- |
| Darcy | `darcy/darcy_10000-128-128_{1..5}.mat` | HDF5 `thresh_a_data`, `thresh_p_data`，原始 `[128,128,N]`；转为 `[N,2,128,128]` |
| Poisson | `poisson/poisson_10000-128-128_{1..5}.mat` | MAT v5 `f_data`, `phi_data`，分别 `[N,128,128]` |
| Helmholtz | `helmholtz/helmholtz_10000-128-128_{1..5}.mat` | MAT v5 `f_data`, `psi_data` |
| NS | `nsnonbounded/nsnonbounded_10000-128-128-10_{1..5}_new.mat` | HDF5 `w0`, `w[..., -1]`，联合初始与最终涡度；时间/黏性等为元信息 |
| Burgers | `burgers/burger_10000-128-128_{1..5}.mat` | MAT v5 `output`，模型状态 `[N,1,T,X]`；注意目录是 `burgers` |

优先复用或移植 FM4PDE 的 `data/load.py`、`data/specs.py`、`data/training_manifest.py` 的输入约定，并接一个 Lightning DataModule。明确使用 `configs/training_data.yaml` 的分片顺序，划出固定训练验证集；验证集不参与最终训练归一化拟合。若要复现某个旧模型的训练预算，则按它的归档训练记录确定拆分，不能仅凭当前启动脚本推断。

无需将所有原始数据搬到本地或重新生成。单个两通道 50,000 × 128 × 128 float32 训练张量约 6.10 GiB，Burgers 约 3.05 GiB；四个两通道模型加 Burgers 的有效张量总量约 27.5 GiB。NS 五个原文件约 100.7 GiB，包含额外速度场和时间帧，训练只取需要的字段及端点，避免把完整轨迹和无用字段复制进每个进程。

## 3. 必须完成的代码适配

### 数据、通道与归一化

- 本地 `dataloaders/loaders.py` 读取旧的逐样本 NPY 目录；Darcy 顺序为 `[p,k]`，FM4PDE 为 `[a,u]`。建议统一使用后者，并同步修改观测、物理算子和指标，避免通道颠倒。
- 新训练采用训练集逐通道标准化，将 normalizer、数据清单、轴语义和模型配置存入 checkpoint。采样时先用同一个 normalizer 标准化观测，在物理空间评估残差与误差。
- 服务器旧 `sample.py` 调用 `data/load_test.py` 后，对整套测试数据重新构造 `PDEtransform(data)`。这会使用测试集统计，必须替换为训练统计。
- 若复用旧权重，必须保留它原有的归一化规则，而不是强行套上新模型的统计。

### 128 分辨率网络

本地 `models/unets.py:186` 采用 `data_size // (4*pool_size)` 作为转置卷积的 kernel/stride。128 分辨率、pool_size=4 时，跳连的 32×32 与解码器的 64×64 不匹配。服务器旧版已改为 `kernel_size=pool_size, stride=pool_size`，可核对后移植。已有基础 checkpoint 的对应权重确认为 `[512,512,4,4]`。

### 基础模型训练

- 从头训练第一阶段应关闭 ControlNet，并设 `lightning.lock_base: false`。本地 Darcy 默认 `controlnet.use: true` 和 `lock_base: true` 是冻结基础网络的配置，不能拿它训练随机初始化的基础模型。
- Burgers 配置缺少 `lock_base`，入口却直接访问该字段；应提供安全默认值。
- 对五个 PDE 分别训练 score-matching 联合先验。建议以 300 轮作为第一阶段预算检查点，保存中间 checkpoint，用训练验证集的生成/条件恢复指标决定是否继续。300 轮不是 CoCoGen 已收敛的保证；历史模型实际训练超过 1000 轮。
- 如果确实需要 ControlNet，再作为加载基础模型后的独立第二阶段评估。不要把同一 PDE 的常数标签当成有效物理条件。完整第二阶段训练耗时不包含在下面的基础模型预算内。
- 补充 max_steps、梯度累积、训练数据 shuffle、验证与可选 ImageLogger 配置。现有 train.py 只显式传递少数 Trainer 参数；仅在 YAML 中添加 max_steps/accumulate_grad_batches 不会自动生效。
- 性能探测时显式禁用 ImageLogger。它默认还会在 1、2、4、8 等早期步采样，单纯把 sample_every_n_steps 调大不能消除这些开销。
- A100 可测试 `precision="bf16-mixed"`；数值比较时记录训练精度、推理精度及 TF32，不能假设它与原始 float32 运行完全等价。

Lightning 参数已按 Context7 核对：[Trainer](https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/trainer.rst)、[混合精度](https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/precision_intermediate.rst)、[梯度累积](https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/gradient_accumulation.rst)。

### CoCoGen 物理采样

- 本地仅有 Darcy、Burgers 算子。Darcy 使用角部正负源项，而 FM4PDE 数据对应 `-div(a grad u)=1` 与零 Dirichlet 边界，不能沿用旧源项。
- Burgers 本地算子的空间/时间轴是 `[B,1,X,T]`，FM4PDE 是 `[B,1,T,X]`，必须对齐。
- Poisson、Helmholtz 和 NS 需要对应的物理算子；NS 仅预测端点时，不能拿未观测的完整真实轨迹计算采样引导。遵循冻结实验指定的残差约定和可用观测。
- 本地 EulerPhysics 混用 pressure 的归一化参数更新整张状态，并在最后 50 步使用硬编码修正；`residual_step_size` 没有控制这里的实际修正。需要检查零梯度时除法、末步负时间导致的噪声方差问题，以及修正对已知观测的保持。
- 可复用 FM4PDE 的数据、观测和物理残差定义，但保留 CoCoGen 的扩散模型与采样算法。直接把 FM4PDE 的速度场和采样引导权重拿来替代，会改变被比较的方法。

### 正式评估入口

本地 `sample.py:74` 只重复第一个样本，后面计算的是标准化空间的 MSE；它不能作为 1000 样本主实验入口。需新增批量评估程序，读取冻结输入与 mask，按 sample_id 输出物理场相对 L2、PDE 残差、耗时、峰值显存、种子、网络调用次数及配置，并支持断点续跑。

## 4. 对齐主实验的精确范围

依据 FM4PDE：

- `outputs/pretrained/resume_main_20260911/main_hyperparameters_verified.csv`：66 行主实验设置。
- `scripts/train/prepare_main_resume_eval.py`：冻结真实输入、mask、批次和样本身份的逻辑。
- server197 的 `outputs/pretrained/resume_main_20260911/evaluation_inputs/catalog.json`：已存在，包含各单元 record 路径、哈希与原批次引用。它是既有实验的输入清单，不代表该续训实验已完成。

| 范围 | 每分布设置 | 分布 | 单元数 |
| --- | --- | --- | ---: |
| Darcy / Poisson / Helmholtz / NS | sparse forward、sparse inverse、sparse joint、full forward、full inverse | ID / Smooth / Rough | 4 × 5 × 3 = 60 |
| Burgers | random、sensor_column，均为 both | ID / Smooth / Rough | 1 × 2 × 3 = 6 |
| 合计 | 每单元 1000 个唯一样本 | | **66,000 次条件预测** |

只跑稀疏设置为 42 个单元 / 42,000 次预测。FM4PDE 通用 sweep 的默认组合只覆盖稀疏主实验的一部分，需要显式加入三种分布和 full 设置。

精确匹配要点：

- 128×128 网格；一般随机观测为每个活跃场 500 点，full 为已知场 16,384 点。
- sparse forward 只提供 a/f/w0 观测；sparse inverse 只提供 u/wT 观测；joint 提供两侧观测。full forward/inverse 也只能提供对应已知场的完整值。
- Burgers 使用 500 随机时空点或 5 个完整传感器列；128 个时间点时后者为 640 个观测，不是 500。
- Darcy、Poisson、Helmholtz、Burgers 使用原样本 ID 0–999；新 NS 主实验使用 2000–2999。
- mask_seed 和 sample_seed 均为 0，但并非所有设置都使用相同的 shared_mask；CSV 中有少量例外。直接读取归档 mask，不能仅重设种子后宣称观测相同。
- 每个 record 的 batches 引用了原始 result.pt/masks.pt 或 NS 批次文件，正式适配时验证引用哈希并读取原 truth。保持样本身份，明确噪声与批次规则。
- 主采样配置是 100 步。CoCoGen 可先报告 100 步且 resample=1 的成本匹配结果，再用验证集选定更长的本方法采样配置。步数相同不自动代表实际网络调用量或耗时相同；同时报告 NFE 和 GPU 时间。
- 按 `sampling/metrics.py`，逐样本计算 `||prediction-truth||₂ / ||truth||₂`，再汇总均值和标准差；forward 报 u、inverse 报 a、joint 分别报两者，Burgers 报完整时空场误差。物理残差用同一评估算子和边界口径。
- 训练验证集用于选 checkpoint 与 CoCoGen 采样超参数，最终 ID/Smooth/Rough 测试集不用于调参。不能直接继承 FM4PDE 的 zeta 值。

## 5. 耗时：证据、外推与限制

### 历史实测记录

服务器四个基础模型的 `output/2025*/lightning_logs/version_0/metrics.csv` 和 `images/train/inputs_gs-*_e-*.png` 提供旧运行证据：

- 配置为 50,000 样本、128×128、每卡 batch 32、devices 0,1、16-mixed。
- checkpoint 步数对应每轮 782 次更新，与 `ceil(50000 / 64)` 一致。
- 相邻图像文件 mtime 对应的每更新墙钟时间中位数约 1.09 秒，包含间歇采样等开销。这是从文件时间推算的旧作业速度，不是当前设备独占微基准。
- 旧模型的 8 样本 / 2000 步无条件采样，CSV 中位数约 248–251 秒/批。它不是完整物理条件采样的实测时间。
- 四个训练的日期区间重叠，配置均指定 0,1；可能存在显著资源争用。因此不能把这些值当成当前空闲两张 A100 的极限吞吐。

### 基础训练预算

按上述旧日志速度简单外推，单个模型双卡运行：

`时间 ≈ 轮数 × 782 × 1.09 秒`

| 轮数 | 单模型历史速度外推 |
| ---: | ---: |
| 100 | 23.7 小时 |
| 300 | 71.0 小时，约 3 天 |
| 1000 | 236.8 小时，约 9.9 天 |
| 5000 | 1183.9 小时，约 49 天 |

五个模型各 300 轮、逐个占用两张卡，在旧速度下约 14.8 天。资源独占、减少日志采样、优化批量后可能明显缩短，但本次未实测其幅度。可先把五模型 300 轮基础训练列为 **1–3 周的排期预算**，然后用正式环境的小规模吞吐测试更新。需要更多训练轮数时近似线性增加，不能保证 300 轮达到比较所需精度。

默认 5000 轮比 300 轮高 16.7 倍。按旧速度五模型串行约 247 天，因此不建议直接启动默认配置。第二阶段 ControlNet、多训练种子、超参数搜索的时间另计。

### 66,000 次采样预算

仅用旧 8 样本 / 2000 步约 250 秒作为网络计算参考，并假定两张卡平均分担：

`时间 ≈ (66000 / 8) × 250秒 × (步数 / 2000) × resample / 2`

| 步数 / resample | 近似 NFE/样本 | 网络部分等比例外推 |
| --- | ---: | ---: |
| 100 / 1 | 100 | 约 14.3 小时 |
| 1000 / 1 | 1000 | 约 6 天 |
| 2000 / 1 | 2000 | 约 12 天 |
| 2000 / 5 | 10000 | 约 60 天 |

这些不是新评估程序的实测结果。实际受批量、ControlNet、物理修正、重采样实现、磁盘输出和历史资源争用影响。100 步方案可先预留 **0.5–2 天**，完整 2000×5 配置应按月级别审视，待短测校准。只跑 42 个稀疏单元时，上表乘 42/66。

### 代码与验证时间

五 PDE 的数据/归一化/算子/评估适配及验证，建议预留 **2–4 个工作日**；这是工程估算，不是执行承诺。若旧代码或物理修正暴露更多数值问题，还需延长。适配完成后，安排约 30–60 分钟测量每种训练/采样的稳定吞吐与显存，再更新整个排期。

## 6. 建议实施顺序

1. 本地保留当前版本，读取服务器旧改动作为参考，创建可追踪的适配分支。服务器使用独立 Git 检出目录，不覆盖没有 Git 历史的旧 CoCoGen。
2. 先完成 Poisson：数据、normalizer、128 网络、基础训练、稀疏 forward/inverse/joint 与完整场评估。用小批次检查输入身份、有限数值、梯度、反归一化和指标；再短跑测吞吐。
3. 接入剩余四 PDE，确认 Darcy 通道和源项、Burgers 时空轴、NS 端点语义及物理参数。先在训练验证集校准采样。
4. 对旧四模型恢复训练统计并做小规模验证。如果精度和来源核查合格，可用于快速基线并补训 Burgers；若目标是训练预算一致、来源明确的新比较，从头训练五个模型。
5. 冻结 checkpoint、采样设置与完整 66 单元清单，按原样本和 mask 执行正式采样。每次只根据可用资源派发，分别比较双卡单训练与每卡独立任务的实际吞吐，不假定某种并行一定更快。
6. 长任务使用独立可辨识 tmux 会话，保存启动命令、配置、日志、退出码、代码身份、checkpoint/输入哈希及逐样本结果。任务结束按指南清理本任务会话。

建议新结果显式写入：

`/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915/`

其中分为 `training/`、`evaluation/`、`protocol/`、`logs/`。这只是建议路径，本次没有创建服务器目录。不要直接运行 FM4PDE 的训练或采样入口来代替 CoCoGen；需要先完成上述适配入口。

## 7. 后续 checkpoint 可用性实查

2026-09-15 应用户追问，在 server197 的现有 fm4pde 环境中执行了 CPU 检查。专用 tmux 会话用于该短任务，检查结束后清理；没有训练、修改模型或执行完整条件采样。逐项记录见 [checkpoint_audit_20260915.json](checkpoint_audit_20260915.json)，包含八个 checkpoint 与各自模型 YAML 的完整服务器路径。

| PDE | 基础模型 epoch / global_step | ControlNet 模型 epoch / global_step |
| --- | --- | --- |
| Darcy | 1129 / 883660 | 249 / 195500 |
| Poisson | 1624 / 1270750 | 244 / 191590 |
| Helmholtz | 1184 / 926670 | 219 / 172040 |
| NS | 1184 / 926670 | 244 / 191590 |

上表 epoch 从 0 开始。对八个 `last.ckpt`，严格加载均无缺失/多余键，模型状态浮点张量全部有限。使用一个随机正态输入、原训练的 PDE 常数标签，在 t=0.5 和 t=0.05 分别执行前向，输出均为 `[1,2,128,128]` 且全部有限。基础模型约 17.66M 参数，ControlNet 模型约 23.40M 参数。

结论是有可加载、可前向执行的预训练权重；这不等同于已经验证了完整采样稳定性或测试精度。八个文件均未保存顶层 normalizer，仍需恢复训练统计。ControlNet 的训练标签和采样条件也需核对：旧训练使用不同 PDE 常数标签，而 inpaint 构造全零条件向量，控制分支在清零条件之前读取它。

`output/withcontrol/cocogen4*_withcontrolnet.pth` 是 run_train.sh 中通过 tool_add_controlnet.py 从基础权重添加随机/零初始化控制分支所生成的文件，不应直接视为已经完成第二阶段训练的权重。应选择 `output/withcontrol/2025-.../checkpoints/last.ckpt` 等实际训练产物，并配对同目录保存的模型 YAML。是否优于基础模型必须通过相同输入与观测的评估决定。

## 8. 后续训练效果核查

2026-09-15 进一步只读取得八份历史训练 CSV，以及 Darcy / Poisson 两份存量 `results.pkl`，在本地重算统计。没有新增推理或训练。数据源相对路径、SHA256、窗口统计及逐样本误差见 [checkpoint_quality_20260915.json](checkpoint_quality_20260915.json)。

| 基础模型 | 第一轮训练 loss | 最后 50 个已记录 epoch 的平均 loss | 最近两个 50-epoch 窗口的变化 |
| --- | ---: | ---: | ---: |
| Darcy | 3434.41 | 147.42 | -0.59% |
| Poisson | 3045.74 | 415.61 | -0.39% |
| Helmholtz | 3044.45 | 417.00 | +0.29% |
| NS | 3064.18 | 389.07 | +0.31% |

所有日志均未发现 val/ 指标列；训练 loss 没有非有限值。训练目标明显下降，后期基本平台，但这不能证明泛化良好，也不能据此判断过拟合。这是按通道和网格求和的 score-matching loss，不是 PDE 求解相对误差；不同 PDE 的值不用于直接排名。日志尾部可能比周期性 last.ckpt 多几轮，因此表格描述训练过程，而非指定 checkpoint 的独立评测。

ControlNet 阶段末 50 轮平均训练 loss 相对基础阶段末 50 轮低约：Darcy 6.05%、Poisson 0.99%、Helmholtz 1.08%、NS 1.62%。这只支持训练目标有小幅改善，不能证明条件生成质量提高。

旧条件采样的真实保存数组显示：

| 数据 | 不同输入数 | 第 0 通道平均相对 L2 | 第 1 通道平均相对 L2 |
| --- | ---: | ---: | ---: |
| Darcy | 25 | 129.30%（按现存 loader 为 a） | 40.20%（u） |
| Poisson | 20 | 88.89%（f） | 86.83%（phi） |

上述相对 L2 在保存的标准化坐标中逐样本重算，再取均值；不是反归一化后的 FM4PDE 主实验结果。重算的 MSE 与存量文件标量在 float32 容差内一致。文件中的 sol_l2norm / coef_l2norm 标签与现存 loader 的通道含义相反，不能按标签直接解释。

Darcy 系数场的空间相关系数均值只有 0.148，解场为 0.930；Poisson 的源项和解场分别为 0.470、0.544。保存数组有限，但现有两组重建整体不理想。目录内的参数记录指向 ControlNet 阶段，分别记录每通道 500 和 250 个观测；这两组设置也不同于完整主实验。历史结果未内嵌生成代码身份、checkpoint 哈希、normalizer、输入 ID 和 mask，不能严格绑定到今天的某个权重文件，亦不应扩展为对所有基础模型的精度结论。未找到 Helmholtz / NS 的同类存量条件预测数组。

现存推理代码仍有训练/测试归一化不一致、ControlNet 条件不一致、场通道及物理算子参数错配等问题，可能影响这些采样结果。因此应先修正评估链路、用固定的一小批独立输入比较基础与 ControlNet checkpoint，再决定是否续训。已有训练曲线不支持仅凭增加到 5000 轮就能解决当前重建问题。
