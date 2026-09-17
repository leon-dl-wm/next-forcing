# Next Forcing 项目分析报告（初学者友好版）

> 论文: *Next Forcing: Causal World Modeling with Multi-Chunk Prediction* (arXiv:2606.11187, 2026-06)
> 代码: `leon-dl-wm/next-forcing`（基于 LingBot-VA 代码库开发, Apache 2.0）
> 本报告基于 **论文 PDF（`next_forcing.pdf`）+ README + 仓库源码** 交叉验证撰写, 所有文件/行号均对应当前代码, 论文数据均标注章节/公式/表格出处。

---

## 目录

1. [一句话认识这个项目](#1-一句话认识这个项目)
2. [原理速成：从零理解 Next Forcing](#2-原理速成从零理解-next-forcing)
3. [仓库结构总览](#3-仓库结构总览)
4. [各模块功能详解](#4-各模块功能详解)
5. [核心代码走读（数据流）](#5-核心代码走读数据流)
6. [快速上手：安装 → 训练 → 评测](#6-快速上手安装--训练--评测)
7. [关键超参数速查表](#7-关键超参数速查表)
8. [常见问题与坑（FAQ）](#8-常见问题与坑faq)
9. [术语表](#9-术语表)
10. [与上游 LingBot-VA 的 feature 对比](#10-与上游-lingbot-va-的-feature-对比)

---

## 1. 一句话认识这个项目

**Next Forcing 是一个"视频-动作"（Video-Action, VA）自回归世界模型**：给定机器人当前画面和语言指令，它一边"想象"未来视频，一边生成机器人动作，用于 RoboTwin 双臂操作等具身智能任务。

它的核心创新是 **MCP（Multi-Chunk Prediction，多块预测）**：训练时除了让主模型去噪"当前视频块"，再用几个轻量小模块同时预测"未来第 1/2/3 个视频块"，给模型提供更长时程的监督信号，从而：

- 训练收敛更快更稳（50fps 高帧率下最高 **2.3×** 加速收敛）；
- RoboTwin 50 个双臂任务达到 SOTA（Clean 94.1 / Random 93.5）；
- 推理时可保留 MCP 模块实现 **2× 并行加速**（类似 LLM 的投机解码；该路径尚未在本仓库发布，当前发布代码以"零开销模式"运行，即推理时丢弃 MCP 模块）。

### 模型规模

| 检查点 | 用途 | 参数量(BF16) | 大小 |
| --- | --- | ---: | ---: |
| `next-forcing-base` | 后训练（post-training）初始化 | 5.1B | ~24 GB |
| `next-forcing-posttrain-robotwin` | RoboTwin 评测 | 6.7B | ~26 GB |

base 模型是纯主干（无 MCP）；MCP 模块在后训练开始时创建，并用主干最后 3 层 Transformer block 的权重初始化，所以后训练检查点更大。

---

## 2. 原理速成：从零理解 Next Forcing

### 2.1 预备知识（3 分钟）

**(a) 视频世界模型**：一个能"预测未来画面"的生成模型。机器人领域用它做策略：模型想象"如果我这样动，世界会变成什么样"，同时直接输出动作。

**(b) 扩散 / Flow Matching**：生成模型的一种训练方式。训练时给干净数据 `x0` 加噪声得到 `xt`，让网络预测"速度场"（本仓库的目标是 `noise - x0`，见 `wan_va/utils/scheduler.py:111` `training_target`）；推理时从纯噪声出发，迭代若干步把噪声"去噪"成数据。

**(c) 分块自回归（chunk-wise autoregressive）视频生成**：长视频不是一次生成的，而是切成固定长度的 **chunk（块）**，逐块生成：第 N 块的生成以第 1..N-1 块为条件（因果性）。本项目中 1 个 chunk = 2 个 latent 帧（RoboTwin 配置），1 个 latent 帧 = 4 个真实视频帧（VAE 时间压缩 4×）。

**(d) KV Cache**：和 LLM 一样，已生成块的 Key/Value 张量缓存起来，生成新块时不必重算历史，只前向新 token。

### 2.2 问题：短视监督（myopic supervision）

现有 World Action Model（WAM）的主流训练目标是 **teacher-forced next-chunk denoising**：以真值历史块（clean context）为条件，只对**紧邻的当前块**去噪（论文 §1、§3.2 式 3）。作者指出两个根本缺陷：

- **外观捷径（appearance shortcut）**：预测下一块本质是个局部任务——相邻块画面高度相似，模型只要学会"从干净历史块到当前块的近似恒等映射 + 少量残差修正"就能大幅压低去噪损失。这条捷径比学习真实动力学容易得多，会吸收大部分梯度信号，使模型缺乏学习长程时序演化的压力（论文称之为 **myopic supervision，短视监督**）；
- **帧率越高问题越严重**：50 fps 下相邻块的外观差异缩小到"捷径几乎无损"的程度，标准 teacher forcing 收敛显著变慢、终值更低（论文图 1：50fps Random 5k 步时 baseline 只有 31.9%，Next Forcing 已达 61.6%）。

> 顺带一提：teacher forcing 还有经典的 **exposure bias**（训练看真值历史、推理看自生成历史）。这由"noisy history augmentation"缓解（训练时以 0.5 概率给条件流加噪，对应代码 `train.py:347` 的 `noisy_cond_prob=0.5`；消融显示去掉它 baseline 从 75.6% 掉到 69.8%，论文表 2）。Next Forcing 解决的是**另一个正交问题：监督目标太短视**。

### 2.3 解法：MCP 多块预测（论文 §4）

灵感直接来自 LLM 的 **多 token 预测（MTP）**：训练辅助模块预测多个未来 token。把该思想搬到视频世界模型有三个非平凡差异（论文 §1）：预测目标是连续视频 latent 而非离散 token；生成靠迭代去噪而非单步采样；时序依赖跨越多个不同尺度的时程。

训练时在主模型（30 层 Transformer 主干）旁边挂 `num_mcp_depths = 3` 个轻量 MCP 模块（每个只有 `mcp_blocks_per_depth = 3` 层 Transformer block），形成一条**因果链**（论文图 2）：

```
                       ┌─────────────── 主干 (30 层) ───────────────┐
 当前块 noisy latent → │  Block0 ... Block3 ... Block11 ... Block19 ... Block29 │ → 当前块去噪输出
                       └──────┬──────────┬──────────┬──────────┬─────┘
              收集第 4/12/20/30 层 hidden states（视频 token: noisy 当前块 + clean 历史）
                              ↓ mcp_hidden_fuser (两层 MLP, 论文式 7)
              ┌───────────────┴────────────────┐
              │  MCP depth 1 (3 blocks)        │ → 预测 next¹ 块（未来第 1 块）
              │  z⁽¹⁾ = W₁[h_fuse ; Embed(next¹ noisy)] （论文式 8）│
              └───────────────┬────────────────┘
                    输出 hidden 作为 h_prev 传给下一深度
              ┌───────────────┴────────────────┐
              │  MCP depth 2 (3 blocks)        │ → 预测 next² 块
              └───────────────┬────────────────┘
              ┌───────────────┴────────────────┐
              │  MCP depth 3 (3 blocks)        │ → 预测 next³ 块
              └────────────────────────────────┘
```

六个设计要点（论文 §4.2-4.3/4.5 ↔ 代码逐条对应）：

1. **时序块平移（论文式 4）**：把训练样本的视频 latent 向未来平移 k 个块 `x₀[k][i] = x₀[min(i+k, F)]`，越界部分复制最后一块填充；**末尾 k 个填充块不计入损失**（论文式 12）。代码：`wan_va/mcp.py:36` `shift_latents_for_mcp`（`frame_shift = (depth+1) × chunk_size` 帧）+ `valid_mask`。
2. **独立加噪 + 更高 timestep shift（论文式 5, s_mcp=10 > s_main=5）**：每个平移目标用独立 timestep/噪声加噪。**动机**：更高噪声意味着 MCP 自己的输入携带的目标信息更少，被迫更多依赖主干表征来去噪——把 MCP 损失的梯度"推进"主干，而不是被轻量 MCP 模块自己吸收（消融：s_mcp 降到 5 会掉 2.6 个点，论文表 2）。代码：独立调度器 `train_scheduler_mcp`（`train.py:252`）。
3. **多层特征融合（论文式 7）**：收集主干第 **4/12/20/30 层**（1-based；代码 0-based `[3,11,19,29]`）的 hidden states——**同时包含 noisy 当前块和 clean 历史块的视频 token**——沿特征维拼接后过两层 MLP 压缩。浅层携带粗结构、深层携带细粒度信息，融合让 MCP 梯度能反传到主干**不同深度**（消融：去掉多层融合掉 2.2 个点）。代码：`model.py:1051` 收集 + `mcp_hidden_fuser`（`model.py:689`）。
4. **跨深度因果链（论文式 8）**：`z[k] = W_k[h_prev[k-1] ; Embed(x_t[k])]`，`h_prev[0] = h_fuse`；每个深度的输出 hidden 同时作为下一深度的 `h_prev`——近期预测为更远期预测提供信息。代码：`_forward_mcp` 中 `previous_hidden_states = mcp_hidden_states[:, :latent_length]`（`model.py:943`）。
5. **RoPE 位置平移（论文式 6）**：`RoPE(x₀[k][i]) = RoPE(i+k)`，让 MCP 模块知道自己预测的是第几个未来块。代码：`_add_noise(..., frame_shift)` → `get_mesh_id(f_shift=frame_shift)`。
6. **共享注意力掩码 + 共享输出头（论文 §4.3、附录 A）**：MCP 序列结构与主干相同（noisy 目标 + clean 上下文），**每个训练步只构建一次 FlexAttention 掩码**，主干和 3 个 MCP 深度共用；MCP 输出与主干共用 `norm_out` + `proj_out` 视频解码头。

**总损失（论文式 13）**：`L = L_video + L_action + 0.5·L_MCP₁ + 0.2·L_MCP₂ + 0.1·L_MCP₃`（`wan_va/train.py:473` `_train_step`）。**梯度穿过 MCP 模块和融合层流回主干各深度**，这就是"稠密的多尺度时序监督"。MCP block 用主干最后 3 层权重初始化（`model.py:777`），fuser/投影层随机初始化（消融：不做权重初始化掉 2.0 个点）。

### 2.4 一个检查点、两种推理模式（论文 §4.6）

- **零开销模式**（当前发布代码）：加载检查点后调用 `disable_mcp_modules()` 删掉全部 MCP 权重（融合 MLP、投影层、轻量 block；`wan_va/wan_va_server.py:83` 以 `disable_mcp=True` 加载），架构/延迟/显存与 baseline 完全一致。**该模式的所有质量增益都来自训练时 MCP 目标反传进主干的更丰富监督信号，推理零成本**。
- **并行块生成模式**（论文报告 2×，代码待发布）：只保留 **depth-1** MCP 模块——一次去噪轨迹中，主干产出当前块的同时 depth-1 MCP 并行产出下一块；MCP block 比主干轻一个数量级，加入前向"几乎免费"，于是**每个自回归步推进 2 个块 → 2× 加速**（思路类似 LLM 的投机/并行解码）。depth-2/3 在此模式不用：它们的预测会在下一步被主干的正式预测取代；论文指出同机制可扩展到更高加速比，但代价是累积漂移（留作未来工作）。

### 2.5 实验结果（论文 §5）

**训练规模**（论文 §5.1）：LingBot-VA 框架 + Wan2.2 30 层主干；先大规模多本体数据预训练（即 `next-forcing-base`），再在 RoboTwin 后训练：**2,500 条 Clean 演示（50/任务）+ 25,000 条 Random 演示（500/任务），64 GPU，至多 50k 步**（消融实验用 16 GPU、仅 Clean、25fps、20k 步）。

**① RoboTwin SOTA**（论文表 1，50 任务平均成功率 %）：

| | X-VLA | π0 | π0.5 | Motus | Being-H0.7 | Fast-WAM | LingBot-VA | **Next Forcing** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 72.9 | 65.9 | 82.7 | 88.7 | 90.2 | 91.9 | 92.9 | **94.1** |
| Random | 72.8 | 58.4 | 76.8 | 87.0 | 89.6 | 91.8 | 91.5 | **93.5** |

**② 收敛加速**（论文图 1 / 表 5）：

| 帧率 | 指标 | LingBot-VA | Next Forcing |
| --- | --- | --- | --- |
| 12 fps | 达到 90% 所需步数 | ~20k | **10k（~2× 加速）** |
| 12 fps | 50k 步终值 (Clean/Random) | 92.8 / 91.8 | **94.1 / 93.5** |
| 50 fps | 5k 步 (Clean/Random) | 45.5 / 31.9 | **70.2 / 61.6（+24.7 / +29.7）** |
| 50 fps | 达到 baseline 45k 精度所需步数 | 45k | **20k（2.3× 加速）** |
| 50 fps | 50k 步终值 (Clean/Random) | 88.6 / 85.2 | **91.8 / 90.5** |

**为什么高帧率下增益最大？**（论文 §5.2.2）从监督信号密度理解：高帧率下相邻块几乎相同，next-chunk 去噪靠"复制外观"就能平凡解决；而往前 2~3 个块与当前块有显著视觉差异，**只有理解底层物理动力学才能预测**，MCP 迫使模型发展时序感知的表征。

**③ 推理加速**（论文表 4，成功率 %）：

| 模式 | 12fps Clean/Random | 25fps Clean/Random | 50fps Clean/Random |
| --- | --- | --- | --- |
| 标准 | 94.1 / 93.5 | 92.6 / 91.4 | 91.8 / 90.5 |
| MCP 加速 (2×) | 93.5 / 90.6 | 91.0 / 89.8 | 92.2 / 91.3 |

**④ PhyWorld 物理规律基准**（论文表 3，去掉动作流只测视频生成）：FVD OOT **4.7** vs 5.3、IT **3.2** vs 3.5；异常率 OOT **8%** vs 12%、IT **2%** vs 3%。OOT（out-of-template）增益更大 → MCP 学到的是可泛化的物理动力学而非模板记忆。

**⑤ 通用视频预训练**（论文 §5.2.4）：3.5M 条 5-10s 视频片段（以人类活动为主），去掉动作流，32 GPU 训练纯视频生成。两个 1024 样本测试集上，50k 步 FVD：Test Set 1（人类活动）**94 vs 225（-58%）**、Test Set 2（相机驱动场景动态）**97 vs 204（-52%）**；且 **10k 步就已超过 baseline 50k 步**——证明 MCP 的收益不限于机器人数据。

### 2.6 方法坐标：Next Forcing 在 "forcing" 家族中的位置（论文 §1、§2.2）

自回归视频生成的现有 "forcing" 方法可按**改了什么**分类，Next Forcing 与它们全部正交、可组合：

| 方法 | 改变的东西 | 代表工作 |
| --- | --- | --- |
| Teacher Forcing | 模型**看到什么上下文**（真值历史） | LingBot-VA, DreamZero |
| Self Forcing | 模型**看到什么上下文**（自生成历史 + 分布匹配损失） | Self Forcing |
| Diffusion Forcing | **噪声如何调度**（逐帧独立噪声水平） | Diffusion Forcing |
| **Next Forcing** | **模型被要求预测什么**（当前块 → 未来多块） | 本项目 |

> 本仓库代码同时使用了逐帧独立 timestep（Diffusion Forcing 风格, `train.py:284`）和条件流加噪（noisy history augmentation），与 MCP 目标叠加——印证"可组合"。

### 2.7 论文消融速览：每个设计选择的代价（论文表 2, RoboTwin Clean 子集 20k 步）

| 配置 | SR (%) | 结论 |
| --- | ---: | --- |
| Baseline（s_main=5 + noisy history aug） | 75.6 | 起点 |
| 去掉 noisy history aug | 69.8 | 模型会走"直接复制干净上下文"的捷径 |
| s_main = 1 / 10 / 20 / 25 | 65.3 / **78.4** / 77.6 / 77.2 | timestep shift 影响大, 10 最优, 再高收益递减 |
| **Baseline + MCP（默认）** | **85.8** | **+10.2 个点** |
| s_mcp = 5（与主干相同） | 83.2 | 高 shift 迫使 MCP 依赖主干表征, 强化耦合 |
| 去掉多层特征融合 | 83.6 | 融合中间层让梯度深入主干 |
| 去掉权重初始化 | 83.8 | 从主干末层初始化有效 |
| MCP block 数 = 1 / 5 | **86.5** / 85.0 | 越轻耦合越紧; 但默认取 3——1 block 的 MCP 生成块视觉伪影更多, 会影响并行块生成模式 |

**给初学者的启示**：MCP 有效的关键不是"多几个头"，而是**让监督梯度尽可能流回主干**——高噪声 shift、轻模块、多层融合、主干初始化，四个设计全部服务于"强化耦合"这一个目标。论文自述局限：MCP 带来额外训练开销（§6）。

---

## 3. 仓库结构总览

```text
next-forcing/
├── wan_va/                  ★ 核心 Python 包：模型 + 训练 + 推理服务
│   ├── modules/
│   │   ├── model.py         ★★ WanTransformer3DModel 主干 + MCP 模块 + FlexAttention 掩码 + KV Cache (1196 行)
│   │   └── utils.py         VAE / UMT5 文本编码器 / tokenizer / transformer 加载器, 流式 VAE 封装
│   ├── mcp.py               MCP 工具函数: 配置校验 + latent 未来平移 (59 行)
│   ├── train.py             ★★ 训练入口: Trainer 类, 加噪/损失/训练循环/存档 (822 行)
│   ├── wan_va_server.py     ★★ 推理服务: VA_Server, 自回归 chunk 生成 + KV cache 管理 (729 行)
│   ├── configs/             所有 EasyDict 配置（推理×4 + 训练×2 + MCP 默认 + 共享）
│   ├── dataset/
│   │   └── lerobot_latent_dataset.py  LeRobot latent 数据集加载 (548 行)
│   ├── distributed/         FSDP2 分片 + 激活检查点 + 分布式工具
│   ├── utils/               FlowMatch 调度器 / RoPE 网格 id / 日志 / websocket 服务封装
│   │   └── Simple_Remote_Infer/  websocket 策略服务器（openpi 风格, msgpack 序列化）
│   └── build_dataset_index.py  离线构建数据集索引缓存
├── script/
│   ├── run_va_posttrain.sh          训练启动器 (torchrun)
│   ├── run_launch_va_server_sync.sh 推理服务启动器 (torchrun)
│   └── create_lerobot_latent_view.py 把"分离存储的 latent"组装成 LeRobot 数据视图 (411 行)
├── evaluation/robotwin/     RoboTwin 2.0 评测：客户端 + 启动脚本 + 统计
│   ├── eval_policy_client_openpi.py  评测主循环 (703 行)
│   ├── websocket_client_policy.py    websocket 客户端
│   ├── launch_server{,_multigpus}.sh / launch_client{,_multigpus}.sh
│   ├── calc_stat.py / geometry.py / test_render.py / msgpack_numpy.py
├── example/                 demo / franka / robotwin 三套初始观测图片（i2va 模式输入）
├── tests/                   pytest 单元测试（MCP、数据集索引缓存、latent 视图）
├── docs/                    项目主页源码（gangweix.github.io/next-forcing）
├── README.md / INSTALL.md / Makefile / pyproject.toml / requirements.txt
```

★ = 初学者必读文件。总代码量约 6600 行 Python。

---

## 4. 各模块功能详解

### 4.1 `wan_va/modules/model.py` — 模型核心（最重要）

**`WanTransformer3DModel`**（`model.py:569`）：Wan 系列 DiT（Diffusion Transformer）的视频-动作改造版。

| 组件 | 说明 |
| --- | --- |
| 主干 | 30 层 `WanTransformerBlock`，hidden = 24 头 × 128 = 3072，FFN 14336 |
| 输入 | 视频 latent 48 通道（Wan2.2 VAE），patch (1,2,2) 后线性嵌入；动作 30 维线性嵌入；文本 UMT5 4096 维投影 |
| 时间步嵌入 | `WanTimeTextImageEmbedding`，视频/动作各一套（`condition_embedder` / `condition_embedder_action`），逐 token 调制（AdaLN scale/shift/gate） |
| 位置编码 | `WanRotaryPosEmbed` 3D RoPE（帧/高/宽），动作 token 用分数帧偏移插在视频帧之间（`wan_va/utils/utils.py:33` `get_mesh_id`） |
| 输出头 | 视频 `proj_out`（48×patch），动作 `action_proj_out`（30 维） |

**论文视角：Mixture-of-Transformers（MoT）双流架构**（论文 §4.4）：视频流和动作流共享 30 层注意力 block，但各自拥有独立的嵌入器（`patch_embedding_mlp` vs `action_embedder`）、独立的时间步条件嵌入（`condition_embedder` vs `condition_embedder_action`）和独立输出头——两流在同一注意力中按掩码交互（跨模态注意力）。联合预测分解为两步（论文式 9）：**先预测未来视觉动态 `x_{i+1}`，再以逆动力学（inverse dynamics）方式解码动作 `a_i ~ g(·|x_{≤i+1}, a_{<i}, ℓ)`**——动作以"包含预测出的未来块"的观测为条件。这解释了掩码中"动作 frame_id = 块id×2+1 > 视频 frame_id = 块id×2"的设计：同块内动作 token 能看到该块视频 token。MCP 只作用于视频流，改进的视频表征通过共享注意力间接惠及动作解码。

**注意力实现（三种模式）**：`torch`（SDPA，推理默认）、`flashattn`、`flex`（FlexAttention，**训练默认**）。

**`FlexAttnFunc`**（`model.py:42`）— 训练时的块因果注意力掩码，是理解本项目的关键：

训练序列布局为 4 段拼接：`[noisy 视频 | clean 视频(条件) | noisy 动作 | clean 动作(条件)]`（batch 展平进序列维，用 `seq_ids` 区分样本）。每个 token 有三个 id：

- `frame_id`：视频 token = `帧号 // chunk_size × 2`，动作 token = `× 2 + 1` → 同一 chunk 内"动作在视频之后"；
- `noise_id`：0 = noisy（待去噪），1 = clean（条件）；
- 掩码规则（`_get_mask_mod`, `model.py:154`；与论文附录 A 图 5 逐条一致）：
  1. clean → clean：块因果（kv 帧 ≤ q 帧）；
  2. noisy → clean：严格过去（kv 帧 < q 帧）——当前块能看到**历史块的干净上下文**，但**排除同块的 clean token，防止当前块真值泄漏**；
  3. noisy → noisy：仅同块（kv 帧 == q 帧）——块内双向、块间隔离；
  4. clean → noisy：**禁止**（掩码中无此规则）；
  5. 再叠加同样本掩码和滑动窗口 `|q帧 - kv帧| ≤ window_size`（训练时随机 4~64）。
- **MCP 模块复用同一掩码**（论文附录 A 的关键设计）：MCP 序列结构与主干相同（noisy 未来块 + clean 上下文），每个训练步只构建一次掩码，主干 + 3 个 MCP 深度共享，降低训练开销。

这正是 **Diffusion Forcing / Self Forcing 风格的因果块扩散**：一次前向并行训练所有 chunk，每个 chunk 只看历史 clean 上下文 + 本 chunk noisy token。

**KV Cache（推理）**（`WanAttention.init_kv_cache/allocate_slots/update_cache`, `model.py:345-411`）：

- 每层自注意力维护一个 token 池，容量 = `(attn_window/2) × 每块视频token数 + (attn_window/2) × 每块动作token数`（RoboTwin: 36 块视频 + 36 块动作）；
- `update_cache` 参数语义：`0` = 临时写入、算完即释放（去噪中间步）；`1` = 写入并标记 `is_pred=True`（生成完的块，属"预测"缓存）；`2` = 写入且 `is_pred=False`（真实观测，永久历史）；
- 池满时按写入 id 淘汰最老 token（`allocate_slots`）；
- `clear_pred_cache`：真实观测到来后，把之前"想象"的预测缓存清掉，换成真实缓存——**闭环纠错**，防止误差累积。

**MCP 相关方法**：

| 方法 | 行号 | 功能 |
| --- | --- | --- |
| `_build_mcp_modules` | 687 | 创建 fuser MLP + 3 个输入投影 + 3×3 个 MCP Transformer block |
| `enable_mcp_training` | 714 | 后训练开始时给 base 模型"装上" MCP（校验架构一致 / 初始化 / 写回 config） |
| `initialize_mcp_blocks_from_backbone` | 777 | 用主干最后 3 层权重初始化每个深度的 MCP block |
| `disable_mcp_modules` | 785 | 推理零开销模式：删除 MCP 模块 |
| `_forward_mcp` | 843 | 训练前向：融合收集层 hidden → 逐深度链式预测 next^1/2/3 块 |

**两条前向路径**：

- `forward_train`（`model.py:967`）：4 段拼接 + FlexAttention 掩码 + 收集第 3/11/19/29 层 hidden → 主干输出（视频/动作速度场）+ MCP 输出列表；
- `forward`（`model.py:1093`）：推理路径，单段输入（视频或动作，`action_mode` 切换），走 KV cache，`train_mode=False`。

### 4.2 `wan_va/mcp.py` — MCP 工具（59 行，建议先读）

- `validate_mcp_settings`：校验深度数/块数/收集层/损失权重合法性；
- `shift_latents_for_mcp(latents, frame_shift)`：把 `[B,C,F,H,W]` latent 向未来平移 `frame_shift` 帧（尾部用最后一帧填充），返回 `(shifted, valid_mask)`。`valid_mask` 标记平移后仍有真实帧支撑的位置，MCP loss 只在这些位置计算。

### 4.3 `wan_va/train.py` — 训练入口

**`Trainer`**（`train.py:48`）职责：

1. **建模**：加载 base transformer（fp32→CPU），`enable_mcp_training` 装 MCP，`apply_ac` 激活检查点，FSDP2 `fully_shard` 分片（bf16 参数 / fp32 规约），AdamW(fused) + warmup-constant LR；
2. **数据**：`MultiLatentLeRobotDataset` + `DistributedSampler`；多卡时 rank0 先建索引缓存、其余 rank barrier 等待（避免网络存储上并发扫描）；
3. **加噪**（`_add_noise`, `train.py:284`）：**逐帧独立采样 timestep**（Diffusion Forcing 风格），FlowMatch 加噪得到 `noisy_latents`，目标 = `noise - latent`；视频条件流以 `noisy_cond_prob=0.5` 的概率也被加噪（t∈[500,1000)，模拟推理时不完美的历史上下文，提升鲁棒性）；
4. **组装输入**（`_prepare_input_dict`, `train.py:337`）：随机 `chunk_size ∈ [1,4]`、随机 `window_size ∈ [4,64]`；为 3 个 MCP 深度分别做 `frame_shift = (d+1)×chunk_size` 的平移 + 加噪（MCP 用独立调度器，`mcp_snr_shift=10`）；
5. **损失**（`compute_loss`, `train.py:393`）：视频 MSE（按帧归一、乘 bell 形 timestep 权重）+ 动作 MSE（乘 `actions_mask`）+ Σ `w_d × MCP_d`；
6. **循环**（`train`, `train.py:609`）：按 step 训练（非 epoch），支持梯度累积、梯度裁剪(2.0)、WandB 记录（含每个 MCP 深度的 loss）、每 `save_interval` 步存 diffusers 格式检查点（bf16 safetensors + config.json，可直接用于评测）。

**命令行**：`python -m wan_va.train --config-name robotwin_train [--pretrained-model-path ... --dataset-path ... --save-root ... --num-steps ... --disable-wandb ...]`，环境变量 `RANK/LOCAL_RANK/WORLD_SIZE` 由 torchrun 注入。

### 4.4 `wan_va/wan_va_server.py` — 推理服务

**`VA_Server`**（`server.py:38`）加载 4 个组件：Wan2.2 VAE（+流式封装 `WanVAEStreamingWrapper`，支持逐 chunk 编码并缓存因果卷积状态）、UMT5 文本编码器、tokenizer、transformer（`disable_mcp=True`，`attn_mode="torch"`）。VAE/文本编码器可 offload 到 CPU 省显存。

三种工作模式（`infer()` 分发, `server.py:605`）：

| 客户端消息 | 服务端行为 |
| --- | --- |
| `{reset: True, prompt}` | `_reset`：清空 KV cache/VAE cache，编码 prompt（含 CFG 负提示），按 `attn_window` 建 KV 池 |
| `{obs, ...}`（普通推理） | `_infer`：生成一个 chunk（先视频后动作，见 §5.2），返回 32 步动作 |
| `{obs, state, compute_kv_cache: True}` | `_compute_kv_cache`：丢弃预测缓存，用**真实观测**编码 latent + 真实机器人状态，以 `update_cache=2` 写入永久历史 |

`infer_mode='i2va'` 时走 `generate()`（`server.py:646`）：从 `example/` 图片出发，纯想象自回归生成 10 个 chunk，VAE 解码导出 `demo.mp4`——**无需机器人环境即可看效果**。

**RoboTwin 的 T 形拼图观测**（`env_type='robotwin_tshape'`）：头部相机 256×320 → latent 16×20；双腕相机各缩到 128×160 → latent 8×10，两个横向拼接成 8×20，再与头部 latent 纵向拼接成 **24×20** 的"T 形"单帧 latent（`_encode_obs`, `server.py:323`）。这样一个 chunk 的视频 token 数 = 2 帧 × 12×10 patch = 240。

### 4.5 `wan_va/configs/` — 配置中心

全部用 `EasyDict`，注册在 `VA_CONFIGS` 字典（`configs/__init__.py`）：

| 配置名 | 文件 | 用途 |
| --- | --- | --- |
| `robotwin` | `va_robotwin_cfg.py` | RoboTwin 推理（server 模式），含动作归一化统计量 q01/q99 |
| `robotwin_i2va` | `va_robotwin_i2va.py` | RoboTwin 图生视频 demo |
| `franka` / `franka_i2va` | `va_franka_cfg.py` 等 | Franka 真机配置 |
| `demo` / `demo_i2va` | `va_demo_cfg.py` 等 | 最小 demo（2 相机 256×256） |
| `robotwin_train` | `va_robotwin_train_cfg.py` | **RoboTwin 后训练**（= robotwin 推理配置 + mcp_train_cfg + 训练超参） |
| `demo_train` | `va_demo_train_cfg.py` | demo 数据后训练（lr 1e-4, 梯度累积 8, 2000 步） |
| （被继承） | `mcp_train_config.py` | MCP 默认超参（与发布检查点一致） |
| （被继承） | `shared_config.py` | host/port、bf16、patch_size、索引缓存开关 |

路径类配置全部支持环境变量覆盖：`NEXT_FORCING_PRETRAINED_MODEL_PATH`（训练初始化）、`NEXT_FORCING_MODEL_PATH`（推理检查点）、`NEXT_FORCING_DATASET_PATH`、`NEXT_FORCING_SAVE_ROOT`。

### 4.6 `wan_va/dataset/lerobot_latent_dataset.py` — 数据集

**设计思想：训练时不跑 VAE/文本编码器，全部用预计算 latent**（省显存、快）。

- `LatentLeRobotDataset`（继承 LeRobot `LeRobotDataset`）：每个样本 = 一个 episode 的一个片段 `[start_frame, end_frame)`，读取：
  - `latents/chunk-XXX/<相机>/episode_XXXXXX_<start>_<end>.pth`：含 `latent`（已展平的 VAE latent）、`latent_num_frames/height/width`、`frame_ids`、`text_emb`（预计算 T5 嵌入）；
  - LeRobot parquet 中的原始 `action` 列；
- `_cat_video_latents`：多相机 latent 拼接（robotwin 用 T 形布局）；以 `cfg_prob=0.1` 概率把 text_emb 换成 `empty_emb.pt`（CFG 训练）；
- `_action_post_process`（`dataset:461`）：RoboTwin 下把双臂绝对位姿转成**相对首帧的相对位姿**（`get_relative_pose`），q01/q99 分位数归一化到 [-1,1]（clip ±1.5），16 维有效通道映射到 30 维统一动作空间，reshape 成 `[C=30, F_latent, 16, 1]`（每 latent 帧 16 个动作步）+ 同形状 mask；
- `MultiLatentLeRobotDataset`：递归扫描 `dataset_path` 下所有含 `meta/info.json` 的 LeRobot 仓库并拼接；
- **索引缓存**：首次运行会逐片段校验 latent 文件存在性（网络存储上很慢），结果按指纹（episodes.jsonl 大小/mtime + latent 根目录 + 相机 keys）缓存到 `<数据集>/.cache/next_forcing/valid_metas_*.json`；多卡训练时 rank0 建缓存、其他 rank 等待。可用 `python -m wan_va.build_dataset_index_cache` 离线预建。

### 4.7 `wan_va/distributed/` — 分布式

- `fsdp.py`：`shard_model` 用 **FSDP2**（`fully_shard`）逐 block 分片（attn1/attn2/ffn 各自分片），MCP block 同样处理；`apply_ac` 给所有 block（含 MCP）套激活检查点；
- `util.py`：`init_distributed`（NCCL）、`_configure_model`（分布式→分片 / 单卡→直接 to(device)）、`dist_mean/dist_max`（跨卡日志聚合）。

### 4.8 `wan_va/utils/` — 工具

- `scheduler.py` **`FlowMatchScheduler`**：Flow Matching 噪声调度。`shift` 参数把 sigma 分布向高噪端偏移（视频 5.0 / 动作 1.0 / MCP 10.0）；`add_noise`（逐帧 t）、`training_target`（= noise - sample）、`training_weight`（bell 形权重，中间 timestep 权重大）、`step`（推理欧拉步）；
- `utils.py`：`get_mesh_id`（3D RoPE 网格坐标，动作 token 特殊处理）、`data_seq_to_patch`（token 序列还原成 latent 张量）、`sample_timestep_id`、`save_async`（后台线程存盘不阻塞训练）；
- `server_utils.py`：多卡推理服务封装——rank0 起 websocket 服务器，其余 rank 在 `worker_loop` 里等 broadcast 同步前向（FSDP 需要所有 rank 参与计算）；
- `Simple_Remote_Infer/`：openpi 风格的 `WebsocketPolicyServer`（msgpack-numpy 序列化，`/healthz` 健康检查）。

### 4.9 `script/` — 启动器与数据工具

- `run_va_posttrain.sh`：`torchrun --nproc_per_node=$NGPU -m wan_va.train --config-name $CONFIG_NAME`，额外参数原样透传（如 `--num-steps 1`）；
- `run_launch_va_server_sync.sh`：同结构的推理服务启动器；
- `create_lerobot_latent_view.py`：**数据准备核心工具**。输入 JSONL manifest（每行 `{repo_id, latent_path}`），输出一个"管理式数据视图"目录：用 symlink 组装 `data/videos/meta/latents`，扫描 latent 文件名 `episode_(\d+)_(\d+)_(\d+).pth` 反推每个 episode 的可用片段，重写 `meta/episodes.jsonl` 的 `action_config`（片段级 start/end/action_text），并链接 `empty_emb.pt`。支持 `--refresh-view` 增量刷新。

### 4.10 `evaluation/robotwin/` — RoboTwin 2.0 评测

- `eval_policy_client_openpi.py`：评测客户端主循环（详见 §5.3）。要点：每个 seed 先跑 **expert check**（内置规划器验证该 seed 可解）才计入测试；每任务默认 100 trials；动作是 16 维相对末端位姿（左臂 7 + 夹爪 1 + 右臂 7 + 夹爪 1），客户端把它加回初始位姿转成绝对位姿后 `take_action`；每 trial 存对比可视化视频（文件名带 `_True/_False` 后缀）和 `res.json`；
- `launch_server.sh` / `launch_server_multigpus.sh`：起 1 / 8 个推理服务（GPU 0-7，端口 29556+i）；
- `launch_client.sh` / `launch_client_multigpus.sh`：单任务 / 按任务组（7 组 × 8 任务 = 50 任务，含重复填充）并行评测；
- `calc_stat.py`：从可视化目录按 `*_True.mp4 / *_False.mp4` 文件名统计成功率，按任务难度类 1/2/3 分组输出均值；
- `geometry.py`（欧拉角/四元数转换）、`test_render.py`（Sapien 渲染自检）、`msgpack_numpy.py`、`websocket_client_policy.py`（自动重连等待服务器）。

### 4.11 `tests/`、`example/`、`docs/`

- `tests/`：3 个纯 CPU 单测——`test_mcp.py`（平移/掩码/配置校验）、`test_dataset_index_cache.py`（缓存读写与指纹失效）、`test_create_lerobot_latent_view.py`（视图组装）。`pytest tests/` 即可跑，无需 GPU/数据；
- `example/`：`demo`（top+wrist 2 图）、`franka`（3 图）、`robotwin`（3 图）初始观测 PNG，供 `*_i2va` 配置使用；
- `docs/`：GitHub Pages 项目主页（含方法图 `next-forcing.png`、收敛曲线、对比视频）。

### 4.12 论文 ↔ 代码对照表（读论文时按图索骥）

| 论文位置 | 内容 | 代码位置 |
| --- | --- | --- |
| §3.1 式 1-2 | Flow Matching 加噪 / 速度目标 `ε - x₀` | `utils/scheduler.py:99` `add_noise`, `:111` `training_target` |
| §3.2 式 3 | teacher-forced next-chunk 去噪 | `train.py:284` `_add_noise` + `model.py:154` 掩码规则 |
| §4.2 式 4 | 时序块平移 `x₀[k][i]=x₀[min(i+k,F)]`, 复制末块填充 | `mcp.py:36` `shift_latents_for_mcp` |
| §4.2 式 5 | MCP 独立加噪, s_mcp=10 | `train.py:252` `train_scheduler_mcp`, `configs/mcp_train_config.py` |
| §4.2 式 6 | RoPE 位置平移 `RoPE(i+k)` | `utils/utils.py:33` `get_mesh_id(f_shift=...)` |
| §4.3 式 7 | 多层特征融合（第 4/12/20/30 层, 两层 MLP） | `model.py:1051` 收集, `:689` `mcp_hidden_fuser`, 配置 `mcp_hidden_collect_layers=[3,11,19,29]` |
| §4.3 式 8 | 因果链 `z[k]=W_k[h_prev;Embed(x_t[k])]` | `model.py:697` `mcp_input_projections`, `:943` 链式传递 |
| §4.4 式 9 | MoT 双流 + 逆动力学动作解码 | `model.py:967` `forward_train` 四段序列 + frame_id 交错设计 |
| §4.5 式 10-13 | 视频/动作/MCP 损失与加权求和 | `train.py:393` `compute_loss`, `:473` `_train_step` |
| §4.6 | 零开销模式（丢弃 MCP） | `model.py:785` `disable_mcp_modules`, `server.py:83` |
| §4.6 | 并行块生成模式（2×, 未发布） | —（checkpoint 含 MCP 权重, `enable_mcp: true`） |
| §5.1 | chunk size M ~ {1..4} 随机 | `train.py:339` `torch.randint(1, 5)` |
| §5.1 | noisy history augmentation p=0.5 | `train.py:347` `noisy_cond_prob=0.5` |
| §5.1 | MCP 权重从主干末层初始化 | `model.py:777` `initialize_mcp_blocks_from_backbone` |
| 附录 A 图 5 | 注意力掩码四条规则 + MCP 共享掩码 | `model.py:154` `_get_mask_mod`, `:95` `init_mask`（每步一次） |
| 附录 C 式 14-15 | timestep shift 公式 `σ̃ = sσ/(1+(s-1)σ)` | `utils/scheduler.py:57` `set_timesteps` |

---

## 5. 核心代码走读（数据流）

### 5.1 训练一个 step（`Trainer._train_step`）

```
batch (DataLoader)
 ├─ latents      [B, 48, F, H, W]     预计算多相机拼接视频 latent
 ├─ actions      [B, 30, F, 16, 1]    归一化动作
 ├─ actions_mask [B, 30, F, 16, 1]
 └─ text_emb     [B, 512, 4096]       预计算 T5 嵌入 (10% 概率为空提示)
        │
        ▼ _prepare_input_dict (train.py:337)
 随机 chunk_size∈[1,4], window_size∈[4,64]
 ├─ latent_dict  = _add_noise(视频)      逐帧独立 t, 50% 概率条件流也加噪
 ├─ action_dict  = _add_noise(动作)      action_snr_shift=1.0
 └─ mcp_latent_dicts ×3 = _add_noise(shift_latents_for_mcp(latents, (d+1)*chunk_size))
        │
        ▼ transformer.forward_train (model.py:967)
 序列 = [noisy视频 | clean视频 | noisy动作 | clean动作] (+padding 到 128 倍数)
 FlexAttention 块因果掩码 → 30 层主干 (收集第 3/11/19/29 层 hidden)
 → _forward_mcp: 3 个深度链式预测 next^1/2/3 块
 → 输出: 视频速度场, 动作速度场, [mcp_out ×3]
        │
        ▼ compute_loss (train.py:393)
 loss = 视频MSE(逐帧归一×timestep权重) + 动作MSE(×mask)
      + 0.5·MCP₁ + 0.2·MCP₂ + 0.1·MCP₃   (MCP 只在 valid_mask 处计算)
        │
        ▼ backward → clip_grad_norm(2.0) → AdamW.step (每 gradient_accumulation_steps 次)
```

### 5.2 推理一个 chunk（`VA_Server._infer`, server.py:441）

```
噪声 latents [1,48,2,24,20]          噪声 actions [1,30,2,16,1]
        │                                     │
        ▼ 25 步 FlowMatch 去噪 (CFG=5)         │  (先视频, 后动作)
 每步: transformer(noisy视频 token,            │
       KV cache=历史 clean 块, update_cache=0) │
 最后一步(t=0): update_cache=1 →              ▼ 50 步去噪 (action CFG=1)
   生成块以 is_pred=True 写入缓存      每步: transformer(action_mode=True)
        │                              最后一步同样写入缓存
        ▼                                     ▼
   (latent 异步存盘)                  postprocess_action: 反归一化
                                              ▼
                              返回 16 维 × 2 帧 × 16 步 = 32 个动作
```

### 5.3 评测闭环（server ↔ client）

```
client: reset(prompt) ──────────────► server: 清缓存, 编码 prompt
client: get_obs() 首帧
loop:
  client: infer(obs) ───────────────► server: _infer 生成 chunk → 32 动作
  client: 逐步 take_action(绝对位姿), 每 4 步采一帧真实观测 (共 8 帧)
  client: infer(obs=8帧, state=动作,
              compute_kv_cache=True) ► server: clear_pred_cache 丢弃想象缓存,
                                        流式 VAE 编码真实观测 (8帧→2 latent帧),
                                        update_cache=2 写入真实历史
  client: 检查 eval_success / step_lim
每 trial: 保存对比视频 (*_True/_False.mp4) + res.json
结束后: calc_stat.py 汇总成功率
```

---

## 6. 快速上手：安装 → 训练 → 评测

> 所有命令在仓库根目录执行。测试环境：Python 3.10 + PyTorch 2.9.0 + CUDA 12.6。

### Step 0. 安装

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt --no-build-isolation   # 含 flash-attn、lerobot 0.3.3
```

评测另需独立的 RoboTwin 2.0 环境（`git checkout 2eeec322`，按官方文档装 Vulkan + 资产），并 `export ROBOTWIN_ROOT=/path/to/your/RoboTwin`。详见 `INSTALL.md`。

### Step 1. 下载检查点

```bash
pip install "huggingface_hub[cli]"
# 后训练用 base（5.1B, ~24GB）
hf download gangweix/next-forcing-base --local-dir ./checkpoints/next-forcing-base
# 评测用 posttrain（6.7B, ~26GB）
hf download gangweix/next-forcing-posttrain-robotwin --local-dir ./checkpoints/next-forcing-posttrain-robotwin
```

目录为标准 diffusers 布局：`transformer/ vae/ text_encoder/ tokenizer/`。**必须用本地路径**（不支持 Hub repo id）。

### Step 2. 准备数据（后训练才需要）

数据集根目录要求：包含若干 LeRobot 仓库（每个有 `meta/info.json`），且每个仓库带预计算 latent：

```text
$NEXT_FORCING_DATASET_PATH/
├── empty_emb.pt                        # 空提示 T5 嵌入（CFG 用）
└── <task>/                             # LeRobot 仓库
    ├── meta/ (info.json, episodes.jsonl 含 action_config 片段, ...)
    ├── data/  videos/
    └── latents/chunk-000/<相机key>/episode_000000_<start>_<end>.pth
```

若 latent 单独存储，用视图工具组装：

```bash
python script/create_lerobot_latent_view.py \
  --manifest my_manifest.jsonl \        # 每行 {"repo_id": "...", "latent_path": "..."}
  --output-root /path/to/dataset_view \
  --empty-emb /path/to/empty_emb.pt
# 之后可 --refresh-view /path/to/dataset_view 增量刷新
```

可选：离线预建索引缓存（避免首次训练卡在扫描）：

```bash
python -m wan_va.build_dataset_index_cache --config-name robotwin_train
```

### Step 3. 训练

```bash
export NEXT_FORCING_PRETRAINED_MODEL_PATH=$PWD/checkpoints/next-forcing-base
export NEXT_FORCING_DATASET_PATH=/path/to/your/dataset      # 内含 empty_emb.pt
export NEXT_FORCING_SAVE_ROOT=/path/to/your/output

# 8 卡正式训练（默认 robotwin_train, MCP 开启, 50k 步, 每 1k 步存档）
NGPU=8 CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh --init-worker 1

# 单卡冒烟测试（1 步, 关 wandb）——推荐先跑这个验证环境
CUDA_VISIBLE_DEVICES=0 NGPU=1 CONFIG_NAME=robotwin_train \
bash script/run_va_posttrain.sh --num-steps 1 --init-worker 1 --load-worker 0 --disable-wandb
```

- 输出：`$NEXT_FORCING_SAVE_ROOT/checkpoints/checkpoint_step_N/transformer/`（diffusers 格式，可直接作为评测的 `NEXT_FORCING_MODEL_PATH`）；
- 常用覆盖参数：`--learning-rate --batch-size --gradient-accumulation-steps --num-steps --save-interval --cfg-prob --disable-wandb`；
- 开 WandB 需设置 `WANDB_BASE_URL / WANDB_API_KEY / WANDB_TEAM_NAME` 环境变量（`train.py:51`）。

### Step 4. RoboTwin 评测

```bash
export NEXT_FORCING_MODEL_PATH=$PWD/checkpoints/next-forcing-posttrain-robotwin
export ROBOTWIN_ROOT=/path/to/your/RoboTwin

# 终端 1：单卡推理服务
CUDA_VISIBLE_DEVICES=0 bash evaluation/robotwin/launch_server.sh
# 终端 2：单任务 100 trials
bash evaluation/robotwin/launch_client.sh /path/to/eval_results adjust_bottle

# 或 8 卡全量：先 launch_server_multigpus.sh，再
bash evaluation/robotwin/launch_client_multigpus.sh /path/to/eval_results 0 0 100
#   参数: SAVE_ROOT TASK_GROUP_ID(0-6) SEED TEST_NUM
```

结果统计：`python evaluation/robotwin/calc_stat.py <可视化结果目录>`。
日志在 `./logs`，模型生成可视化在 `./visualization`。

### Step 5.（可选）零门槛看效果：i2va demo

不需要机器人环境和数据集，只要检查点 + `example/` 图片：

```bash
NGPU=1 CONFIG_NAME=robotwin_i2va bash script/run_launch_va_server_sync.sh
# 从 example/robotwin/*.png 出发想象 10 个 chunk, 导出 $SAVE_ROOT/demo.mp4
```

### Step 6.（可选）跑单元测试

```bash
pytest tests/ -v      # 纯 CPU, 无需数据/权重
```

---

## 7. 关键超参数速查表

### MCP（`configs/mcp_train_config.py`，与发布检查点一致）

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `enable_mcp` | True | 所有训练配置默认开启 |
| `num_mcp_depths` | 3 | 预测 next^1/2/3 三个未来块 |
| `mcp_blocks_per_depth` | 3 | 每个深度 3 层 Transformer block（轻量） |
| `mcp_hidden_collect_layers` | [3,11,19,29] | 收集主干第 4/12/20/30 层（0-based 索引）hidden |
| `mcp_loss_weights` | [0.5,0.2,0.1] | 越远的未来块权重越小 |
| `mcp_snr_shift` | 10.0 | MCP 独立噪声调度（偏高噪） |
| `mcp_init_from_backbone` | True | 用主干最后 3 层初始化 MCP block |

### 训练（`va_robotwin_train_cfg.py`）

| 参数 | 值 | | 参数 | 值 |
| --- | --- | --- | --- | --- |
| lr | 2e-5 (warmup 100, constant) | | batch_size | 1 / GPU |
| optimizer | AdamW β=(0.9,0.95), wd=0.1 | | grad clip | 2.0 |
| num_steps | 50000 | | save_interval | 1000 |
| snr_shift (视频/动作) | 5.0 / 1.0 | | cfg_prob | 0.1 |
| chunk_size | 随机 1~4 latent 帧/样本 | | window_size | 随机 4~64 |
| noisy_cond_prob | 0.5（条件流加噪增强） | | 精度 | bf16 参数 / fp32 规约 |

> 论文口径（§5.1）：RoboTwin 后训练用 **64 GPU**、2,500 Clean + 25,000 Random 演示、至多 50k 步；发布配置 `num_steps=50000` 与之一致，但 `NGPU=8` 时全局 batch 是论文的 1/8，复现终值精度需相应增加步数或梯度累积。消融实验用 16 GPU、仅 Clean 数据、25fps、20k 步。

### RoboTwin 推理（`va_robotwin_cfg.py`）

| 参数 | 值 | 说明 |
| --- | --- | --- |
| 分辨率 | 256×320（头部）, 128×160（腕部） | T 形拼接后 latent 24×20 |
| frame_chunk_size | 2 latent 帧 = 8 真实帧 | 每 chunk 240 视频 token |
| action_per_frame | 16 | 每 chunk 32 个动作步 |
| num_inference_steps | 视频 25 / 动作 50 | FlowMatch 欧拉步 |
| guidance_scale | 视频 5 / 动作 1 | CFG |
| attn_window | 72 | KV cache 容量 = 36 chunk 视频 + 36 chunk 动作 |
| action_dim | 30（有效 16 维） | 双臂 7+1 ×2, q01/q99 归一化 |

---

## 8. 常见问题与坑（FAQ）

1. **模型路径必须是本地目录**。代码用 `os.path.join(path, 'transformer')` 解析子目录，填 Hub repo id 会报错。
2. **推理为什么看不到 MCP？** 发布代码以 `disable_mcp=True` 加载（零开销模式），MCP 权重在 checkpoint 里但加载后被删除；2× 加速推理路径官方说明将单独发布。
3. **首次训练卡在 "Loading dataset indexes" 很久？** 正常——要逐片段校验 latent 文件是否存在（网络存储更慢）。缓存建好后（`.cache/next_forcing/`）后续秒级加载；多卡时只有 rank0 扫描，其余 rank 在 barrier 等待，不要手动 kill。
4. **flash-attn 装不上？** 需要与 CUDA/torch 匹配的预编译 wheel；训练实际用 `flex`（FlexAttention + torch.compile），推理用 `torch` SDPA，flash-attn 仅是可选后端。
5. **评测 client 连不上 server？** 两者必须在同一台机器（client 连本地端口）；client 会自动每 5s 重试直到 server 就绪。多卡评测注意端口对应：server GPU i ↔ 端口 29556+i ↔ client 任务 i。
6. **动作维度对不上？** RoboTwin 配置的有效动作通道是 `used_action_channel_ids`（0-6 左臂 + 28 左夹爪 + 7-13 右臂 + 29 右夹爪），其余通道被 mask 置零；换自己的机器人需要改 `norm_stat`（q01/q99）和通道映射。
7. **想复现论文不同帧率结果？** 训练数据的 latent `frame_ids` 步长（frame_stride）决定帧率，动作数 = latent 帧数 × stride × 4；配置里的 `action_per_frame` 需与数据一致。
8. **单卡能训练吗？** 可以（冒烟测试命令见 §6 Step 3），但 5.1B+MCP 全参训练单卡显存压力极大（FSDP 无分片收益 + 激活检查点），正式训练建议 8 卡。
9. **checkpoint 能直接评测吗？** 能。训练存档就是 diffusers 布局（`checkpoint_step_N/transformer/`），把 `NEXT_FORCING_MODEL_PATH` 指到 `checkpoint_step_N` 即可。
10. **优化器状态/断点续训？** `save_checkpoint` 中优化器状态保存代码被注释掉了（`train.py:526` 附近），`resume_from` 只恢复权重不恢复 step/optimizer——长训练中断需要留意。

---

## 9. 术语表

| 术语 | 解释 |
| --- | --- |
| **WAM (World Action Model)** | 世界动作模型：联合建模未来视频与动作的具身智能范式，先预测视觉动态再解码动作（论文 §2.1） |
| **VA 模型** | Video-Action 模型：同一 Transformer 同时去噪视频 latent 和动作序列（本仓库的实现形态） |
| **MoT (Mixture-of-Transformers)** | 视频/动作两流共享注意力 block、各自独立嵌入器与输出头的架构（论文 §4.4） |
| **逆动力学（inverse dynamics）** | 由"观测到的前后画面"反推"中间执行了什么动作"；本模型动作流以含预测未来块的观测为条件解码动作（论文式 9） |
| **MTP (Multi-Token Prediction)** | LLM 中预测多个未来 token 的训练目标，MCP 的灵感来源（论文 §1） |
| **chunk（块）** | 自回归生成的最小单位；RoboTwin 中 = 2 个 latent 帧 = 8 个真实帧 = 32 个动作步 |
| **latent** | VAE 压缩后的视频表示（Wan2.2 VAE：空间 16×、时间 4× 压缩，48 通道） |
| **Flow Matching** | 扩散模型的一种形式：预测速度场 `noise - x0`，推理用欧拉步积分（论文 §3.1） |
| **timestep shift (s)** | 噪声调度偏移 `σ̃ = sσ/(1+(s-1)σ)`，s 越大训练越偏向高噪区间；主干 5 / MCP 10（论文附录 C） |
| **MCP** | Multi-Chunk Prediction：本论文核心，轻量模块链式预测未来 1~3 块，提供长程监督 |
| **短视监督（myopic supervision）** | 只监督下一块导致模型学外观捷径而非动力学的问题（论文 §1） |
| **外观捷径（appearance shortcut）** | 相邻块高度相似时，"近似恒等映射 + 小残差"即可压低损失的现象 |
| **Teacher / Self / Diffusion Forcing** | 分别改变"上下文来源（真值/自生成）"与"噪声调度（逐帧独立）"的既有范式；Next Forcing 改变"预测目标"，与三者正交可组合（论文 §2.2） |
| **exposure bias** | 训练看真值历史、推理看自生成历史的分布差；本仓库用 noisy history augmentation（p=0.5）缓解 |
| **Diffusion Forcing** | 每帧独立采样噪声水平的训练范式（本仓库逐帧独立 timestep） |
| **KV Cache** | 缓存历史块 Key/Value，避免重复计算；本项目带窗口淘汰 + 预测/真实双标记 |
| **is_pred 缓存** | 模型"想象"出的块先以 pred 身份入缓存，真实观测到来后被清除替换（闭环纠错） |
| **CFG** | Classifier-Free Guidance：条件/无条件两路前向按 guidance_scale 外推 |
| **FSDP2** | PyTorch 全分片数据并行（`fully_shard`），逐 block 分片参数/梯度/优化器状态 |
| **FlexAttention** | PyTorch 可编程注意力掩码 API（torch.compile 加速），训练时实现块因果掩码 |
| **LeRobot 数据集** | HuggingFace 机器人数据格式（parquet 动作 + 视频 + meta/episodes.jsonl） |
| **RoboTwin 2.0** | 双臂机器人操作仿真基准（50 任务，Clean 固定初始配置 / Random 随机物体位姿与场景布局） |
| **PhyWorld** | 检验视频生成是否遵循物理规律（匀速直线、弹性碰撞、抛物线等）的基准（论文 §5.1） |
| **FVD** | Fréchet Video Distance：视频生成质量指标，越低越好 |
| **OOT / IT** | Out-of-Template / In-Template：PhyWorld 的模板外（组合泛化）/ 模板内设置 |
| **T 形拼图** | RoboTwin 三相机 latent 拼接方式：头部相机在上、双腕相机（半分辨率）横向拼接在下 |
| **i2va** | Image-to-Video-Action：从静态图片出发纯想象生成视频+动作的 demo 模式 |
| **expert check** | 评测前用内置规划器验证 seed 可解，保证测试集有效性 |

---

## 附：初学者推荐阅读顺序

1. `README.md`（项目全貌 + 结果表）
2. 本报告 §2（原理）；有余力再读论文原文：§1 引言（问题动机）→ §4 方法（式 4-13）→ 附录 A（注意力掩码图）→ §5.3 消融（设计取舍）
3. `wan_va/mcp.py`（59 行，MCP 数据变换一目了然）
4. `wan_va/configs/mcp_train_config.py` + `va_robotwin_train_cfg.py`（超参，对照论文 §5.1 实现细节）
5. `wan_va/train.py` 的 `_add_noise → _prepare_input_dict → compute_loss → _train_step`（训练数据流）
6. `wan_va/modules/model.py` 的 `FlexAttnFunc._get_mask_mod`（因果掩码，对照论文附录 A 图 5）和 `_forward_mcp`（MCP 前向，对照论文式 7-8）
7. `wan_va/wan_va_server.py` 的 `_infer / _compute_kv_cache`（推理闭环）
8. `evaluation/robotwin/eval_policy_client_openpi.py` 的 `eval_policy`（评测协议）

---

## 10. 与上游 LingBot-VA 的 feature 对比

> 对比对象：`lingbot-va`（HEAD `b591d16`；最新代码提交 `7c6ffa9`, 2026-07-10，其后仅文档提交）。
> 方法：全文件树 diff + 逐文件代码 diff（核心文件变更行数实测）。

### 10.1 两者关系与定位

Next Forcing 是 **LingBot-VA 的研究型 fork**（README 致谢明确说明），论文中 LingBot-VA 即 baseline。实测代码重合度极高：

| 文件 | diff 行数 | 结论 |
| --- | ---: | --- |
| `wan_va/utils/scheduler.py` / `utils.py` | 0 | 逐字节相同 |
| `evaluation/robotwin/calc_stat.py` / `geometry.py` / `msgpack_numpy.py` | 0 | 逐字节相同 |
| `wan_va/wan_va_server.py` | 18 | 仅导入方式 + `disable_mcp=True` + typo 修复，**推理逻辑相同** |
| `wan_va/modules/model.py` | 297 | **几乎全部是 MCP 新增**，主干架构相同 |
| `wan_va/train.py` | 321 | MCP 训练 + 索引缓存协调 + CLI 覆盖 |
| `wan_va/dataset/lerobot_latent_dataset.py` | 229 | 几乎全部是索引缓存系统 |
| `requirements.txt` | 0 | 相同 |

**定位差异**：LingBot-VA 是通用 VA 基础模型工具箱（RoboTwin + LIBERO + Franka 真机，附 VA/VA2 两篇论文）；Next Forcing 收窄到 **RoboTwin 单基准 + MCP 训练目标**，换取 SOTA 精度和更强的复现工程。

### 10.2 Next Forcing 新增的 feature

| # | Feature | 代码位置 | 说明 |
| --- | --- | --- | --- |
| 1 | **MCP 多块预测训练目标**（核心） | `mcp.py`(新)、`model.py`(+297)、`train.py`(+321)、`fsdp.py`(+14)、`configs/mcp_train_config.py`(新) | next¹/²/³ 链式预测、多层融合 [3,11,19,29]、独立调度器 s_mcp=10、损失加权 [0.5,0.2,0.1]、MCP block 也做 FSDP 分片 + 激活检查点、逐深度 WandB 日志 |
| 2 | **零开销推理开关** | `modules/utils.py` `load_transformer(disable_mcp=...)`、`model.py` `disable_mcp_modules` | 同一检查点训练带 MCP / 推理删 MCP；也能借此加载无 MCP 的旧检查点（架构兼容） |
| 3 | **数据集索引缓存系统** | `dataset/lerobot_latent_dataset.py`(+229)、`build_dataset_index.py`(新)、`shared_config.py` 开关 | 指纹校验的 valid_metas JSON 缓存 + HF Arrow 文件直载 + 原子写入；多卡时 rank0 建缓存其余 barrier 等待；lingbot 每次全量扫描且 `Pool(128)` 硬编码 |
| 4 | **LeRobot latent 数据视图工具** | `script/create_lerobot_latent_view.py`(新, 411 行) | symlink 组装分离存储的 latent 成标准 LeRobot 数据集，自动重写 episodes.jsonl 的 action_config；lingbot 无对应工具 |
| 5 | **单元测试** | `tests/` ×3(新) | MCP 平移/校验、索引缓存、数据视图；**lingbot-va 完全没有测试** |
| 6 | **完整 CLI 覆盖 + 环境变量配置** | `train.py` argparse（12 个新参数）、`NEXT_FORCING_*` / `ROBOTWIN_ROOT` 环境变量 | lingbot 训练只有 `--save-root`，路径全部硬编码 `/path/to/...`，wandb key 直接写在启动脚本里 |
| 7 | **正规 Python 打包** | 相对导入（去掉 `sys.path.append` hack）、`pyproject.toml` `packages.find`、`-m wan_va.xxx` 启动 | lingbot 用文件路径启动 + 运行时改 sys.path |
| 8 | **工程清理** | typo 修复：`eval_polict→eval_policy`、`i2av→i2va`、`sever_utils→server_utils`；`infer_mode` 收敛到 shared_config；black 替代 yapf；依赖精确 pin（`torch==2.9.0`）+ `train` extras | 提高可维护性与可复现性 |
| 9 | **项目主页** | `docs/`(新, GitHub Pages) | 方法图、收敛曲线、对比视频；lingbot 只有 assets/teaser |
| 10 | **训练默认值调整** | `va_robotwin_train_cfg.py` | lr 1e-5→**2e-5**、warmup 10→**100**、wandb 默认开→**关**、init_worker 默认 1 |

### 10.3 LingBot-VA 有、Next Forcing 移除的 feature

| # | Feature | 位置（lingbot-va） | 影响 |
| --- | --- | --- | --- |
| 1 | **LIBERO 基准全套支持** | `evaluation/libero/`（client 224 行 + 启动脚本）、`va_libero_{cfg,i2va,train_cfg}.py`、`example/libero/`、配置注册 `libero*` | next-forcing **不能直接跑 LIBERO**；需要时得从上游回移 |
| 2 | **额外部署工具** | `Simple_Remote_Infer/deploy/{qwenpi_policy, replay_policy, image_tools, websocket_client_policy}.py` + README | OpenVLA 风格策略评测、回放策略、图像传输压缩工具不再随包提供 |
| 3 | **更详尽的 README**（26KB vs 13KB） | 自定义数据集准备指南、attn_mode 配置说明、真机部署结果、ModelScope 下载链接、News 时间线 | 自定义数据接入文档变薄（部分由本报告 §6 弥补） |
| 4 | **两篇 LingBot 论文 PDF + teaser 素材** | `LingBot_VA_paper.pdf`、`LingBot_VA2_paper.pdf`、`assets/` | 仅资料层面 |

### 10.4 性能对比（Next Forcing 论文口径）

| 维度 | LingBot-VA | Next Forcing |
| --- | --- | --- |
| RoboTwin 50 任务 (Clean/Random) | 92.9 / 91.5 | **94.1 / 93.5** |
| 50fps 收敛速度 | 45k 步 | **20k 步（2.3×）** |
| Clean 子集 20k 步消融 | 75.6% | **85.8%（+10.2）** |
| 推理成本（当前发布代码） | 基准 | 与基准相同（零开销模式） |
| 推理成本（MCP 2× 模式） | — | 论文报告 2×，**代码未发布** |
| 训练成本 | 基准 | 更高（MCP 额外前向/反向 + 1.6B 额外参数，论文自述局限） |

### 10.5 选型建议

- **跑 LIBERO、真机 Franka 部署、需要 replay/qwenpi 工具** → 用 `lingbot-va`；
- **追求 RoboTwin SOTA 精度、高帧率场景、更快收敛、更好的复现工程（测试/缓存/CLI/环境变量）** → 用 `next-forcing`；
- **自定义数据后训练**：两者数据格式相同（LeRobot + 预计算 latent），next-forcing 额外提供 `create_lerobot_latent_view.py` 组装工具和索引缓存，大规模/网络存储下体验明显更好；但自定义数据集的文档要参考 lingbot-va README；
- **检查点**：架构兼容（同为 30 层 Wan2.2 主干、diffusers 布局）。next-forcing 的推理代码通过 `disable_mcp` 也能加载无 MCP 的主干检查点（如 `next-forcing-base`）；反向用 lingbot-va 代码加载 next-forcing posttrain 检查点时，diffusers 会丢弃其不认识的 `enable_mcp` 配置项、MCP 权重成为 unexpected keys（理论上退化为零开销模式，未经上游验证）。
