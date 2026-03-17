# 功能差异报告：`releases/v0.11.0` → `add_quant_mode`

**生成时间**: 2026-03-05
**仓库**: vllm
**变更规模**: 59 个提交 · 130 个文件变更 · +4751 -4772 行

---

## 总体说明

`add_quant_mode` 分支主要实现了 **AFD (Attention-FFN Disaggregation)** 架构支持，允许将 Transformer 模型的注意力层和 FFN 层分离部署到不同的计算节点上，从而实现更灵活的分布式推理。该分支还增加了量化模式配置、多流通信支持以及微批处理增强功能。

---

## 功能变更详情

### AFD (Attention-FFN 分离部署)

**之前的行为**：模型的前向推理在同一计算节点上完整执行，注意力和 FFN 层串联计算。

**现在的行为**：支持将注意力和 FFN 层分离到不同的计算节点（attention workers 和 ffn servers），通过 AFD 连接器进行节点间数据传输。

**变化点**：
- 新增 AFD 配置系统，支持配置节点角色（attention/ffn）、连接器类型、通信端口等
- 支持三种 AFD 连接器：dummy（本地测试）、p2p（点对点通信）、stepmesh（基于 StepMesh 的参数服务器）
- 支持多阶段流水线并行（num_afd_stages），可将计算分成多个阶段
- 新增 AFDMetadata，用于管理 AFD 模式下的元数据（token 分布、阶段索引等）
- 在 ForwardContext 中集成 AFD 元数据，支持按阶段分割的注意力计算

**涉及文件**：[vllm/config/afd.py](vllm/config/afd.py)（新增）、[vllm/distributed/afd_transfer/](vllm/distributed/afd_transfer/)（新增）、[vllm/forward_context.py](vllm/forward_context.py)（修改）、[vllm/attention/layer.py](vllm/attention/layer.py)（修改）

---

### FFN 服务器模式

**之前的行为**：vLLM 只提供单一的服务入口，所有计算都在同一个进程中完成。

**现在的行为**：新增独立的 FFN 服务器入口点 `vllm fserver`，可以单独启动 FFN 计算节点。

**变化点**：
- 新增 `vllm fserver` CLI 命令，用于启动 FFN 服务器
- 新增 `GPUFFNModelRunner`，专门处理 FFN 服务器的计算逻辑
- FFN 服务器支持通过 AFD 连接器接收来自 attention 节点的数据并返回计算结果
- 支持 tensor parallel 模式下的 FFN 服务器部署

**涉及文件**：[vllm/entrypoints/cli/fserver.py](vllm/entrypoints/cli/fserver.py)（新增）、[vllm/entrypoints/afd_ffn_server.py](vllm/entrypoints/afd_ffn_server.py)（新增）、[vllm/v1/worker/gpu_ffn_model_runner.py](vllm/v1/worker/gpu_ffn_model_runner.py)（新增）

---

### 量化模式配置 (Quantization Mode)

**之前的行为**：量化配置通过模型配置单独设置，没有统一的量化模式参数。

**现在的行为**：在 AFD 配置中新增 `quant_mode` 参数，支持统一管理量化模式。

**变化点**：
- AFDConfig 新增 `quant_mode` 字段（int 类型），用于指定量化模式
- 支持在分布式场景下配置量化参数

**涉及文件**：[vllm/config/afd.py](vllm/config/afd.py)（新增）

---

### 多流通信 (MultiStream)

**之前的行为**：节点间通信使用单一的通信流，可能导致通信与计算重叠不充分。

**现在的行为**：支持配置多个通信流，可指定专用的通信核心数。

**变化点**：
- AFDConfig 新增 `multistream_info` 配置项，支持 `enable` 和 `core_num` 参数
- 当启用多流时，可以指定用于通信的 CPU 核心数，提高通信效率

**涉及文件**：[vllm/config/afd.py](vllm/config/afd.py)（新增）

---

### 微批处理增强 (Microbatching/UBatching)

**之前的行为**：微批处理功能只能通过 `enable_dbo`（Dual Batch Overlap）启用，固定为 2 个批次。

**现在的行为**：支持直接配置微批大小，提供更灵活的微批处理控制。

**变化点**：
- ParallelConfig 新增 `ubatch_size` 参数，可直接指定微批数量
- 新增 `use_ubatching` 和 `num_ubatches` 属性，用于判断是否启用微批处理及批次数量
- 支持 DBO 和自定义 ubatch_size 两种微批处理模式
- CLI 新增 `--ubatch-size` 参数，支持从命令行配置

**涉及文件**：[vllm/config/parallel.py](vllm/config/parallel.py)（修改）、[vllm/engine/arg_utils.py](vllm/engine/arg_utils.py)（修改）

---

### 数据并行优化

**之前的行为**：数据并行同步固定使用 NCCL 通信后端。

**现在的行为**：支持在数据并行同步中使用 Gloo 替代 NCCL，提供更灵活的通信后端选择。

**变化点**：
- ParallelConfig 新增 `disable_nccl_for_dp_synchronization` 参数
- 当设置为 True 时，DP 同步使用 Gloo 而非 NCCL

**涉及文件**：[vllm/config/parallel.py](vllm/config/parallel.py)（修改）

---

### DeepSeek 模型适配

**之前的行为**：DeepSeek V2/V3 模型支持 `deepseek_v32` 变体和 sparse MLA 注意力机制。

**现在的行为**：
- 移除 `deepseek_v32` 模型类型支持
- 移除 sparse MLA 注意力机制及相关后端
- DeepSeek MTP 模块适配 AFD 架构

**变化点**：
- ModelConfig 的 MLA 检测中移除 `deepseek_v32`
- DeepSeekV2DecoderLayer 移除 `topk_indices_buffer` 参数
- DeepSeekMTP 新增对 AFD 配置的支持
- 移除 flashmla_sparse 和 indexer MLA 后端

**涉及文件**：[vllm/model_executor/models/deepseek_v2.py](vllm/model_executor/models/deepseek_v2.py)（修改）、[vllm/model_executor/models/deepseek_mtp.py](vllm/model_executor/models/deepseek_mtp.py)（修改）、[vllm/config/model.py](vllm/config/model.py)（修改）

---

### KV 缓存计算优化

**之前的行为**：KV 缓存的缩放因子计算集成在注意力计算中。

**现在的行为**：新增独立的 KV 缩放因子计算函数，支持按需计算。

**变化点**：
- 新增 `maybe_calc_kv_scales` 自定义操作，用于计算 KV 缓存的缩放因子
- 支持通过 `enable_kv_scales_calculation` 元数据标志控制是否计算

**涉及文件**：[vllm/attention/layer.py](vllm/attention/layer.py)（修改）

---

## 新增的功能

- **AFD 架构支持**：完整的 Attention-FFN 分离部署框架，包括配置系统、连接器抽象、元数据管理
- **FFN 服务器模式**：独立的 FFN 计算节点，支持 `vllm fserver` 命令启动
- **量化模式配置**：统一的量化模式参数 `quant_mode`
- **多流通信**：支持配置专用通信流的核心数
- **灵活的微批处理**：支持直接配置微批大小 `ubatch_size`
- **数据并行后端选择**：支持在 DP 同步中使用 Gloo 替代 NCCL

---

## 不再兼容的变更

- **移除 `deepseek_v32` 模型类型**：ModelConfig 中不再支持 `deepseek_v32`，相关配置需要迁移到 `deepseek_v3`
- **移除 sparse MLA 注意力机制**：
  - Attention 层移除 `use_sparse` 参数
  - 移除 `flashmla_sparse` 和 `indexer` MLA 后端
  - 删除相关测试文件 `test_flashmla_sparse.py`、`test_sparse_mla_backends.py`
- **移除 DeepSeekV3 配置文件**：删除 `vllm/transformers_utils/configs/deepseek_v3.py`
- **DeepSeekV2DecoderLayer 构造函数变化**：移除 `topk_indices_buffer` 参数
- **移除 MTP 测试文件**：删除 `tests/v1/spec_decode/test_mtp.py`
- **移除 DeepGemm 注意力测试**：删除 `tests/kernels/attention/test_deepgemm_attention.py`
- **移除 pack/unpack Triton 测试**：删除 `tests/kernels/attention/test_pack_unpack_triton.py`
- **MTP 名称变更**：错误消息中 `mtp` 改为 `deepseek_mtp`

---

## 提交历史

| SHA | 提交信息 |
|-----|----------|
| `507be5c8a` | start_load_kv return |
| `b13c8c63a` | fix bs>64 |
| `de5335674` | update_mla_attn_params for DBO,fix FIA |
| `5663992da` | shared_output=none,enable_force_load_balance |
| `886cb3fd8` | add afdconfig quant_mode |
| `75cd53ebc` | add multistream default configuration; modify npu hard coding |
| `4f0a88318` | modify running script |
| `72d012b74` | modify stream initialization |
| `0cd6873db` | add fix ubatch_size=3 |
| `9dfc103f2` | remove dbo yield |
| `437c5a22e` | AFD refactor |
| `fe769707d` | AFDConnector, ubatch refactor |
| `3b7fd2176` | Add a multi-node startup script and a performance test script. |
| `1e345b068` | clean code |
| `10aa7425f` | [temp]graph curl slow,eager todo |
| `44bd28e65` | add multistream and core limitation of communication stream |
| `b20a5adad` | [temp] aclgraph + dbo + thread ,attn ok |
| `069db8987` | [temp] CustomOp attn comm |
| `1b42159de` | add num_ubatches |
| `eb8848cbf` | [temp] dbo + aclgraph + thread |
| `2453f0930` | add apply_dbo_yield |
| `b9f7683df` | add camp2pconnector and fused opapi, support Aclgraph. |
| `491a635b6` | Add the decode_only switch to shared_storage_connector. |
| `5f0943c49` | add switch of afd with distributed DP |
| `835e36b37` | 加入所有量化和MTP所需代码，修正bug |
| `aa274e5dd` | ffn server use vllm serve and dp |
| `01186849c` | add extends TBO |
| `d4ff4731a` | Fix: For MTP layer, transform mlp.gate into gate |
| `ed84d28e2` | [afd] DBO adaption |
| `db39fd357` | cam support aclgraph full-graph. |
| `c64d0dcc4` | Fix: MTP/AFD single-machine load weights |
| `1d9ab3355` | 去除多余代码 |
| `2b2579157` | [temp] ubatch adaption v0.1 |
| `5316eb4e7` | [AFD][M2N]ffn side support dp |
| `a50175a74` | cam aclgraph ok. |
| `17d02811a` | [BugFix]support NPU run p2p |
| `ff7a63d86` | [AFD]add ffn.sh and online_attn.sh |
| `658580f6a` | fix log for graph |
| `1a8b6c6c1` | remove original decodelayer forward |
| `df2b67daa` | add attn side send recv multi stream |
| `b225c24c8` | modify deepseekv2model and decodelayer forward |
| `c9cbf11f8` | add |
| `28e926ef7` | add |
| `ca5c66047` | add |
| `e1c8c9291` | temp |
| `e15bfeee1` | temp |
| `ca9d6d459` | add first version vllm afd adapt ascend |
| `6c9f4c449` | add ascend connector |
| `f69fdd6eb` | add metadata |
| `4d1a48601` | rebase from jcz-afd-dbo |
