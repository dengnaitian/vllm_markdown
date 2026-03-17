# AFD (Attention-FFN Disaggregation) 架构设计报告

**生成时间**: 2026-03-05
**分支**: add_quant_mode
**仓库**: vLLM

---

## 目录

1. [架构概述](#架构概述)
2. [核心概念](#核心概念)
3. [系统架构图](#系统架构图)
4. [组件详解](#组件详解)
5. [通信流程](#通信流程)
6. [部署模式](#部署模式)
7. [配置说明](#配置说明)
8. [性能优化](#性能优化)

---

## 架构概述

AFD (Attention-FFN Disaggregation) 是一种创新的分布式推理架构，将 Transformer 模型的**注意力层 (Attention)** 和**前馈网络层 (FFN)** 分离部署到不同的计算节点上。这种架构设计解决了以下核心问题：

### 核心优势

1. **资源隔离**：Attention 和 FFN 可以使用不同的硬件资源配置
2. **弹性扩展**：可以根据计算负载独立扩展 Attention 和 FFN 节点
3. **流水线并行**：支持多阶段流水线并行，提高吞吐量
4. **专用优化**：不同类型的计算可以使用针对性的优化策略

### 适用场景

- **MoE (Mixture of Experts) 模型**：FFN 层计算密集，需要专门的计算资源
- **大规模推理**：单节点无法容纳完整模型或吞吐量不足
- **异构硬件**：Attention 和 FFN 使用不同类型的硬件加速器

---

## 核心概念

### 1. 节点角色 (Node Roles)

```
┌─────────────────────────────────────────────────────────────┐
│                        AFD 架构角色                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ Attention Worker │         │   FFN Server     │         │
│  │  (注意力计算节点)  │         │  (前馈网络节点)   │         │
│  └──────────────────┘         └──────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Attention Worker (注意力计算节点)**
- 执行多头注意力计算 (Multi-Head Attention)
- 管理 KV 缓存
- 处理请求调度和批次管理
- 通过连接器向 FFN 节点发送数据

**FFN Server (前馈网络节点)**
- 执行 FFN 层计算 (包括 MoE 专家网络)
- 处理激活函数和层归一化
- 通过连接器返回计算结果

### 2. 连接器 (Connectors)

AFD 提供多种连接器实现，支持不同的通信场景：

| 连接器类型 | 用途 | 特点 |
|-----------|------|------|
| **dummy** | 本地测试 | 无实际通信，用于单机调试 |
| **p2p** | 点对点通信 | 基于 torch.distributed，适合小规模部署 |
| **stepmesh** | 参数服务器 | 基于 StepMesh 框架，适合大规模多节点部署 |

### 3. 元数据 (Metadata)

AFD 使用轻量级的元数据在节点间传递必要的信息：

```python
@dataclass
class AFDConnectorMetadata:
    layer_idx: int              # 当前层索引
    stage_idx: int              # 流水线阶段索引
    seq_lens: list[int]         # 序列长度列表
    dtype: torch.dtype          # 数据类型
    device: torch.device        # 设备
    num_ubatches: int           # 微批数量
    topk_ids: torch.Tensor      # MoE 专家选择索引
    topk_weights: torch.Tensor  # MoE 专家权重
    # ... 其他 MoE 和量化相关字段
```

---

## 系统架构图

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AFD 分布式推理架构                               │
└─────────────────────────────────────────────────────────────────────────────┘

                          ┌─────────────────────┐
                          │   客户端请求 (API)   │
                          └──────────┬──────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Attention Workers 节点                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │  Worker 0   │    │  Worker 1   │    │  Worker N   │                     │
│  │             │    │             │    │             │                     │
│  │ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │                     │
│  │ │Prompt   │ │    │ │Prompt   │ │    │ │Prompt   │ │                     │
│  │ │Process  │ │    │ │Process  │ │    │ │Process  │ │                     │
│  │ └────┬────┘ │    │ └────┬────┘ │    │ └────┬────┘ │                     │
│  │      │      │    │      │      │    │      │      │                     │
│  │ ┌────▼────┐ │    │ ┌────▼────┐ │    │ ┌────▼────┐ │                     │
│  │ │Attention│ │    │ │Attention│ │    │ │Attention│ │                     │
│  │ │ Layers  │ │    │ │ Layers  │ │    │ │ Layers  │ │                     │
│  │ └────┬────┘ │    │ └────┬────┘ │    │ └────┬────┘ │                     │
│  │      │      │    │      │      │    │      │      │                     │
│  │ ┌────▼────┐ │    │ ┌────▼────┐ │    │ ┌────▼────┐ │                     │
│  │ │KV Cache │ │    │ │KV Cache │ │    │ │KV Cache │ │                     │
│  │ └─────────┘ │    │ └─────────┘ │    │ └─────────┘ │                     │
│  │      │      │    │      │      │    │      │      │                     │
│  │ ┌────▼────┐ │    │ ┌────▼────┐ │    │ ┌────▼────┐ │                     │
│  │ │AFD      │ │    │ │AFD      │ │    │ │AFD      │ │                     │
│  │ │Connector│ │    │ │Connector│ │    │ │Connector│ │                     │
│  │ └────┬────┘ │    │ └────┬────┘ │    │ └────┬────┘ │                     │
│  └──────┼──────┘    └──────┼──────┘    └──────┼──────┘                     │
│         │                  │                  │                              │
└─────────┼──────────────────┼──────────────────┼──────────────────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                    ┌────────▼────────┐
                    │  通信网络 (NCCL/ │
                    │   Gloo/GRPC)   │
                    └────────┬────────┘
                             │
┌────────────────────────────┼───────────────────────────────────────────────┐
│                            │                  FFN Servers 节点              │
│                            ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                    ┌─────────────┐    ┌─────────────┐                │ │
│  │                    │  FFN Node 0 │    │  FFN Node 1 │    ...          │ │
│  │                    │             │    │             │                │ │
│  │                    │ ┌─────────┐ │    │ ┌─────────┐ │                │ │
│  │                    │ │AFD      │ │    │ │AFD      │ │                │ │
│  │                    │ │Connector│ │    │ │Connector│ │                │ │
│  │                    │ └────┬────┘ │    │ └────┬────┘ │                │ │
│  │                    │      │      │    │      │      │                │ │
│  │                    │ ┌────▼────┐ │    │ ┌────▼────┐ │                │ │
│  │                    │ │FFN      │ │    │ │FFN      │ │                │ │
│  │                    │ │Layers   │ │    │ │Layers   │ │                │ │
│  │                    │ │(MoE)    │ │    │ │(MoE)    │ │                │ │
│  │                    │ └────┬────┘ │    │ └────┬────┘ │                │ │
│  │                    │      │      │    │      │      │                │ │
│  │                    │ ┌────▼────┐ │    │ ┌────▼────┐ │                │ │
│  │                    │ │Expert   │ │    │ │Expert   │ │                │ │
│  │                    │ │Network │ │    │ │Network │ │                │ │
│  │                    │ └─────────┘ │    │ └─────────┘ │                │ │
│  │                    └─────────────┘    └─────────────┘                │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 数据流架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AFD 数据流向详解                                   │
└─────────────────────────────────────────────────────────────────────────────┘

Attention Worker                                          FFN Server
     │                                                         │
     │ 1. 输入处理                                              │
     ├─▶ [Token IDs] ──────────────────────────────────────▶ │
     │                                                         │
     │ 2. Embedding + Position Encoding                       │
     ├─▶ [Hidden States: (seq_len, hidden_dim)] ────────────▶│
     │                                                         │
     │ 3. Multi-Head Attention                                │
     ├─▶ [Q, K, V Calculation]                               │
     │    │                                                    │
     │    ├─▶ [Store K, V to KV Cache]                       │
     │    │                                                    │
     │    └─▶ [Attention Output: (seq_len, hidden_dim)]      │
     │                                                         │
     │ 4. AFD Connector: send_attn_output() ◀─────────────────┤
     │    │                                                    │
     │    │  Metadata:                                        │
     │    │  - layer_idx: current layer index                │
     │    │  - stage_idx: pipeline stage index               │
     │    │  - seq_lens: sequence lengths                    │
     │    │  - topk_ids: MoE expert selection (if MoE)       │
     │    │  - topk_weights: MoE expert weights              │
     │                                                         │
     │ 5. AFD Connector: recv_ffn_output() ◀─────────────────┤
     │    │                                                    │
     │    ▼                                                   │
     │ [FFN Output: (seq_len, hidden_dim)]                   │
     │                                                         │
     │ 6. Residual Connection + Layer Norm                    │
     ├─▶ Add & Norm                                          │
     │                                                         │
     │ 7. Next Layer / Sampling                               │
     ├─▶ [Next Token]                                        │
     │                                                         │
     │                                                         │ 4. AFD Connector: recv_attn_output()
     │                                                         │    │
     │                                                         │    ▼
     │                                                         │ [Attention Output]
     │                                                         │    │
     │                                                         │ 5. FFN Computation
     │                                                         │    │
     │                                                         │    ├─▶ Gate Projection
     │                                                         │    ├─▶ Up Projection
     │                                                         │    ├─▶ Activation (SiLU/GeLU)
     │                                                         │    ├─▶ Down Projection
     │                                                         │    │
     │                                                         │    └─▶ If MoE:
     │                                                         │         - Router selects experts
     │                                                         │         - Expert computation
     │                                                         │         - Expert output merge
     │                                                         │    │
     │                                                         │    ▼
     │                                                         │ [FFN Output]
     │                                                         │    │
     │                                                         │ 6. AFD Connector: send_ffn_output()
     │                                                         │    │
     │                                                         │    ▼
     │                                                         │ Return to Attention Worker
```

---

## 组件详解

### 1. AFDConfig 配置类

位置：`vllm/config/afd.py`

```python
@dataclass
class AFDConfig:
    # 连接器配置
    afd_connector: str = "dummy"           # 连接器类型: dummy/p2p/stepmesh
    afd_role: Literal["attention", "ffn"]  # 节点角色
    afd_host: str = "127.0.0.1"           # StepMesh 服务器地址
    afd_port: int = 1239                  # StepMesh 服务器端口

    # 并行配置
    num_afd_stages: int = 3               # 流水线阶段数
    num_attention_servers: int = 1         # Attention 服务器数量
    num_ffn_servers: int = 1               # FFN 服务器数量
    afd_server_rank: int = 0               # 当前服务器 rank

    # 计算配置
    compute_gate_on_attention: bool = False  # 是否在 Attention 侧计算 gate

    # 通信优化
    multistream_info: dict = {             # 多流通信配置
        "enable": "False",
        "core_num": "8"
    }

    # 量化配置
    quant_mode: int = 0                    # 量化模式

    # 扩展配置
    afd_extra_config: dict = {}            # 连接器特定配置
```

**关键配置说明**：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `afd_connector` | `"dummy"` | 生产环境应使用 `"p2p"` 或 `"stepmesh"` |
| `num_afd_stages` | `3` | 流水线阶段数，越大吞吐量越高但延迟增加 |
| `compute_gate_on_attention` | `False` | MoE 模型中是否在 Attention 侧计算 gate |
| `multistream_info.enable` | `"False"` | 启用多流可提高通信效率 |
| `quant_mode` | `0` | 量化模式，支持不同的量化策略 |

### 2. AFDConnector 连接器接口

位置：`vllm/distributed/afd_transfer/afd_connector/base.py`

```python
class AFDConnectorBase(ABC):
    """AFD 连接器抽象基类"""

    # Attention Worker 侧接口
    @abstractmethod
    def send_attn_output(self, hidden_states, metadata, **kwargs):
        """发送 Attention 输出到 FFN 服务器"""
        pass

    @abstractmethod
    def recv_ffn_output(self, hidden_states=None, metadata=None):
        """接收 FFN 计算结果"""
        pass

    # FFN Server 侧接口
    @abstractmethod
    def recv_attn_output(self, metadata=None, **kwargs):
        """接收来自 Attention Worker 的输出"""
        pass

    @abstractmethod
    def send_ffn_output(self, ffn_output, metadata, **kwargs):
        """发送 FFN 计算结果回 Attention Worker"""
        pass

    # MoE 相关接口
    def select_experts(self, hidden_states, router_logits, top_k, ...):
        """选择 MoE 专家"""
        pass

    def compute_moe(self, experts, hidden_states, **kwargs):
        """执行 MoE 计算"""
        pass
```

**连接器实现**：

- **DummyConnector**：本地测试用，无实际通信
- **P2PConnector**：基于 `torch.distributed` 的点对点通信
- **StepMeshConnector**：基于 StepMesh 框架的参数服务器通信
- **Camp2PConnector**：华为 Ascend NPU 专用的 P2P 连接器

### 3. AFDMetadata 元数据管理

位置：`vllm/distributed/afd_transfer/afd_connector/metadata.py`

元数据在节点间传递关键信息，支持：

```python
@dataclass
class AFDConnectorMetadata:
    # 基础信息
    layer_idx: int              # 层索引
    stage_idx: int              # 阶段索引
    seq_lens: list[int]         # 序列长度
    dtype: torch.dtype          # 数据类型
    device: torch.device        # 设备
    num_ubatches: int           # 微批数量

    # MoE 相关
    topk_idx: torch.Tensor      # 专家选择索引
    topk_weights: torch.Tensor  # 专家权重
    topk_ids: torch.Tensor      # 专家 ID
    row_idx: torch.Tensor       # 行索引
    moe_expert_num: int         # MoE 专家数量
    shared_expert_num: int      # 共享专家数量

    # 量化相关
    scale: torch.Tensor         # 量化缩放因子

    # 通信句柄
    send_handle_list: list      # 发送句柄列表
    recv_handle_list: list      # 接收句柄列表
```

**元数据工厂方法**：

```python
# Attention 侧创建元数据
metadata = AFDConnectorMetadata.create_attention_metadata(
    layer_idx=0,
    stage_idx=0,
    seq_len=1024,
    dtype=torch.float16,
    device=torch.device("cuda:0"),
    topk_weights=topk_weights,
    topk_ids=topk_ids
)

# FFN 侧创建元数据
metadata = AFDConnectorMetadata.create_ffn_metadata(
    layer_idx=0,
    stage_idx=0,
    seq_lens=[512, 256, 256],  # 多个序列
    dtype=torch.float16,
    device=torch.device("cuda:1")
)
```

### 4. GPUModelRunner (Attention Worker)

位置：`vllm/v1/worker/gpu_model_runner.py`

**核心职责**：
1. 管理模型加载和初始化
2. 执行 Attention 层计算
3. 通过 AFD 连接器与 FFN 节点通信
4. 管理 KV 缓存
5. 处理采样和输出生成

**关键流程**：

```python
class GPUModelRunner:
    def __init__(self, vllm_config):
        self.afd_config = vllm_config.afd_config
        if self.afd_config and self.afd_config.is_attention_server:
            # 初始化 AFD 连接器
            self.connector = AFDConnectorFactory.create_connector(...)

    def execute_model(self, model_executable, ...):
        # 1. 执行 Attention 层
        hidden_states = model_executable.attention_layers(...)

        # 2. 创建 AFD 元数据
        metadata = AFDConnectorMetadata.create_attention_metadata(...)

        # 3. 发送到 FFN 节点
        self.connector.send_attn_output(hidden_states, metadata, ...)

        # 4. 接收 FFN 结果
        ffn_output = self.connector.recv_ffn_output(...)

        # 5. 继续后续计算
        return ffn_output
```

### 5. GPUFFNModelRunner (FFN Server)

位置：`vllm/v1/worker/gpu_ffn_model_runner.py`

**核心职责**：
1. 只加载 FFN 层权重（节省内存）
2. 通过 AFD 连接器接收 Attention 输出
3. 执行 FFN/MoE 计算
4. 返回结果到 Attention 节点

**关键流程**：

```python
class GPUFFNModelRunner:
    def __init__(self, vllm_config):
        self.afd_config = vllm_config.afd_config
        assert self.afd_config.is_ffn_server

        # 初始化 AFD 连接器
        self.connector = AFDConnectorFactory.create_connector(...)

    def execute_model(self, model_executable, ...):
        # 1. 接收 Attention 输出
        attn_output, metadata = self.connector.recv_attn_output(...)

        # 2. 执行 FFN 层
        ffn_output = model_executable.ffn_layers(attn_output, metadata)

        # 3. 发送结果回 Attention 节点
        self.connector.send_ffn_output(ffn_output, metadata, ...)
```

---

## 通信流程

### 1. 请求处理流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        单次请求的完整通信流程                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Attention Worker                                      FFN Server
     │                                                    │
     │ [1. 接收用户请求]                                   │
     ├─▶ input_tokens: [1, 2, 3, ...]                    │
     │                                                    │
     │ [2. Token Embedding]                               │
     ├─▶ hidden_states = embed(input_tokens)             │
     │                                                    │
     │ [3. 遍历 Transformer 层]                           │
     │                                                    │
     │ for layer_idx in range(num_layers):                │
     │                                                    │
     │   [3.1 Multi-Head Attention]                       │
     │   ├─▶ Q, K, V = compute_qkv(hidden_states)        │
     │   ├─▶ attn_output = attention(Q, K, V)            │
     │   │                                                │
     │   [3.2 创建 AFD 元数据]                            │
     │   ├─▶ metadata = AFDConnectorMetadata(            │
     │   │        layer_idx=layer_idx,                   │
     │   │        stage_idx=layer_idx % num_stages,      │
     │   │        seq_lens=seq_len,                      │
     │   │        ...)                                   │
     │   │                                                │
     │   [3.3 发送到 FFN 节点] ──────────────────────────▶ │ [3.4 接收 Attention 输出]
     │   │                                                │ ├─▶ attn_output, metadata = recv_attn_output()
     │   │ send_attn_output(attn_output, metadata)        │ │
     │   │                                                │
     │   │                                                │ [3.5 执行 FFN 计算]
     │   │                                                │ ├─▶ ffn_output = ffn_layers(
     │   │                                                │ │        attn_output,       │
     │   │                                                │ │        metadata.topk_ids,  # MoE
     │   │                                                │ │        metadata.topk_weights)
     │   │                                                │ │
     │   │                                                │ [3.6 发送结果]
     │   │ ◀──────────────────────────────────────── send_ffn_output()
     │   │                                                │
     │   [3.7 接收 FFN 结果]                              │
     │   ├─▶ ffn_output = recv_ffn_output()              │
     │   │                                                │
     │   [3.8 残差连接和层归一化]                         │
     │   ├─▶ hidden_states = layer_norm(attn_output + ffn_output)
     │   │                                                │
     │   [3.9 准备下一层]                                 │
     │   └─▶ continue to next layer                      │
     │                                                    │
     │ [4. 采样生成下一个 Token]                          │
     ├─▶ next_token = sample(hidden_states[-1])          │
     │                                                    │
     │ [5. 返回结果]                                      │
     └─▶ return next_token                              │
```

### 2. MoE 专家选择流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MoE 模型的专家选择与分配                              │
└─────────────────────────────────────────────────────────────────────────────┘

Attention Worker                                      FFN Server
     │                                                    │
     │ [1. Router 计算]                                   │
     ├─▶ router_logits = router(hidden_states)           │
     │                                                    │
     │ [2. 专家选择]                                      │
     ├─▶ topk_weights, topk_ids, row_idx = select_experts(│
     │        router_logits,                              │
     │        top_k=8,                                    │
     │        use_grouped_topk=True)                      │
     │                                                    │
     │ [3. 在元数据中包含专家选择信息]                     │
     ├─▶ metadata.topk_weights = topk_weights            │
     │   metadata.topk_ids = topk_ids                    │
     │   metadata.row_idx = row_idx                      │
     │                                                    │
     │ [4. 发送 Attention 输出 + 专家信息] ───────────────▶ │ [5. 接收数据]
     │   send_attn_output(                                │ ├─▶ attn_output, metadata
     │       hidden_states,                               │ │    = recv_attn_output()
     │       metadata,                                    │ │
     │       topk_weights=topk_weights,                   │ │
     │       topk_ids=topk_ids,                           │ │
     │       row_idx=row_idx)                             │ │
     │                                                    │
     │                                                    │ [6. 根据 topk_ids 分配 token]
     │                                                    │ ├─▶ expert_tokens = distribute_tokens(
     │                                                    │ │        attn_output,
     │                                                    │ │        metadata.topk_ids,
     │                                                    │ │        metadata.row_idx)
     │                                                    │ │
     │                                                    │ [7. 并行执行专家计算]
     │                                                    │ ├─▶ for expert_id in selected_experts:
     │                                                    │ │        expert_output[expert_id] = \
     │                                                    │ │            experts[expert_id]( \
     │                                                    │ │                expert_tokens[expert_id])
     │                                                    │ │
     │                                                    │ [8. 合并专家输出]
     │                                                    │ ├─▶ ffn_output = merge_expert_outputs(
     │                                                    │ │        expert_output,
     │                                                    │ │        metadata.topk_weights,
     │                                                    │ │        metadata.row_idx)
     │                                                    │ │
     │                                                    │ [9. 发送结果]
     │                                                    │ send_ffn_output(ffn_output, metadata)
     │ ◀──────────────────────────────────────────────── │
     │                                                    │
     │ [10. 接收 FFN 输出]                                │
     └─▶ ffn_output = recv_ffn_output()                  │
```

### 3. 流水线并行 (Pipeline Parallelism)

当 `num_afd_stages > 1` 时，支持多阶段流水线并行：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      流水线并行示例 (num_afd_stages=3)                       │
└─────────────────────────────────────────────────────────────────────────────┘

时间轴
  │
  ▼

  T0:  [Req1] Layer 0 (Attn)    ─────────────────────────────────────▶
       [Req1] Layer 0 (FFN)

  T1:  [Req1] Layer 1 (Attn)    [Req2] Layer 0 (Attn) ──────────────▶
       [Req1] Layer 1 (FFN)     [Req2] Layer 0 (FFN)

  T2:  [Req1] Layer 2 (Attn)    [Req2] Layer 1 (Attn) [Req3] Layer 0 (Attn)
       [Req1] Layer 2 (FFN)     [Req2] Layer 1 (FFN)  [Req3] Layer 0 (FFN)

  T3:  [Req1] Sampling          [Req2] Layer 2 (Attn) [Req3] Layer 1 (Attn)
                             [Req2] Layer 2 (FFN)  [Req3] Layer 1 (FFN)
                                                  [Req4] Layer 0 (Attn)
                                                  [Req4] Layer 0 (FFN)

  T4:                          [Req2] Sampling     [Req3] Layer 2 (Attn)
                                                   [Req3] Layer 2 (FFN) [Req4] Layer 1
                                                                   [Req4] Layer 1

说明：
- 同一时刻，不同请求处于不同的处理阶段
- 提高吞吐量，但增加单个请求的延迟
- 适合高吞吐量场景
```

---

## 部署模式

### 1. 单机双进程模式 (Development)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        单机双进程部署 (用于测试)                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              单台服务器                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐    ┌─────────────────────────┐                │
│  │   Process 1             │    │   Process 2             │                │
│  │   (Attention Worker)    │    │   (FFN Server)          │                │
│  │                         │    │                         │                │
│  │  GPU 0: Attention       │    │  GPU 1: FFN             │                │
│  │  ┌─────────────────┐   │    │  ┌─────────────────┐   │                │
│  │  │ Attention Layer │   │    │  │  FFN Layer      │   │                │
│  │  │ KV Cache        │   │    │  │  (MoE Experts)  │   │                │
│  │  └─────────────────┘   │    │  └─────────────────┘   │                │
│  │                         │    │                         │                │
│  │  AFD Connector: dummy   │    │  AFD Connector: dummy   │                │
│  └─────────────────────────┘    └─────────────────────────┘                │
│           │                              │                                  │
│           └────────── Shared Memory ─────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

启动命令：
# Terminal 1: Attention Worker
python -m vllm.entrypoints.cli.main \
    --model deepseek-ai/deepseek-v2 \
    --afd-config '{"afd_role": "attention", "afd_connector": "dummy"}'

# Terminal 2: FFN Server
python -m vllm.entrypoints.cli.fserver \
    --model deepseek-ai/deepseek-v2 \
    --afd-config '{"afd_role": "ffn", "afd_connector": "dummy"}'
```

### 2. 多机多卡模式 (Production)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        多机多卡部署 (生产环境)                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Attention Workers 节点组                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐   │
│  │  Node 0 ( attn-0)  │  │  Node 1 ( attn-1)  │  │  Node 2 ( attn-2)  │   │
│  │                    │  │                    │  │                    │   │
│  │  GPU 0-7           │  │  GPU 0-7           │  │  GPU 0-7           │   │
│  │  ┌──────────────┐  │  │  ┌──────────────┐  │  │  ┌──────────────┐  │   │
│  │  │ Attention    │  │  │  │ Attention    │  │  │  │ Attention    │  │   │
│  │  │ Layers       │  │  │  │ Layers       │  │  │  │ Layers       │  │   │
│  │  │ KV Cache     │  │  │  │ KV Cache     │  │  │  │ KV Cache     │  │   │
│  │  └──────────────┘  │  │  └──────────────┘  │  │  └──────────────┘  │   │
│  │                    │  │                    │  │                    │   │
│  │  AFD Connector:    │  │  AFD Connector:    │  │  AFD Connector:    │   │
│  │  P2P/StepMesh      │  │  P2P/StepMesh      │  │  P2P/StepMesh      │   │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘   │
│           │                       │                       │               │
└───────────┼───────────────────────┼───────────────────────┼───────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   高速互连网络 (InfiniBand)  │
                    │   NCCL / Gloo / StepMesh    │
                    └──────────────┬──────────────┘
                                   │
┌──────────────────────────────────┼───────────────────────────────────────────┐
│                                  │              FFN Servers 节点组           │
│                                  ▼                                          │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐    │
│  │  Node 3 ( ffn-0)   │  │  Node 4 ( ffn-1)   │  │  Node 5 ( ffn-2)   │    │
│  │                    │  │                    │  │                    │    │
│  │  GPU 0-7           │  │  GPU 0-7           │  │  GPU 0-7           │    │
│  │  ┌──────────────┐  │  │  ┌──────────────┐  │  │  ┌──────────────┐  │    │
│  │  │ FFN Layers   │  │  │  │ FFN Layers   │  │  │  │ FFN Layers   │  │    │
│  │  │ MoE Experts  │  │  │  │ MoE Experts  │  │  │  │ MoE Experts  │  │    │
│  │  └──────────────┘  │  │  └──────────────┘  │  │  └──────────────┘  │    │
│  │                    │  │                    │  │                    │    │
│  │  AFD Connector:    │  │  AFD Connector:    │  │  AFD Connector:    │    │
│  │  P2P/StepMesh      │  │  P2P/StepMesh      │  │  P2P/StepMesh      │    │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

配置示例：
# Attention Worker (attn-0)
python -m vllm.entrypoints.cli.main \
    --model deepseek-ai/deepseek-v2 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 1 \
    --afd-config '{
        "afd_role": "attention",
        "afd_connector": "stepmesh",
        "afd_host": "stepmesh-server",
        "afd_port": 1239,
        "num_afd_stages": 3,
        "num_attention_servers": 3,
        "num_ffn_servers": 3,
        "afd_server_rank": 0
    }'

# FFN Server (ffn-0)
python -m vllm.entrypoints.cli.fserver \
    --model deepseek-ai/deepseek-v2 \
    --tensor-parallel-size 8 \
    --afd-config '{
        "afd_role": "ffn",
        "afd_connector": "stepmesh",
        "afd_host": "stepmesh-server",
        "afd_port": 1239,
        "afd_server_rank": 0
    }'
```

### 3. 混合并行模式 (Hybrid Parallelism)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     混合并行：TP + PP + AFD                                   │
└─────────────────────────────────────────────────────────────────────────────┘

Attention Workers (TP=2, PP=2, AFD Servers=2)
┌─────────────────┐         ┌─────────────────┐
│  Node 0         │         │  Node 1         │
│  PP Stage 0     │         │  PP Stage 0     │
│  TP Rank 0/1    │         │  TP Rank 0/1    │
│  AFD Rank 0     │         │  AFD Rank 1     │
│                 │         │                 │
│  GPU 0: Attn    │         │  GPU 0: Attn    │
│  GPU 1: Attn    │         │  GPU 1: Attn    │
└────────┬────────┘         └────────┬────────┘
         │                          │
         └──────────┬───────────────┘
                    │
         ┌──────────▼──────────┐
         │  Communication     │
         │  (NCCL/Gloo)       │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐         ┌─────────────────┐
         │  Node 2             │         │  Node 3         │
         │  PP Stage 1         │         │  PP Stage 1     │
         │  TP Rank 0/1        │         │  TP Rank 0/1    │
         │  AFD Rank 0         │         │  AFD Rank 1     │
         │                     │         │                 │
         │  GPU 0: FFN         │         │  GPU 0: FFN     │
         │  GPU 1: FFN         │         │  GPU 1: FFN     │
         └─────────────────────┘         └─────────────────┘

配置示例：
--tensor-parallel-size 2 \
--pipeline-parallel-size 2 \
--afd-config '{
    "afd_role": "attention",
    "afd_connector": "p2p",
    "num_afd_stages": 2,
    "num_attention_servers": 2,
    "num_ffn_servers": 2
}'
```

---

## 配置说明

### 1. 环境变量

```bash
# AFD 相关环境变量
export VLLM_AFD_CONNECTOR="p2p"           # 连接器类型
export VLLM_AFD_ROLE="attention"          # 节点角色
export VLLM_AFD_HOST="127.0.0.1"          # 服务器地址
export VLLM_AFD_PORT=1239                 # 服务器端口

# 多流通信配置
export VLLM_MULTISTREAM_ENABLE="True"     # 启用多流
export VLLM_MULTISTREAM_CORE_NUM=8         # 通信流核心数

# 量化模式
export VLLM_QUANT_MODE=0                  # 量化模式

# 数据并行配置
export VLLM_DISABLE_NCCL_FOR_DP_SYNC="False"  # DP 同步使用 Gloo
```

### 2. 命令行参数

```bash
# Attention Worker 启动参数
python -m vllm.entrypoints.cli.main \
    --model <model_path> \
    --tensor-parallel-size <tp_size> \
    --pipeline-parallel-size <pp_size> \
    --enable-dbo \                          # 启用 DBO (Dual Batch Overlap)
    --ubatch-size 4 \                       # 微批大小
    --afd-config '{
        "afd_role": "attention",
        "afd_connector": "p2p",
        "num_afd_stages": 3,
        "num_attention_servers": 2,
        "num_ffn_servers": 2,
        "afd_server_rank": 0,
        "multistream_info": {
            "enable": "True",
            "core_num": "8"
        },
        "quant_mode": 0
    }'

# FFN Server 启动参数
python -m vllm.entrypoints.cli.fserver \
    --model <model_path> \
    --tensor-parallel-size <tp_size> \
    --afd-config '{
        "afd_role": "ffn",
        "afd_connector": "p2p",
        "num_afd_stages": 3,
        "num_attention_servers": 2,
        "num_ffn_servers": 2,
        "afd_server_rank": 0
    }'
```

### 3. 配置文件示例

```python
# afd_config.py
from vllm.config import AFDConfig

# Attention Worker 配置
attention_config = AFDConfig(
    afd_connector="stepmesh",
    afd_role="attention",
    afd_host="stepmesh-server",
    afd_port=1239,
    num_afd_stages=3,
    num_attention_servers=3,
    num_ffn_servers=3,
    afd_server_rank=0,
    compute_gate_on_attention=True,  # MoE 模型
    multistream_info={
        "enable": "True",
        "core_num": "8"
    },
    quant_mode=0
)

# FFN Server 配置
ffn_config = AFDConfig(
    afd_connector="stepmesh",
    afd_role="ffn",
    afd_host="stepmesh-server",
    afd_port=1239,
    num_afd_stages=3,
    num_attention_servers=3,
    num_ffn_servers=3,
    afd_server_rank=0,
    multistream_info={
        "enable": "True",
        "core_num": "8"
    },
    quant_mode=0
)
```

---

## 性能优化

### 1. 通信优化

**多流通信 (Multi-Stream)**
```python
# 启用多流可提高通信效率
afd_config = AFDConfig(
    multistream_info={
        "enable": "True",      # 启用多流
        "core_num": "8"        # 指定通信流使用的 CPU 核心数
    }
)
```

**通信后端选择**
```python
# 对于 DP 同步，可以使用 Gloo 替代 NCCL
parallel_config = ParallelConfig(
    disable_nccl_for_dp_synchronization=True  # 使用 Gloo
)
```

### 2. 计算优化

**微批处理 (UBatching)**
```python
# 启用 DBO 或自定义 ubatch_size
parallel_config = ParallelConfig(
    enable_dbo=True,        # Dual Batch Overlap (固定 2 个批次)
    ubatch_size=4           # 自定义批次大小
)
```

**Gate 计算 (MoE 模型)**
```python
# 在 Attention 侧计算 gate 可以减少通信
afd_config = AFDConfig(
    compute_gate_on_attention=True
)
```

### 3. 流水线优化

**阶段数选择**
- `num_afd_stages = 1`: 最低延迟，适合低延迟场景
- `num_afd_stages = 2-3`: 平衡延迟和吞吐量
- `num_afd_stages > 3`: 最高吞吐量，适合批处理场景

### 4. 内存优化

**KV 缓存共享**
```python
# 启用 KV 缓存共享减少内存占用
cache_config = CacheConfig(
    kv_sharing=KVSharingType.SHARED_ACROSS_PP
)
```

**量化配置**
```python
# 使用量化减少通信量和内存占用
afd_config = AFDConfig(
    quant_mode=1  # 启用量化
)
```

---

## 总结

AFD 架构通过将 Attention 和 FFN 分离部署，实现了：

1. **灵活的资源分配**：不同类型的计算可以使用不同的硬件配置
2. **独立的扩展能力**：可以根据负载独立扩展 Attention 和 FFN 节点
3. **高效的并行策略**：支持流水线并行，提高吞吐量
4. **丰富的连接器支持**：提供多种通信后端，适应不同场景

这种架构特别适合：
- 大规模 MoE 模型（如 DeepSeek V2/V3）
- 需要高吞吐量的推理服务
- 异构硬件环境
- 需要灵活扩展的场景

通过合理配置 `num_afd_stages`、`multistream_info`、`quant_mode` 等参数，可以在延迟、吞吐量和资源利用率之间取得最佳平衡。
