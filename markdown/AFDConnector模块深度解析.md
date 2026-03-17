# AFDConnector 模块深度解析

**生成时间**: 2026-03-05
**分支**: add_quant_mode
**仓库**: vLLM
**模块**: vllm.distributed.afd_transfer.afd_connector

---

## 目录

1. [模块概述](#模块概述)
2. [核心接口设计](#核心接口设计)
3. [连接器实现详解](#连接器实现详解)
4. [工厂模式与插件机制](#工厂模式与插件机制)
5. [元数据管理](#元数据管理)
6. [通信协议详解](#通信协议详解)
7. [节点连接拓扑](#节点连接拓扑)
8. [异步通信机制](#异步通信机制)
9. [最佳实践](#最佳实践)

---

## 模块概述

### 什么是 AFDConnector？

AFDConnector 是 AFD (Attention-FFN Disaggregation) 架构中的**核心通信抽象层**，负责在 Attention Workers 和 FFN Servers 之间传输数据和元数据。它屏蔽了底层通信机制的复杂性，提供统一的接口供上层调用。

### 设计目标

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AFDConnector 设计目标                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐      统一接口      ┌──────────────┐                      │
│  │   Attention  │ ──────────────────▶│ AFDConnector │                      │
│  │   Worker     │                     │   (抽象层)    │                      │
│  └──────────────┘                     └──────┬───────┘                      │
│                                             │                               │
│                                             │ 可插拔后端                      │
│                                             ▼                               │
│  ┌──────────────┐                   ┌─────────────────────────┐            │
│  │    FFN       │ ◀──────────────────│  Dummy │  P2P │ StepMesh│           │
│  │   Server     │                   └─────────────────────────┘            │
│  └──────────────┘                                                        │
│                                                                             │
│  核心特性：                                                                  │
│  1. 接口统一：不同通信后端提供相同 API                                        │
│  2. 可插拔：支持运行时切换连接器类型                                          │
│  3. 高性能：异步通信，零拷贝优化                                              │
│  4. 可扩展：插件机制支持自定义连接器                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 模块结构

```
vllm/distributed/afd_transfer/afd_connector/
├── __init__.py                    # 模块初始化
├── base.py                        # 抽象基类 AFDConnectorBase
├── metadata.py                    # 元数据定义
├── factory.py                     # 连接器工厂
├── dummy_connector.py             # Dummy 连接器（测试用）
├── p2p_connector.py               # P2P 连接器（生产环境）
└── stepmesh_connector.py          # StepMesh 连接器（大规模部署）
```

---

## 核心接口设计

### AFDConnectorBase 抽象基类

位置：`vllm/distributed/afd_transfer/afd_connector/base.py`

```python
class AFDConnectorBase(ABC):
    """AFD 连接器抽象基类

    定义了所有 AFD 连接器必须实现的核心接口。
    """

    # ========== 初始化与管理 ==========
    @abstractmethod
    def __init__(self, rank: int, local_rank: int, config: VllmConfig):
        """初始化连接器

        Args:
            rank: 全局进程 rank
            local_rank: 节点内本地 rank
            config: VllmConfig 包含 AFDConfig
        """
        pass

    @abstractmethod
    def init_afd_connector(self) -> None:
        """初始化连接器资源和通信后端"""
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭连接器，释放资源"""
        pass

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """检查连接器是否已初始化"""
        pass

    # ========== Attention → FFN 通信 ==========
    @abstractmethod
    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        metadata: AFDConnectorMetadata,
        **kwargs
    ) -> Any:
        """Attention Worker 侧：发送 Attention 输出到 FFN 服务器

        Args:
            hidden_states: Attention 层输出
            metadata: AFD 元数据（层索引、阶段索引、序列长度等）
            **kwargs: 额外参数（MoE 专家选择信息等）

        Returns:
            Any: 请求句柄，用于跟踪异步操作
        """
        pass

    @abstractmethod
    def recv_ffn_output(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        metadata: Optional[AFDConnectorMetadata] = None
    ) -> Optional[torch.Tensor]:
        """Attention Worker 侧：接收 FFN 计算结果

        Args:
            hidden_states: 可选的 hidden states（某些连接器需要）
            metadata: 可选的元数据

        Returns:
            torch.Tensor: FFN 计算结果
        """
        pass

    # ========== FFN → Attention 通信 ==========
    @abstractmethod
    def recv_attn_output(
        self,
        metadata: Optional[AFDConnectorMetadata] = None,
        **kwargs
    ) -> tuple[torch.Tensor, AFDConnectorMetadata]:
        """FFN Server 侧：接收来自 Attention Worker 的输出

        Args:
            metadata: 可选的元数据
            **kwargs: 额外参数

        Returns:
            tuple: (hidden_states, metadata)
        """
        pass

    @abstractmethod
    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        metadata: AFDConnectorMetadata,
        **kwargs
    ) -> None:
        """FFN Server 侧：发送 FFN 计算结果回 Attention Worker

        Args:
            ffn_output: FFN 计算结果
            metadata: AFD 元数据（包含序列长度用于拆分）
            **kwargs: 额外参数
        """
        pass

    # ========== MoE 特定接口 ==========
    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """选择 MoE 专家

        Args:
            hidden_states: 输入 hidden states
            router_logits: Router logits
            top_k: 选择的专家数量
            use_grouped_topk: 是否使用分组 topk
            renormalize: 是否重新归一化权重

        Returns:
            tuple: (topk_weights, topk_ids, row_idx)
        """
        raise NotImplementedError("select_experts not implemented")

    def compute_moe(
        self,
        experts: torch.nn.Module,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> Any:
        """执行 MoE 计算

        Args:
            experts: 专家网络模块
            hidden_states: 输入 hidden states
            **kwargs: 额外参数

        Returns:
            Any: MoE 计算结果
        """
        return experts.afd_ffn_compute(
            layer=experts,
            hidden_states=hidden_states,
            **kwargs
        )

    # ========== 辅助方法 ==========
    def configure_metadata(
        self,
        metadata: AFDConnectorMetadata,
        **kwargs
    ) -> None:
        """允许连接器向元数据注入特定数据

        Args:
            metadata: AFD 元数据
            **kwargs: 额外参数
        """
        pass  # 默认实现：什么都不做
```

### 接口调用流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AFDConnector 接口调用流程                            │
└─────────────────────────────────────────────────────────────────────────────┘

Attention Worker                                      FFN Server
     │                                                    │
     │ [1. 初始化阶段]                                    │ [1. 初始化阶段]
     │ connector = AFDConnectorFactory.create_connector()│ connector = ...
     │ connector.init_afd_connector()                    │ connector.init_afd_connector()
     │                                                    │
     │ [2. 前向传播 - Attention 层]                       │ [2. 等待接收]
     │ attn_output = attention_layer(hidden_states)      │
     │                                                    │
     │ [3. 创建元数据]                                    │
     │ metadata = AFDConnectorMetadata.create_...()       │
     │                                                    │
     │ [4. 发送到 FFN] ──────────────────────────────▶    │ [3. 接收 Attention 输出]
     │ connector.send_attn_output(                       │ hidden_states, metadata = \
     │     attn_output, metadata,                         │     connector.recv_attn_output()
     │     topk_weights=...,  # MoE 信息                 │
     │     topk_ids=...)                                 │
     │                                                    │
     │ [5. 继续其他计算 / 等待 FFN 结果]                  │ [4. 执行 FFN 计算]
     │                                                    │ ffn_output = ffn_layer(
     │                                                    │     hidden_states,
     │                                                    │     metadata.topk_ids)
     │                                                    │
     │ [6. 接收 FFN 结果] ◀─────────────────────────────  │ [5. 发送结果]
     │ ffn_output = connector.recv_ffn_output()          │ connector.send_ffn_output(
     │                                                    │     ffn_output, metadata)
     │                                                    │
     │ [7. 继续后续层计算]                                │ [6. 准备下一批次]
     │ hidden_states = attn_output + ffn_output          │
     │                                                    │
     │ ... 重复上述步骤 ...                               │
```

---

## 连接器实现详解

### 1. DummyAFDConnector - 测试连接器

**文件位置**: `vllm/distributed/afd_transfer/afd_connector/dummy_connector.py`

**用途**：
- 本地开发和测试
- 单元测试
- AFD 功能验证

**实现特点**：

```python
class DummyAFDConnector(AFDConnectorBase):
    """虚拟连接器，不进行实际通信"""

    def __init__(self, rank, local_rank, config):
        self.rank = rank
        self.local_rank = local_rank
        self.hidden_size = config.model_config.hf_config.hidden_size
        self.num_stages = config.afd_config.num_afd_stages
        self.events = deque(maxlen=self.num_stages)  # 存储事件

    def send_attn_output(self, hidden_states, metadata):
        """验证元数据并存储事件"""
        # 验证元数据一致性
        if not metadata.validate_tensor_shape(hidden_states.shape):
            raise ValueError("Shape mismatch")

        # 存储事件供后续 recv 使用
        self.events.append((None, metadata))

    def recv_ffn_output(self, timeout_ms=None):
        """返回零张量"""
        _, metadata = self.events.popleft()
        seq_len = metadata.seq_lens[0]
        # 返回零张量，形状与输入一致
        return torch.zeros(
            seq_len, self.hidden_size,
            dtype=metadata.dtype,
            device=metadata.device
        )

    def recv_attn_output(self, timeout_ms=None):
        """模拟多个 Attention Workers 的数据"""
        # 生成模拟数据
        dummy_seq_lens = [2, 2, 2]  # 来自不同 worker 的序列长度
        total_tokens = sum(dummy_seq_lens)

        dummy_tensor = torch.zeros(
            total_tokens, self.hidden_size,
            dtype=torch.bfloat16,
            device="cuda"
        )

        dummy_metadata = AFDConnectorMetadata.create_ffn_metadata(
            layer_idx=0,
            stage_idx=0,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
            seq_lens=dummy_seq_lens
        )

        return dummy_tensor, dummy_metadata

    def send_ffn_output(self, ffn_output, metadata):
        """验证并记录日志"""
        if not metadata.validate_tensor_shape(ffn_output.shape):
            logger.warning("Shape mismatch")

        # 模拟拆分操作（仅用于日志）
        if metadata.get_split_indices():
            split_outputs = torch.split(ffn_output, metadata.seq_lens, dim=0)
```

**使用场景**：

```python
# 配置
afd_config = AFDConfig(
    afd_connector="dummy",
    afd_role="attention"
)

# 适用场景：
# 1. 单元测试 - 验证 AFD 逻辑正确性
# 2. 开发调试 - 无需部署多节点
# 3. 功能验证 - 测试元数据传递
```

---

### 2. P2PAFDConnector - 点对点连接器

**文件位置**: `vllm/distributed/afd_transfer/afd_connector/p2p_connector.py`

**用途**：
- 小规模生产环境（2-8 节点）
- 基于 `torch.distributed` 的点对点通信
- 支持 NCCL (CUDA) 和 HCCL (Ascend NPU) 后端

**核心架构**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    P2PAFDConnector 进程组架构                                │
└─────────────────────────────────────────────────────────────────────────────┘

全局进程组 (AFD Process Group)
│
├─ Attention Workers (Ranks 0-N-1)
│  ├─ Worker 0 (Rank 0)
│  ├─ Worker 1 (Rank 1)
│  └─ Worker N-1 (Rank N-1)
│
└─ FFN Servers (Ranks N-2N-1)
   ├─ FFN 0 (Rank N)
   ├─ FFN 1 (Rank N+1)
   └─ FFN N-1 (Rank 2N-1)

每个 Worker-FFN 对形成独立的子进程组：
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  Worker 0 ───────────── a2e_group ────────────▶ FFN 0                    │
│            (Attention → Expert)              ▲                          │
│                                           │                             │
│            (Expert → Attention)              │                             │
│  Worker 0 ◀──────────── e2a_group ────────────┘ FFN 0                    │
│                                                                          │
│  说明：                                                                   │
│  - a2e_group: 用于 Worker → FFN 通信 (send_attn, recv_attn)             │
│  - e2a_group: 用于 FFN → Worker 通信 (send_ffn, recv_ffn)               │
│  - 两个组使用相同的 rank 范围，但 group_name 不同，形成独立的通信域      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**初始化流程**：

```python
def init_afd_connector(self):
    """初始化 P2P 连接器"""

    # 1. 解析配置
    afd_size = self.config.afd_config.afd_extra_config.get("afd_size")
    # afd_size 格式: "2:2" 表示 2 个 Attention, 2 个 FFN
    self.attn_size, self.ffn_size = map(
        int, re.match(r"(\d+)\D+(\d+)", afd_size).groups()
    )

    # 2. 计算全局 rank
    role = self.config.afd_config.afd_role
    world_rank = (self.rank if role == "attention"
                  else self.rank + self.attn_size)

    # 3. 创建 AFD 全局进程组
    afd_pg = init_afd_process_group(
        backend=self.backend,  # "nccl" or "hccl"
        init_method=f"tcp://127.0.0.1:29500",
        world_size=self.ffn_size + self.attn_size,
        rank=world_rank,
        group_name="afd",
        timeout=timedelta(minutes=2)
    )

    # 4. 创建子进程组（Worker-FFN 配对）
    ffn_ranks = [i for i in range(self.ffn_size, self.ffn_size + self.attn_size)]
    attn_ranks = [i for i in range(self.attn_size)]

    sub_group_ranks = []
    for i in range(len(ffn_ranks)):
        ranks = [attn_ranks[i], ffn_ranks[i]]
        sub_group_ranks.append(ranks)

    # 5. 创建 a2e 和 e2a 通信组
    self.a2e_group = init_model_parallel_group(
        sub_group_ranks, self.local_rank,
        backend=self.backend, group_name="a2e"
    )
    self.e2a_group = init_model_parallel_group(
        sub_group_ranks, self.local_rank,
        backend=self.backend, group_name="e2a"
    )
```

**核心通信方法**：

#### 2.1 异步张量字典发送

```python
def _send_tensor_dict_async(
    self,
    tensor_dict: dict[str, torch.Tensor],
    dst: int,
    process_group: GroupCoordinator
) -> list:
    """异步发送张量字典

    优化点：
    1. 元数据先发送（同步），因为元数据小且在 CPU 上
    2. 张量异步发送，使用 isend
    3. 区分 CPU 和 GPU 张量，使用不同的通信组
    """

    # 1. 分离元数据和张量
    metadata_list, tensor_list = _split_tensor_dict(tensor_dict)

    # 2. 同步发送元数据（小数据，同步开销小）
    process_group.send_object(metadata_list, dst=dst)

    # 3. 异步发送张量
    work_list = []
    group = process_group.device_group      # GPU 张量
    metadata_group = process_group.cpu_group # CPU 张量

    for tensor in tensor_list:
        if tensor.numel() == 0:
            continue  # 跳过空张量

        if tensor.is_cpu:
            work = torch.distributed.isend(
                tensor,
                dst=process_group.ranks[dst],
                group=metadata_group
            )
        else:
            work = torch.distributed.isend(
                tensor,
                dst=process_group.ranks[dst],
                group=group
            )
        work_list.append(work)

    return work_list  # 返回 work 对象，可用于等待
```

#### 2.2 异步张量字典接收

```python
def _recv_tensor_dict_async(
    self,
    src: int,
    process_group: GroupCoordinator
) -> tuple[dict[str, torch.Tensor | Any], list]:
    """异步接收张量字典

    流程：
    1. 先同步接收元数据（需要知道张量形状和类型）
    2. 根据元数据创建空张量
    3. 异步接收张量数据
    """

    # 1. 同步接收元数据
    recv_metadata_list = process_group.recv_object(src=src)

    # 2. 准备接收
    tensor_dict = {}
    work_list = []
    group = process_group.device_group
    metadata_group = process_group.cpu_group

    for key, value in recv_metadata_list:
        if isinstance(value, TensorMetadata):
            # 根据元数据创建空张量
            tensor = torch.empty(
                value.size,
                dtype=value.dtype,
                device=value.device
            )

            if tensor.numel() == 0:
                tensor_dict[key] = tensor
                continue

            # 异步接收
            if tensor.is_cpu:
                work = torch.distributed.irecv(
                    tensor,
                    src=process_group.ranks[src],
                    group=metadata_group
                )
            else:
                work = torch.distributed.irecv(
                    tensor,
                    src=process_group.ranks[src],
                    group=group
                )
            work_list.append(work)
            tensor_dict[key] = tensor
        else:
            # 非张量值直接添加
            tensor_dict[key] = value

    return tensor_dict, work_list
```

#### 2.3 Attention → FFN 通信

```python
def send_attn_output(
    self,
    hidden_states: torch.Tensor,
    metadata: AFDConnectorMetadata,
    **kwargs
):
    """Attention Worker 发送数据到 FFN"""

    # 1. 创建中间张量字典
    intermediate_tensors = self.create_intermediate_tensors(
        backend=self.backend,
        hidden_states=hidden_states,
        **kwargs
    )

    # 2. 同步流
    self.current_stream_synchronize(self.backend)

    # 3. 计算目标 rank（环形拓扑）
    dst = (self.a2e_group.rank_in_group + 1) % self.a2e_group.world_size

    # 4. 异步发送张量字典
    work_list = self._send_tensor_dict_async(
        intermediate_tensors.tensors,
        dst=dst,
        process_group=self.a2e_group,
    )

    # 5. 发送元数据
    self.a2e_group.send_object(metadata, dst)

    # 6. 存储句柄供后续等待
    if metadata is not None:
        metadata.send_handle_list = work_list

def recv_attn_output(self) -> Any:
    """FFN Server 接收 Attention 输出"""

    # 1. 计算源 rank
    src = (self.a2e_group.rank_in_group - 1) % self.a2e_group.world_size

    # 2. 异步接收张量字典
    intermediate_tensors, work_list = self._recv_tensor_dict_async(
        src=src,
        process_group=self.a2e_group,
    )

    # 3. 接收元数据
    metadata = self.a2e_group.recv_object(src)
    metadata.recv_handle_list = work_list

    # 4. 返回结果
    return AFDRecvOutput(
        hidden_states=intermediate_tensors["hidden_states"],
        metadata=metadata,
        router_logits=intermediate_tensors.get("router_logits"),
        topk_weights=intermediate_tensors.get("topk_weights"),
        topk_ids=intermediate_tensors.get("topk_ids"),
        row_idx=intermediate_tensors.get("row_idx")
    )
```

#### 2.4 FFN → Attention 通信

```python
def send_ffn_output(
    self,
    hidden_states: torch.Tensor,
    metadata: AFDConnectorMetadata
):
    """FFN Server 发送结果回 Attention Worker"""

    # 1. 创建中间张量
    intermediate_tensors = IntermediateTensors({
        "hidden_states": hidden_states,
    })

    # 2. 同步流
    self.current_stream_synchronize(self.backend)

    # 3. 计算目标 rank
    dst = (self.e2a_group.rank_in_group + 1) % self.e2a_group.world_size

    # 4. 异步发送
    work_list = self._send_tensor_dict_async(
        intermediate_tensors.tensors,
        dst=dst,
        process_group=self.e2a_group,
    )

    # 5. 发送元数据
    self.e2a_group.send_object(metadata, dst)

    if metadata is not None:
        metadata.send_handle_list = work_list

def recv_ffn_output(self) -> torch.Tensor:
    """Attention Worker 接收 FFN 结果"""

    # 1. 计算源 rank
    src = (self.e2a_group.rank_in_group - 1) % self.e2a_group.world_size

    # 2. 异步接收张量字典
    intermediate_tensors, work_list = self._recv_tensor_dict_async(
        src=src,
        process_group=self.e2a_group,
    )

    # 3. 接收元数据
    metadata = self.e2a_group.recv_object(src)
    metadata.recv_handle_list = work_list

    # 4. 返回 hidden_states
    return intermediate_tensors["hidden_states"], metadata
```

**支持的后端**：

| 后端 | 硬件平台 | 特点 |
|------|---------|------|
| **nccl** | NVIDIA GPU | 高性能，支持 RDMA |
| **hccl** | 华为 Ascend NPU | 支持 MoE gate 计算 |
| **gloo** | CPU/GPU | 灵活，性能较低 |

**使用示例**：

```python
# 配置
afd_config = AFDConfig(
    afd_connector="p2pconnector",
    afd_role="attention",
    afd_extra_config={"afd_size": "4:4"}  # 4 Attention, 4 FFN
)

# 启动
# Attention Workers (Ranks 0-3)
python -m vllm.entrypoints.cli.main \
    --model deepseek-ai/deepseek-v2 \
    --afd-config '{"afd_role": "attention", "afd_connector": "p2pconnector"}'

# FFN Servers (Ranks 4-7)
python -m vllm.entrypoints.cli.fserver \
    --model deepseek-ai/deepseek-v2 \
    --afd-config '{"afd_role": "ffn", "afd_connector": "p2pconnector"}'
```

---

### 3. StepMeshAFDConnector - 参数服务器连接器

**文件位置**: `vllm/distributed/afd_transfer/afd_connector/stepmesh_connector.py`

**用途**：
- 大规模生产环境（8+ 节点）
- 基于 StepMesh 参数服务器框架
- 支持 RDMA，高性能网络优化

**核心架构**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    StepMeshAFDConnector 架构                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                              StepMesh 集群                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Scheduler (调度器)                            │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │  - 管理通信拓扑                                             │  │   │
│  │  │  - 分配通信 ID                                             │  │   │
│  │  │  - 协调 push_pull 操作                                     │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ▲ │                                         │
│                              │ │ push_pull                              │
│  ┌───────────────────────────┼─┼──────────────────────────────────┐    │
│  │                           │ │                                    │    │
│  │  Attention Workers        │ │  FFN Servers                      │    │
│  │  (DMLC_ROLE=worker)       │ │  (DMLC_ROLE=server)               │    │
│  │                           │ │                                    │    │
│  │  ┌─────────────────────┐  │ │  ┌─────────────────────────────┐  │    │
│  │  │ Worker 0            │  │ │  │ Server 0                    │  │    │
│  │  │ - Rank: 0           │  │ │  │ - Rank: 0                   │  │    │
│  │  │ - GPU: 0            │  │ │  │ - GPU: 0                    │  │    │
│  │  │ send_key: 0XXXXXX   │──┼──┼─▶│ recv_key: 1000             │  │    │
│  │  │ recv_key: 1000      │◀─┼──┼──│ send_key: 0XXXXXX           │  │    │
│  │  └─────────────────────┘  │ │  └─────────────────────────────┘  │    │
│  │                           │ │                                    │    │
│  │  ┌─────────────────────┐  │ │  ┌─────────────────────────────┐  │    │
│  │  │ Worker 1            │  │ │  │ Server 1                    │  │    │
│  │  │ - Rank: 1           │  │ │  │ - Rank: 1                   │  │    │
│  │  │ - GPU: 1            │  │ │  │ - GPU: 1                    │  │    │
│  │  │ send_key: 1XXXXXX   │──┼──┼─▶│ recv_key: 1000             │  │    │
│  │  │ recv_key: 1000      │◀─┼──┼──│ send_key: 1XXXXXX           │  │    │
│  │  └─────────────────────┘  │ │  └─────────────────────────────┘  │    │
│  │                           │ │                                    │    │
│  │  ...                       │ │  ...                              │    │
│  │                           │ │                                    │    │
│  └───────────────────────────┼─┼──────────────────────────────────┘    │
│                              │ │                                         │
└──────────────────────────────┼─┼─────────────────────────────────────────┘
                               │ │
                    ┌──────────┼─┼──────────┐
                    │          ▼ ▼          │
                    │   Parameter Server     │
                    │   (StepMesh PS)        │
                    │   - 管理张量存储       │
                    │   - 处理 push_pull    │
                    │   - RDMA 优化         │
                    └────────────────────────┘

Key 命名约定：
- send_key = stage_id + node_rank * 1,000,000
- recv_key = stage_id + 1000
  例如：Worker 0, Stage 2
    - send_key = 2 + 0 * 1,000,000 = 2
    - recv_key = 2 + 1000 = 1002
```

**环境变量配置**：

```python
def _setup_stepmesh_env(self):
    """配置 StepMesh 环境变量"""

    if self.afd_config.afd_role == "attention":
        os.environ["DMLC_ROLE"] = "worker"
    elif self.afd_config.afd_role == "ffn":
        os.environ["DMLC_ROLE"] = "server"

    # 进程组配置
    os.environ["DMLC_NUM_WORKER"] = str(self.afd_config.num_attention_servers)
    os.environ["DMLC_NUM_SERVER"] = str(self.afd_config.num_ffn_servers)

    # RDMA 配置
    os.environ["DMLC_ENABLE_RDMA"] = "ibverbs"
    os.environ["DMLC_INTERFACE"] = "auto"

    # CPU 绑定配置
    os.environ["STEPMESH_BIND_CPU_CORE"] = "1"
    os.environ["STEPMESH_GPU"] = str(self.local_rank)

    # 网络配置
    os.environ["DMLC_PS_ROOT_PORT"] = str(self.afd_config.afd_port)
    os.environ["DMLC_PS_ROOT_URI"] = self.afd_config.afd_host
    os.environ["DMLC_NODE_HOST"] = self.afd_config.afd_host

    # 调度器配置
    os.environ["SCHEDULER_IP"] = self.afd_config.afd_host

    # 进程 rank
    os.environ["DMLC_NODE_RANK"] = str(self.afd_config.afd_server_rank)
    os.environ["DMLC_GROUP_SIZE"] = str(self.world_size)
```

**初始化流程**：

```python
def init_afd_connector(self):
    """初始化 StepMesh 连接器"""

    # 1. 设置环境变量
    self._setup_stepmesh_env()

    # 2. FFN Server Rank 0 启动调度器进程
    if (self.afd_config.afd_role == "ffn"
        and self.afd_config.afd_server_rank == 0
        and self.local_rank == 0):
        self._start_scheduler_process()

    # 3. 初始化 StepMesh
    ps.init()  # fserver_lib.init()

    # 4. 分配缓冲区
    if self.afd_config.afd_role == "attention":
        # Attention Worker: 为每个阶段分配发送和接收缓冲区
        self.max_num_tokens = (
            config.scheduler_config.max_num_batched_tokens // self.num_stages
        )

        self.recv_buffer = [[
            torch.empty(
                (self.max_num_tokens, hidden_size),
                dtype=torch.bfloat16,
                device="cuda"
            ).contiguous()
            for _ in range(self.num_recv_times)
        ] for _ in range(self.num_afd_stages)]

        self.send_buffer = [
            torch.empty(
                (self.max_num_tokens, hidden_size),
                dtype=torch.bfloat16,
                device="cuda"
            ).contiguous()
            for _ in range(self.num_afd_stages)
        ]

        # 事件队列（用于跟踪异步操作）
        self.events = deque(maxlen=self.num_stages)

    else:  # FFN Server
        # FFN Server: 分配合并后的返回缓冲区
        self.max_num_tokens = (
            config.scheduler_config.max_num_batched_tokens // self.num_stages
        ) * self.num_recv_times

        self.ret_buffer = torch.empty(
            (self.max_num_tokens, hidden_size),
            dtype=torch.bfloat16,
            device="cuda"
        ).contiguous()
```

**核心通信方法**：

#### 3.1 Attention → FFN (push_pull)

```python
def send_attn_output(
    self,
    hidden_states: torch.Tensor,
    metadata: AFDConnectorMetadata
):
    """Attention Worker 通过 push_pull 发送数据"""

    # 1. 验证
    if not metadata.validate_tensor_shape(hidden_states.shape):
        raise ValueError("Shape mismatch")

    if not metadata.is_single_sequence:
        raise ValueError("Attention side should have single sequence")

    # 2. 提取参数
    seq_len = metadata.seq_lens[0]
    stage_id = metadata.stage_idx

    # 3. 计算 key
    node_rank_offset = int(self.rank * 1e6)
    recv_key = [stage_id + 1000]
    send_key = [stage_id + node_rank_offset]

    # 4. 准备缓冲区
    recv_buff = [t[:seq_len] for t in self.recv_buffer[stage_id]]
    send_buff = [self.send_buffer[stage_id][:seq_len]]

    # 5. 拷贝数据到发送缓冲区
    send_buff[0].copy_(hidden_states[:seq_len])

    # 6. 执行 push_pull（异步）
    event = ps.push_pull(
        send_buff,    # 发送缓冲区
        send_key,     # 发送 key
        recv_buff,    # 接收缓冲区
        recv_key,     # 接收 key
    )

    # 7. 存储事件供后续等待
    self.events.append((event, metadata))

    return event

def recv_attn_output(self) -> tuple[torch.Tensor, AFDConnectorMetadata]:
    """FFN Server 通过 get_batch 接收数据"""

    # 1. 从调度器获取批次
    batches = ps.get_batch()

    # 2. 提取张量和元数据
    recv_tensors = []
    seq_lens = []
    comm_handles = []

    for node_rank in range(self.num_recv_times):
        tensor = batches[node_rank][1][0]
        comm_id = batches[node_rank][0]

        recv_tensors.append(tensor)
        seq_lens.append(tensor.shape[0])
        comm_handles.append(comm_id)

    # 3. 合并张量
    merged_tensor = torch.cat(recv_tensors, dim=0)

    # 4. 创建元数据
    inferred_metadata = AFDConnectorMetadata.create_ffn_metadata(
        layer_idx=-1,  # TODO: 从 comm_id 提取
        stage_idx=-1,
        seq_lens=seq_lens,
        dtype=merged_tensor.dtype,
        device=merged_tensor.device,
        request_id=f"ffn_batch_{time.time()}"
    )

    # 5. 存储句柄用于响应
    self._current_comm_handles = comm_handles
    self._current_metadata = inferred_metadata

    return merged_tensor, inferred_metadata
```

#### 3.2 FFN → Attention (respond)

```python
def send_ffn_output(
    self,
    ffn_output: torch.Tensor,
    metadata: AFDConnectorMetadata
):
    """FFN Server 通过 respond_vec 发送结果"""

    # 1. 拷贝到返回缓冲区
    self.ret_buffer[:ffn_output.shape[0]].copy_(ffn_output)

    # 2. 根据序列长度拆分
    split_indices = metadata.get_split_indices()
    if split_indices:
        split_outputs = torch.split(ffn_output, metadata.seq_lens, dim=0)
    else:
        split_outputs = [ffn_output]

    # 3. 发送响应
    ps.respond_vec(
        self.ret_buffer,      # 完整缓冲区
        split_outputs,        # 拆分后的输出
        self._current_comm_handles  # 通信句柄
    )

def recv_ffn_output(self) -> torch.Tensor:
    """Attention Worker 等待并接收 FFN 结果"""

    # 1. 获取事件
    if len(self.events) > 0:
        event, metadata = self.events.popleft()

        # 2. 等待 push_pull 完成
        ps.wait(event, timeout_ms=50000)

    # 3. 从接收缓冲区提取结果
    if metadata:
        stage_idx = metadata.stage_idx
        seq_len = metadata.seq_lens[0]

        if len(self.recv_buffer[stage_idx]) == 1:
            return self.recv_buffer[stage_idx][0][:seq_len]
        else:
            # 多个 FFN 结果需要合并
            return torch.stack(
                [t[:seq_len] for t in self.recv_buffer[stage_idx]],
                dim=0
            ).sum(dim=0)
```

**性能优化**：

1. **零拷贝优化**：使用预分配缓冲区，避免动态分配
2. **RDMA 支持**：通过 `DMLC_ENABLE_RDMA=ibverbs` 启用
3. **CPU 绑定**：通过 `STEPMESH_BIND_CPU_CORE=1` 绑定通信线程
4. **流水线并行**：多个 stage 可以并发执行 push_pull
5. **批量处理**：`get_batch` 一次性接收多个 worker 的数据

**使用示例**：

```python
# 配置
afd_config = AFDConfig(
    afd_connector="stepmesh",
    afd_role="attention",
    afd_host="stepmesh-server",
    afd_port=1239,
    num_afd_stages=3,
    num_attention_servers=4,
    num_ffn_servers=4,
    afd_server_rank=0
)

# 启动（需要先启动 StepMesh 服务器）
# Attention Workers
for rank in range(4):
    python -m vllm.entrypoints.cli.main \
        --model deepseek-ai/deepseek-v2 \
        --afd-config f'{{"afd_role": "attention", "afd_server_rank": {rank}}}' &

# FFN Servers
for rank in range(4):
    python -m vllm.entrypoints.cli.fserver \
        --model deepseek-ai/deepseek-v2 \
        --afd-config f'{{"afd_role": "ffn", "afd_server_rank": {rank}}}' &
```

---

## 工厂模式与插件机制

### AFDConnectorFactory

位置：`vllm/distributed/afd_transfer/afd_connector/factory.py`

```python
class AFDConnectorFactory:
    """AFD 连接器工厂

    负责：
    1. 连接器注册（延迟加载）
    2. 连接器创建
    3. 插件加载
    """

    # 连接器注册表（延迟加载）
    _registry: dict[str, Callable[[], type[AFDConnectorBase]]] = {}
    _plugins_loaded: bool = False

    @classmethod
    def register_connector(
        cls,
        name: str,
        module_path: str,
        class_name: str
    ) -> None:
        """注册连接器（延迟加载）

        Args:
            name: 连接器名称（如 "p2pconnector"）
            module_path: 模块路径（如 "vllm....p2p_connector"）
            class_name: 类名（如 "P2PAFDConnector"）
        """
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' already registered")

        def loader() -> type[AFDConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(
        cls,
        rank: int,
        local_rank: int,
        config: VllmConfig
    ) -> AFDConnectorBase:
        """创建连接器实例

        Args:
            rank: 全局 rank
            local_rank: 本地 rank
            config: VllmConfig

        Returns:
            AFDConnectorBase: 连接器实例
        """
        cls.load_plugins()

        afd_config = config.afd_config
        connector_name = afd_config.afd_connector

        if connector_name not in cls._registry:
            raise ValueError(f"Unsupported connector: {connector_name}")

        # 延迟加载：只在需要时导入模块
        connector_cls = cls._registry[connector_name]()
        assert issubclass(connector_cls, AFDConnectorBase)

        return connector_cls(rank, local_rank, config)

    @classmethod
    def load_plugins(cls):
        """从 entry points 加载插件

        允许第三方扩展自定义连接器
        """
        if cls._plugins_loaded:
            return

        cls._plugins_loaded = True

        # 加载 vllm.afd_connectors entry points
        if sys.version_info < (3, 10):
            from importlib.metadata import entry_points
            eps = entry_points()
            plugin_eps = eps.get("vllm.afd_connectors", [])
        else:
            from importlib.metadata import entry_points
            plugin_eps = entry_points(group="vllm.afd_connectors")

        for entry_point in plugin_eps:
            try:
                register_func = entry_point.load()
                register_func()
                logger.info(f"Loaded AFD connector plugin: {entry_point.name}")
            except Exception as e:
                logger.warning(
                    f"Failed to load plugin {entry_point.name}: {e}"
                )


# 内置连接器注册
AFDConnectorFactory.register_connector(
    "stepmesh",
    "vllm.distributed.afd_transfer.afd_connector.stepmesh_connector",
    "StepMeshAFDConnector"
)

AFDConnectorFactory.register_connector(
    "dummy",
    "vllm.distributed.afd_transfer.afd_connector.dummy_connector",
    "DummyAFDConnector"
)

AFDConnectorFactory.register_connector(
    "p2pconnector",
    "vllm.distributed.afd_transfer.afd_connector.p2p_connector",
    "P2PAFDConnector"
)
```

### 自定义连接器插件

**步骤 1：实现连接器**

```python
# my_connector.py
from vllm.distributed.afd_transfer.afd_connector.base import AFDConnectorBase
from vllm.distributed.afd_transfer.afd_connector.metadata import AFDConnectorMetadata
import torch

class MyCustomAFDConnector(AFDConnectorBase):
    def __init__(self, rank, local_rank, config):
        self.rank = rank
        self.local_rank = local_rank
        self.config = config
        self._initialized = False

    def init_afd_connector(self):
        # 自定义初始化逻辑
        self._initialized = True

    def close(self):
        self._initialized = False

    @property
    def is_initialized(self):
        return self._initialized

    def send_attn_output(self, hidden_states, metadata, **kwargs):
        # 自定义发送逻辑
        pass

    def recv_ffn_output(self, hidden_states=None, metadata=None):
        # 自定义接收逻辑
        return torch.zeros_like(hidden_states)

    def recv_attn_output(self, metadata=None, **kwargs):
        # 自定义接收逻辑
        dummy_tensor = torch.zeros(100, 768)
        dummy_metadata = AFDConnectorMetadata(...)
        return dummy_tensor, dummy_metadata

    def send_ffn_output(self, ffn_output, metadata, **kwargs):
        # 自定义发送逻辑
        pass
```

**步骤 2：注册插件**

```python
# my_connector_plugin.py
from vllm.distributed.afd_transfer.afd_connector.factory import (
    AFDConnectorFactory
)

def register_my_connector():
    AFDConnectorFactory.register_connector(
        "my_connector",
        "my_connector",
        "MyCustomAFDConnector"
    )
```

**步骤 3：配置 entry point**

```python
# setup.py
setup(
    name="vllm-my-connector",
    packages=["my_connector"],
    entry_points={
        "vllm.afd_connectors": [
            "my_connector = my_connector_plugin:register_my_connector"
        ]
    }
)
```

**步骤 4：使用自定义连接器**

```python
afd_config = AFDConfig(
    afd_connector="my_connector",  # 使用自定义连接器
    afd_role="attention"
)
```

---

## 元数据管理

### AFDConnectorMetadata

位置：`vllm/distributed/afd_transfer/afd_connector/metadata.py`

```python
@dataclass
class AFDConnectorMetadata:
    """AFD 通信元数据

    设计原则：
    1. 轻量级：只包含必要的通信信息
    2. 可序列化：支持跨进程传输
    3. 类型安全：使用 dataclass 确保类型正确
    """

    # ========== 基础信息 ==========
    layer_idx: int                    # Transformer 层索引
    stage_idx: int                    # 流水线阶段索引
    seq_lens: list[int]               # 序列长度列表（支持变长）
    dtype: torch.dtype                # 数据类型
    device: torch.device              # 设备
    num_ubatches: int = 1             # 微批数量

    # ========== 连接器特定数据 ==========
    connector_data: Optional["AFDConnectorData"] = None

    # ========== MoE 相关 ==========
    topk_idx: Optional[torch.Tensor] = None      # 专家选择索引
    topk_weights: Optional[torch.Tensor] = None  # 专家权重
    topk_ids: Optional[torch.Tensor] = None      # 专家 ID
    row_idx: Optional[torch.Tensor] = None       # 行索引
    moe_expert_num: Optional[int] = None         # MoE 专家数量
    shared_expert_num: Optional[int] = None      # 共享专家数量

    # ========== 量化相关 ==========
    scale: Optional[torch.Tensor] = None         # 量化缩放因子

    # ========== 通信句柄 ==========
    send_handle_list: Optional[list[Any]] = None  # 发送句柄
    recv_handle_list: Optional[list[Any]] = None  # 接收句柄

    # ========== 调试信息 ==========
    request_id: Optional[str] = None             # 请求 ID
    timestamp: Optional[float] = None            # 时间戳

    # ========== 验证与辅助方法 ==========
    def __post_init__(self):
        """验证数据一致性"""
        if not self.seq_lens:
            raise ValueError("seq_lens cannot be empty")
        if any(length <= 0 for length in self.seq_lens):
            raise ValueError("All sequence lengths must be positive")

    @property
    def total_tokens(self) -> int:
        """总 token 数量"""
        return sum(self.seq_lens)

    @property
    def num_sequences(self) -> int:
        """序列数量"""
        return len(self.seq_lens)

    @property
    def is_single_sequence(self) -> bool:
        """是否为单个序列（Attention 侧特征）"""
        return len(self.seq_lens) == 1

    @property
    def is_multi_sequence(self) -> bool:
        """是否为多个序列（FFN 侧特征）"""
        return len(self.seq_lens) > 1

    def get_split_indices(self) -> list[int]:
        """获取张量拆分索引（FFN 侧使用）"""
        if len(self.seq_lens) <= 1:
            return []

        indices = []
        cumsum = 0
        for length in self.seq_lens[:-1]:
            cumsum += length
            indices.append(cumsum)
        return indices

    def validate_tensor_shape(self, tensor_shape: tuple[int, ...]) -> bool:
        """验证张量形状是否与元数据一致"""
        if len(tensor_shape) < 1:
            return False
        return tensor_shape[0] == self.total_tokens

    def to_dict(self) -> dict:
        """转换为字典（用于序列化和调试）"""
        return {
            "layer_idx": self.layer_idx,
            "stage_idx": self.stage_idx,
            "seq_lens": self.seq_lens,
            "dtype": self.dtype,
            "device": self.device,
            "total_tokens": self.total_tokens,
            "num_sequences": self.num_sequences,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
        }

    # ========== 工厂方法 ==========
    @classmethod
    def create_attention_metadata(
        cls,
        layer_idx: int,
        stage_idx: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        num_ubatches: int = 1,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        row_idx: Optional[torch.Tensor] = None,
        **kwargs
    ) -> "AFDConnectorMetadata":
        """创建 Attention 侧元数据（单个序列）"""
        return cls(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=[seq_len],  # 单个序列
            dtype=dtype,
            device=device,
            num_ubatches=num_ubatches,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            row_idx=row_idx,
        )

    @classmethod
    def create_ffn_metadata(
        cls,
        layer_idx: int,
        stage_idx: int,
        seq_lens: list[int],
        dtype: torch.dtype,
        device: torch.device,
        request_id: Optional[str] = None
    ) -> "AFDConnectorMetadata":
        """创建 FFN 侧元数据（多个序列）"""
        return cls(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=seq_lens.copy(),  # 防止外部修改
            dtype=dtype,
            device=device,
            request_id=request_id,
            timestamp=time.time()
        )
```

### 元数据流转

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         元数据流转示意图                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Attention Worker                                      FFN Server
     │                                                    │
     │ [1. 创建元数据]                                    │
     │ metadata = AFDConnectorMetadata.create_...()       │
     │ ┌─────────────────────────────────────────┐       │
     │ │ layer_idx: 0                            │       │
     │ │ stage_idx: 0                            │       │
     │ │ seq_lens: [1024]  # 单个序列             │       │
     │ │ dtype: torch.bfloat16                   │       │
     │ │ device: cuda:0                          │       │
     │ │ topk_weights: [0.1, 0.2, ...]  # MoE    │       │
     │ │ topk_ids: [5, 12, ...]                  │       │
     │ └─────────────────────────────────────────┘       │
     │                                                    │
     │ [2. 附加 MoE 信息]                                │
     │ metadata.topk_weights = topk_weights              │
     │ metadata.topk_ids = topk_ids                      │
     │                                                    │
     │ [3. 发送数据和元数据] ───────────────────────▶    │ [4. 接收数据和元数据]
     │ connector.send_attn_output(                       │ recv_tensors, seq_lens = \
     │     hidden_states,                                │     extract_from_comm()
     │     metadata)                                     │
     │                                                    │
     │                                                    │ [5. 创建 FFN 侧元数据]
     │                                                    │ ffn_metadata = AFDConnectorMetadata(
     │                                                    │     layer_idx=0,
     │                                                    │     stage_idx=0,
     │                                                    │     seq_lens=[512, 256, 256]  # 合并
     │                                                    │ )
     │                                                    │
     │                                                    │ [6. 使用元数据执行计算]
     │                                                    │ ffn_output = ffn_layers(
     │                                                    │     hidden_states,
     │                                                    │     metadata.topk_ids,  # MoE
     │                                                    │     metadata.seq_lens   # 拆分
     │                                                    │ )
     │                                                    │
     │ [7. 接收 FFN 结果和元数据] ◀───────────────────── │ [8. 发送结果和元数据]
     │ ffn_output, metadata = connector.recv_ffn_output()│ connector.send_ffn_output(
     │                                                    │     ffn_output,
     │                                                    │     metadata)
     │                                                    │
     │ [9. 使用元数据处理结果]                           │
     │ if metadata.seq_lens:                             │
     │     # 根据 seq_lens 拆分结果                       │
     │     outputs = torch.split(ffn_output,             │
     │                          metadata.seq_lens)       │
```

---

## 通信协议详解

### 协议分层

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AFD 通信协议分层                                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 应用层 (Application Layer)                                                  │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ AFDConnectorBase.send_attn_output() / recv_attn_output()               │ │
│ │ AFDConnectorBase.send_ffn_output() / recv_ffn_output()                 │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 中间层 (Middleware Layer)                                                   │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ 元数据管理 (AFDConnectorMetadata)                                      │ │
│ │ 张量字典序列化 (_split_tensor_dict / _join_tensor_dict)                │ │
│ │ 句柄管理 (send_handle_list / recv_handle_list)                         │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 传输层 (Transport Layer)                                                    │
│ ┌───────────────────┬───────────────────┬─────────────────────────────────┐ │
│ │  Dummy Connector  │  P2P Connector    │  StepMesh Connector            │ │
│ │  (内存拷贝)       │  (torch.dist)     │  (push_pull)                  │ │
│ │                   │                   │                                │ │
│ │  - deque         │  - isend/irecv    │  - ps.push_pull()             │ │
│ │  - zeros()       │  - NCCL/HCCL      │  - ps.get_batch()             │ │
│ │                   │  - 进程组         │  - ps.respond_vec()           │ │
│ └───────────────────┴───────────────────┴─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 网络层 (Network Layer)                                                      │
│ ┌───────────────────┬───────────────────┬─────────────────────────────────┐ │
│ │  N/A              │  NCCL/HCCL        │  RDMA (InfiniBand)            │ │
│ │                   │  TCP              │  Socket                       │ │
│ └───────────────────┴───────────────────┴─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 通信模式对比

| 特性 | Dummy | P2P | StepMesh |
|------|-------|-----|----------|
| **通信方式** | 内存拷贝（deque） | 点对点（isend/irecv） | 参数服务器（push_pull） |
| **拓扑** | 无拓扑 | 环形/全连接 | 星形（通过 PS） |
| **延迟** | 极低（μs 级） | 低（ms 级） | 中（ms 级） |
| **吞吐** | 无限制 | 高 | 很高 |
| **扩展性** | 1 节点 | 2-8 节点 | 8+ 节点 |
| **容错** | N/A | 进程组级别 | PS 级别 |
| **配置复杂度** | 低 | 中 | 高 |
| **适用场景** | 测试、开发 | 小规模生产 | 大规模生产 |

### 通信时序

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      单次前向传播的通信时序                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Attention Worker                                      FFN Server
     │                                                    │
     │ T0: 开始 Attention 层计算                           │
     ├─▶ attention_start = time.time()                    │
     │                                                    │
     │ T1: Attention 计算完成                              │
     ├─▶ attn_output = attention_layer(hidden_states)     │
     │                                                    │
     │ T2: 创建元数据                                      │
     ├─▶ metadata = AFDConnectorMetadata(...)             │
     │                                                    │
     │ T3: 发送 Attention 输出 ──────────────────────────▶ │ T4: 接收 Attention 输出
     │ connector.send_attn_output(                        │ hidden_states, metadata = \
     │     attn_output, metadata)                         │     connector.recv_attn_output()
     │                                                    │
     │ [通信中...]                                        │ T5: FFN 计算
     │                                                    │ ffn_output = ffn_layer(
     │                                                    │     hidden_states,
     │                                                    │     metadata.topk_ids)
     │                                                    │
     │                                                    │ T6: 发送 FFN 输出
     │                                                    │ connector.send_ffn_output(
     │                                                    │     ffn_output, metadata)
     │                                                    │
     │ T7: 接收 FFN 输出 ◀─────────────────────────────── │
     │ ffn_output = connector.recv_ffn_output()           │
     │                                                    │
     │ T8: 继续后续计算                                   │
     ├─▶ hidden_states = attn_output + ffn_output        │
     │                                                    │
     时序分析：                                                  │
     - 计算时间：T1 - T0 (Attention) + T5 - T4 (FFN)           │
     - 通信延迟：T7 - T3 + T6 - T4 (往返)                        │
     - 总时间：T8 - T0                                          │
```

---

## 节点连接拓扑

### 1:1 连接拓扑

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         1:1 连接拓扑（最简单）                                │
└─────────────────────────────────────────────────────────────────────────────┘

Attention Worker 0 ─────────────────────▶ FFN Server 0
                      send_attn_output
                      recv_ffn_output

                      ◀───────────────────
                      recv_attn_output
                      send_ffn_output

配置：
num_attention_servers = 1
num_ffn_servers = 1

特点：
- 最简单的拓扑
- 无需负载均衡
- 适合单机双进程部署
```

### N:N 连接拓扑

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         N:N 连接拓扑（P2P）                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Attention Worker 0 ─────────────────────▶ FFN Server 0
Attention Worker 1 ─────────────────────▶ FFN Server 1
Attention Worker 2 ─────────────────────▶ FFN Server 2
Attention Worker 3 ─────────────────────▶ FFN Server 3

配置：
num_attention_servers = 4
num_ffn_servers = 4

P2P 连接器子进程组：
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  Worker 0 ───────────── a2e_group ────────────▶ FFN 0                    │
│            (Attention → Expert)              ▲                          │
│                                           │                             │
│            (Expert → Attention)              │                             │
│  Worker 0 ◀──────────── e2a_group ────────────┘ FFN 0                    │
│                                                                          │
│  Worker 1 ───────────── a2e_group ────────────▶ FFN 1                    │
│            ...                              ...                          │
│  Worker 1 ◀──────────── e2a_group ────────────┘ FFN 1                    │
│                                                                          │
│  Worker 2 ───────────── a2e_group ────────────▶ FFN 2                    │
│            ...                              ...                          │
│  Worker 2 ◀──────────── e2e_group ────────────┘ FFN 2                    │
│                                                                          │
│  Worker 3 ───────────── a2e_group ────────────▶ FFN 3                    │
│            ...                              ...                          │
│  Worker 3 ◀──────────── e2a_group ────────────┘ FFN 3                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

特点：
- 每个 Worker 连接到对应的 FFN
- 点对点通信，无中心节点
- 适合同构集群
```

### M:N 连接拓扑

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         M:N 连接拓扑（StepMesh）                             │
└─────────────────────────────────────────────────────────────────────────────┘

Attention Workers                StepMesh Parameter Server              FFN Servers
(DMLC_ROLE=worker)                           ▲                  (DMLC_ROLE=server)
    │                                        │                            │
    │ 1. push_pull(send_key, recv_key)       │                            │
    ├─────────────────────────────────────────┼────────────────────────────▶ │
    │                                        │                        2. get_batch()
    │                                        │                            │
    │                                        │                            │
    │ 4. recv_buffer ◀────────────────────────┼─────────────────────────── │
    │                                        │                            │
    │                                        │                            │ 3. respond_vec()
    │                                        │                     ◀───────┘
    │                                        │

Key 分配：
- Attention Worker i 的 send_key = stage_id + i * 1,000,000
- FFN Server 的 recv_key = stage_id + 1000

示例（4 Attention, 2 FFN, Stage 2）：
┌─────────────────────┐       push_pull       ┌──────────────────────────┐
│ Worker 0           │ ──────────────────────▶│                        │
│ send_key: 2000000  │                       │   Parameter Server      │
│ recv_key: 1002     │ ◀──────────────────────│   - 聚合所有 worker     │
└─────────────────────┘                       │   - 分发给所有 server   │
                                              │                        │
┌─────────────────────┐                       └──────────────────────────┘
│ Worker 1           │                                    │
│ send_key: 2000001  │                                    │
│ recv_key: 1002     │                                    ▼
└─────────────────────┘                       ┌──────────────────────────┐
│ Worker 2           │                       │ Server 0                │
│ send_key: 2000002  │                       │ recv_key: 1002          │
│ recv_key: 1002     │                       │ - 接收所有 worker 数据   │
└─────────────────────┘                       │ - 计算后返回            │
┌─────────────────────┐                       └──────────────────────────┘
│ Worker 3           │                       ┌──────────────────────────┐
│ send_key: 2000003  │                       │ Server 1                │
│ recv_key: 1002     │                       │ recv_key: 1002          │
└─────────────────────┘                       │ - 接收所有 worker 数据   │
                                              │ - 计算后返回            │
                                              └──────────────────────────┘

特点：
- M:N 拓扑（M 个 Worker，N 个 Server）
- 中心化参数服务器
- 支持 RDMA
- 适合大规模异构集群
```

### 负载均衡

**StepMesh 负载均衡**：

```python
# StepMesh 自动处理负载均衡

# Attention Worker 侧
# 所有 worker 使用相同的 recv_key，StepMesh 会：
# 1. 聚合所有 worker 的数据
# 2. 分发给可用的 server

# FFN Server 侧
# 所有 server 使用相同的 recv_key，StepMesh 会：
# 1. 从 PS 获取分配的数据
# 2. 计算后返回给对应的 worker

# 配置
afd_config = AFDConfig(
    num_attention_servers=4,  # 4 个 Worker
    num_ffn_servers=2,        # 2 个 Server
    afd_extra_config={
        "load_balance": "round_robin"  # 轮询分配
    }
)
```

---

## 异步通信机制

### 异步操作的优势

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         同步 vs 异步通信                                    │
└─────────────────────────────────────────────────────────────────────────────┘

【同步通信】
Attention Worker                            GPU 状态
     │                                        │
     │ [计算 Attention]                       │ [████████████] 100% 利用
     │                                        │
     │ [发送数据] ─────────────────────▶      │ [░░░░░░░░░░░░] 空闲等待
     │    wait()                             │
     │                                        │
     │ ◀───────────────────────── [FFN 计算]  │ [░░░░░░░░░░░░] 空闲等待
     │                                        │
     │ [继续计算]                             │ [████████████] 恢复计算
     │                                        │
     问题：GPU 在通信和 FFN 计算期间空闲

【异步通信】
Attention Worker                            GPU 状态
     │                                        │
     │ [计算 Attention]                       │ [████████████] 100% 利用
     │                                        │
     │ [发送数据 - isend] ─────────────▶      │ [████████████] 继续计算!
     │    不等待                              │
     │                                        │
     │ [计算下一层或其他操作]                 │ [████████████] 持续利用
     │                                        │
     │ [需要结果时 - wait] ◀────────────────   │ [████████████] 仅等待必要时间
     │                                        │
     优势：
     1. 计算与通信重叠
     2. 提高 GPU 利用率
     3. 降低总体延迟
```

### 异步操作实现

**P2P 连接器**：

```python
# 1. 异步发送
def send_attn_output(self, hidden_states, metadata, **kwargs):
    # 准备数据
    intermediate_tensors = self.create_intermediate_tensors(...)

    # 同步流（确保数据准备好）
    self.current_stream_synchronize(self.backend)

    # 异步发送（立即返回）
    work_list = self._send_tensor_dict_async(
        intermediate_tensors.tensors,
        dst=dst,
        process_group=self.a2e_group,
    )
    # isend 立即返回，不等待完成

    # 存储句柄
    metadata.send_handle_list = work_list

    # 可以继续其他工作
    return work_list

# 2. 异步接收
def recv_attn_output(self):
    # 异步接收（立即返回）
    intermediate_tensors, work_list = self._recv_tensor_dict_async(
        src=src,
        process_group=self.a2e_group,
    )
    # irecv 立即返回，不等待数据到达

    # 存储句柄
    metadata.recv_handle_list = work_list

    # 可以继续其他工作
    return intermediate_tensors, metadata

# 3. 等待完成
def recv_ffn_output(self):
    # 从元数据获取句柄
    work_list = metadata.recv_handle_list

    # 等待所有异步操作完成
    for work in work_list:
        work.wait()  # 阻塞直到完成

    # 现在可以使用数据了
    return intermediate_tensors["hidden_states"]
```

**StepMesh 连接器**：

```python
# 1. 异步 push_pull
def send_attn_output(self, hidden_states, metadata):
    # 准备缓冲区
    send_buff[0].copy_(hidden_states)

    # push_pull 是异步的
    event = ps.push_pull(
        send_buff,
        send_key,
        recv_buff,
        recv_key,
    )
    # 立即返回，不等待完成

    # 存储事件
    self.events.append((event, metadata))

    return event

# 2. 等待完成
def recv_ffn_output(self):
    # 获取事件
    event, metadata = self.events.popleft()

    # 等待 push_pull 完成
    ps.wait(event, timeout_ms=50000)

    # 现在可以从 recv_buffer 读取数据
    return self.recv_buffer[stage_idx][0][:seq_len]
```

### 流水线并行中的异步通信

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    流水线并行 + 异步通信                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Stage 0                    Stage 1                    Stage 2
   │                          │                          │
   │ [T0] Req 1: Attn        │                          │
   ├─▶ send_attn_output() ───┼──────────────────────────┼──▶
   │    (异步)                │                          │
   │                          │                          │
   │ [T1] Req 2: Attn        │ [T0] Req 1: FFN          │
   ├─▶ send_attn_output() ───┼──▶                     │
   │    (异步)                │                          │
   │                          │                          │
   │ [T2] Req 3: Attn        │ [T1] Req 2: FFN          │ [T0] Req 1: 完成
   ├─▶ send_attn_output() ───┼──▶ send_ffn_output() ───┼──▶
   │    (异步)                │    (异步)                │
   │                          │                          │

关键点：
1. 异步通信允许不同请求在不同阶段并发执行
2. 每个 stage 独立处理自己的请求
3. 通信在后台进行，不阻塞计算
4. 使用事件队列跟踪异步操作

实现：
class PipelineManager:
    def __init__(self, num_stages):
        self.num_stages = num_stages
        self.events = deque(maxlen=num_stages)

    def submit(self, request_id, hidden_states, metadata):
        stage_idx = request_id % self.num_stages

        # 异步发送
        event = connector.send_attn_output(hidden_states, metadata)

        # 存储事件
        self.events.append((request_id, stage_idx, event))

    def get_completed(self):
        # 检查最早的事件
        if self.events:
            request_id, stage_idx, event = self.events[0]

            # 非阻塞检查
            if event.is_completed():
                self.events.popleft()
                return request_id

        return None
```

---

## 最佳实践

### 1. 连接器选择

```python
# 决策树
def choose_connector(config):
    """根据配置选择合适的连接器"""

    # 1. 测试/开发环境
    if config.environment == "development":
        return "dummy"

    # 2. 小规模生产（2-8 节点）
    elif config.num_nodes <= 8:
        if config.hardware == "nvidia":
            return "p2pconnector"  # NCCL
        elif config.hardware == "ascend":
            return "p2pconnector"  # HCCL
        else:
            return "p2pconnector"  # Gloo

    # 3. 大规模生产（8+ 节点）
    elif config.num_nodes > 8:
        return "stepmesh"

    else:
        raise ValueError(f"Unsupported config: {config}")
```

### 2. 配置优化

```python
# P2P 连接器优化配置
afd_config = AFDConfig(
    afd_connector="p2pconnector",
    afd_role="attention",
    afd_extra_config={
        "afd_size": "4:4",  # 4 Attention, 4 FFN
    }
)

# StepMesh 连接器优化配置
afd_config = AFDConfig(
    afd_connector="stepmesh",
    afd_role="attention",
    afd_host="stepmesh-server",
    afd_port=1239,
    num_afd_stages=3,  # 流水线阶段数
    num_attention_servers=8,
    num_ffn_servers=8,
    multistream_info={
        "enable": "True",     # 启用多流
        "core_num": "8"       # 通信流核心数
    },
    quant_mode=1  # 启用量化减少通信量
)

# 环境变量优化
export VLLM_MULTISTREAM_ENABLE="True"
export VLLM_MULTISTREAM_CORE_NUM="8"
export DMLC_ENABLE_RDMA="ibverbs"  # 启用 RDMA
export STEPMESH_BIND_CPU_CORE="1"  # CPU 绑定
```

### 3. 错误处理

```python
# 连接器错误处理示例
class SafeAFDConnector:
    """带错误处理的 AFD 连接器包装器"""

    def __init__(self, connector):
        self.connector = connector
        self.max_retries = 3
        self.retry_delay = 1.0

    def send_attn_output(self, hidden_states, metadata, **kwargs):
        """带重试的发送"""
        for attempt in range(self.max_retries):
            try:
                return self.connector.send_attn_output(
                    hidden_states, metadata, **kwargs
                )
            except RuntimeError as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Send failed (attempt {attempt + 1}): {e}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    raise

    def recv_ffn_output(self, timeout_ms=5000):
        """带超时的接收"""
        try:
            return self.connector.recv_ffn_output(timeout_ms=timeout_ms)
        except Exception as e:
            logger.error(f"Recv failed: {e}")
            # 返回零张量作为回退
            return torch.zeros_like(
                self.last_hidden_states,
                device=self.last_device
            )
```

### 4. 性能监控

```python
# 连接器性能监控
class MonitoredAFDConnector:
    """带性能监控的 AFD 连接器"""

    def __init__(self, connector):
        self.connector = connector
        self.metrics = {
            "send_count": 0,
            "recv_count": 0,
            "send_time": [],
            "recv_time": [],
            "comm_time": [],
        }

    def send_attn_output(self, hidden_states, metadata, **kwargs):
        """监控发送性能"""
        start = time.time()

        result = self.connector.send_attn_output(
            hidden_states, metadata, **kwargs
        )

        elapsed = time.time() - start
        self.metrics["send_count"] += 1
        self.metrics["send_time"].append(elapsed)

        logger.debug(f"Send took {elapsed:.4f}s")

        return result

    def get_metrics(self):
        """获取性能指标"""
        return {
            "avg_send_time": np.mean(self.metrics["send_time"]),
            "avg_recv_time": np.mean(self.metrics["recv_time"]),
            "total_comm_time": sum(self.metrics["comm_time"]),
            "send_count": self.metrics["send_count"],
            "recv_count": self.metrics["recv_count"],
        }
```

### 5. 调试技巧

```python
# 连接器调试
def debug_afd_connector(connector, rank):
    """调试 AFD 连接器"""

    logger.info(f"Rank {rank} Connector Info:")
    logger.info(f"  Type: {type(connector).__name__}")
    logger.info(f"  Initialized: {connector.is_initialized}")
    logger.info(f"  Rank: {connector.get_connector_rank()}")
    logger.info(f"  Local Rank: {connector.get_connector_local_rank()}")

    # 测试通信
    if isinstance(connector, DummyAFDConnector):
        logger.info("  Testing dummy connector...")

        # 创建测试数据
        hidden_states = torch.randn(100, 768)
        metadata = AFDConnectorMetadata.create_attention_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_len=100,
            dtype=torch.float32,
            device=torch.device("cpu")
        )

        # 测试发送
        connector.send_attn_output(hidden_states, metadata)
        logger.info("  Send test passed")

        # 测试接收
        ffn_output = connector.recv_ffn_output()
        logger.info(f"  Recv test passed: shape={ffn_output.shape}")
```

---

## 总结

### AFDConnector 模块核心价值

1. **统一抽象**：屏蔽底层通信差异，提供一致接口
2. **可插拔设计**：支持多种连接器，运行时切换
3. **高性能**：异步通信，零拷贝优化，计算与通信重叠
4. **可扩展**：插件机制支持自定义连接器

### 连接器选择指南

| 场景 | 推荐连接器 | 原因 |
|------|-----------|------|
| 单元测试 | Dummy | 无依赖，快速验证 |
| 开发调试 | Dummy | 简化问题定位 |
| 小规模生产（≤8 节点） | P2P | 配置简单，性能好 |
| 大规模生产（>8 节点） | StepMesh | 可扩展，支持 RDMA |
| 异构硬件 | P2P + HCCL/NCCL | 自动适配硬件 |

### 性能优化建议

1. **启用异步通信**：使用 `isend`/`irecv` 实现计算与通信重叠
2. **配置多流**：启用 `multistream_info` 提高通信效率
3. **优化缓冲区**：预分配缓冲区，避免动态分配
4. **启用 RDMA**：在 InfiniBand 网络上启用 RDMA
5. **CPU 绑定**：绑定通信线程到专用核心
6. **量化优化**：使用 `quant_mode` 减少通信量

### 未来扩展方向

1. **更多连接器**：支持 gRPC、UCX 等通信后端
2. **智能路由**：根据负载动态调整连接
3. **容错机制**：支持连接断开重连
4. **自动调优**：根据网络条件自动优化参数
5. **性能分析**：集成性能分析工具
