# vLLM PagedAttention 源码分析 - 技术图表

基于 vllm-main 源码的实际实现，绘制详细的技术架构和工作流程图。

---

## 1. vLLM 系统架构总览

### 核心组件交互图

```mermaid
flowchart TB
    subgraph Client ["客户端层"]
        API["LLM API"]
        Req["请求输入"]
    end

    subgraph SchedulerLayer ["调度层 - scheduler.py"]
        Scheduler["Scheduler<br/>调度器"]
        WaitingQueue["waiting: RequestQueue<br/>等待队列"]
        RunningList["running: list[Request]<br/>运行列表"]
        Policy["SchedulingPolicy<br/>FCFS/Priority"]
    end

    subgraph KVCacheLayer ["KV Cache 管理层 - kv_cache_manager.py"]
        KVManager["KVCacheManager<br/>KV Cache 管理器"]
        BlockPool["BlockPool<br/>块池"]
        FreeQueue["free_block_queue<br/>空闲块队列"]
        CacheMap["cached_block_hash_to_block<br/>哈希缓存映射"]
        EncoderMgr["EncoderCacheManager<br/>编码器缓存"]
    end

    subgraph AttentionLayer ["计算层 - attention/"]
        AttnLayer["Attention Layer<br/>注意力层"]
        AttnBackend["AttentionBackend<br/>后端选择器"]
        PagedAttn["PagedAttention<br/>分页注意力"]
        Ops["ops.paged_attention_v1/v2<br/>内核实现"]
    end

    subgraph MemoryLayer ["存储层"]
        GPUKV["GPU KV Cache<br/>Shape: (2, num_blocks,<br/>block_size × kv_heads × head_size)"]
        CPUKV["CPU Swap Space<br/>swap_space GiB"]
    end

    Req --> Scheduler
    Scheduler --> WaitingQueue
    Scheduler --> RunningList
    WaitingQueue --> Policy
    Policy --> Scheduler

    Scheduler --> KVManager
    KVManager --> BlockPool
    BlockPool --> FreeQueue
    BlockPool --> CacheMap
    KVManager --> EncoderMgr

    RunningList --> AttnLayer
    AttnLayer --> AttnBackend
    AttnBackend --> PagedAttn
    PagedAttn --> Ops

    KVManager --> GPUKV
    GPUKV --> CPUKV

    style SchedulerLayer fill:#e1f5ff
    style KVCacheLayer fill:#fff4e6
    style AttentionLayer fill:#e8f5e9
    style MemoryLayer fill:#f3e5f5
```

---

## 2. PagedAttention V1/V2 选择逻辑

### 基于源码的决策树

```mermaid
flowchart TD
    Start(["forward_decode()<br/>Decode 阶段开始"]) --> Init

    Init --> GetParams["获取参数:<br/>num_seqs, num_heads, head_size<br/>max_seq_len, block_size"]

    GetParams --> CalcPart["计算分区数:<br/>max_num_partitions =<br/>⌈max_seq_len / 512⌉<br/><br/>_PARTITION_SIZE = 512"]

    CalcPart --> CheckLen{"max_seq_len ≤ 8192?"}

    CheckLen -->|是| CheckPartitions{"max_num_partitions == 1?"}
    CheckLen -->|否| CheckParallel{"num_seqs × num_heads > 512?"}

    CheckPartitions -->|是| UseV1["use_v1 = True<br/>选择 PagedAttention V1"]
    CheckPartitions -->|否| CheckParallel

    CheckParallel -->|是| UseV1
    CheckParallel -->|否| UseV2["use_v1 = False<br/>选择 PagedAttention V2"]

    UseV1 --> V1Kernel["ops.paged_attention_v1()<br/><br/>特点:<br/>• 直接计算 Attention<br/>• 无需分区归约<br/>• 共享内存占用少<br/>• 适合短序列或大批量"]

    UseV2 --> V2Prep["准备 V2 临时存储:<br/>tmp_output: [num_seqs, num_heads,<br/>max_num_partitions, head_size]<br/>exp_sums: [num_seqs, heads,<br/>max_num_partitions]<br/>max_logits: [num_seqs, heads,<br/>max_num_partitions]"]

    V2Prep --> V2Kernel["ops.paged_attention_v2()<br/><br/>特点:<br/>• 分区归约 (512 tokens/区)<br/>• 避免共享内存不足<br/>• 适合长序列 (>8192)"]

    V1Kernel --> Output(["返回 output"])
    V2Kernel --> Output

    style CheckLen fill:#fff9c4
    style CheckPartitions fill:#fff9c4
    style CheckParallel fill:#fff9c4
    style UseV1 fill:#c8e6c9
    style UseV2 fill:#64b5f6
    style Output fill:#b2dfdb
```

### 源码参考

```python
# vllm/attention/ops/paged_attn.py:134-135
use_v1 = (max_seq_len <= 8192
          and (max_num_partitions == 1 or num_seqs * num_heads > 512))
```

---

## 3. KV Cache 内存布局

### 数据结构详解

```mermaid
graph TB
    subgraph KVCache ["KV Cache Tensor<br/>Shape: (2, num_blocks, block_size × num_kv_heads × head_size)"]
        direction TB
        KV["kv_cache"]

        subgraph Index0 ["维度 0: 2 (Key/Value)"]
            K["Index 0: key_cache<br/>(num_blocks, block_size × kv_heads × head_size)"]
            V["Index 1: value_cache<br/>(num_blocks, block_size × kv_heads × head_size)"]
        end

        subgraph Index1 ["维度 1: num_blocks"]
            B0["Block 0"]
            B1["Block 1"]
            B2["Block 2"]
            BN["Block N"]
        end

        subgraph Index2 ["维度 2: Data"]
            Data["block_size × num_kv_heads × head_size<br/><br/>示例 (block_size=16, kv_heads=32, head_size=128):<br/>16 × 32 × 128 = 65,536 elements"]
        end
    end

    subgraph BlockTable ["Block Table<br/>Shape: (batch_size, max_blocks_per_seq)"]
        BT["block_tables"]
        Seq0["Seq 0: [5, 2, 9, 12, 3, 8, 15]<br/>→ tokens 分布在物理块 5,2,9,12,3,8,15"]
        Seq1["Seq 1: [2, 9, 12, 7]<br/>→ 与 Seq 0 共享块 2,9,12"]
    end

    subgraph PhysicalBlocks ["物理块池 - BlockPool"]
        Pool["blocks: list[KVCacheBlock]"]
        Block0["KVCacheBlock(0)<br/>block_id=0<br/>ref_count=0"]
        Block1["KVCacheBlock(1)<br/>block_id=1<br/>ref_count=2"]
        BlockN["KVCacheBlock(N)<br/>block_id=N<br/>ref_count=1"]
    end

    BlockTable -.->|"逻辑到物理映射"| PhysicalBlocks
    KVCache -.->|"实际存储"| PhysicalBlocks

    style Index0 fill:#e1f5ff
    style Index1 fill:#fff4e6
    style Index2 fill:#e8f5e9
    style BlockTable fill:#f3e5f5
    style PhysicalBlocks fill:#ffe0b2
```

### 源码参考

```python
# vllm/attention/ops/paged_attn.py:48-54
@staticmethod
def get_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> Tuple[int, ...]:
    return (2, num_blocks, block_size * num_kv_heads * head_size)
```

---

## 4. 块分配与生命周期管理

### BlockPool 工作流程

```mermaid
stateDiagram-v2
    [*] --> Initialized: BlockPool.__init__(num_gpu_blocks)

    Initialized --> Free: blocks: list[KVCacheBlock]<br/>free_block_queue: FreeKVCacheBlockQueue

    Free --> Allocated: allocate(block_hash)<br/>从 free_block_queue 获取
    Allocated --> Cached: 缓存已满块<br/>cached_block_hash_to_block

    Cached --> InUse: 请求引用<br/>ref_count++
    InUse --> Free: 请求完成<br/>free(block_id)<br/>ref_count--

    Cached --> Evicted: 缓存淘汰<br/>LRU 策略
    Evicted --> Free: pop(block_hash, block_id)

    note right of Free
        free_block_queue:
        - 双向链表
        - LRU 顺序
        - O(1) 分配/释放
    end note

    note right of Cached
        cached_block_hash_to_block:
        - Hash → Block 映射
        - 支持 Prefix Caching
        - 多块相同 hash
    end note
```

### 块分配详细流程

```mermaid
flowchart TD
    Start(["新请求到达"]) --> CalcNeeded["计算需要的块数:<br/>num_blocks =<br/>⌈num_tokens / block_size⌉"]

    CalcNeeded --> CheckHash{启用<br/>Prefix Cache?}

    CheckHash -->|是| ComputeHash["计算块哈希:<br/>block_hash =<br/>get_block_hash(tokens)<br/><br/>算法: sha256/sha256_cbor"]
    CheckHash -->|否| CheckFree

    ComputeHash --> LookupCache["查找缓存:<br/>cached_block =<br/>cached_block_hash_to_block<br/>.get_one_block(block_hash)"]

    LookupCache --> Hit{"缓存命中?"}

    Hit -->|是| IncRef["增加引用计数:<br/>cached_block.ref_count++"]
    Hit -->|否| CheckFree

    IncRef --> AddToReq["添加到请求:<br/>req.blocks.append(block)"]

    CheckFree{free_block_queue<br/>有足够块?}

    CheckFree -->|是| Allocate["分配新块:<br/>block =<br/>free_block_queue.popleft()"]
    CheckFree -->|否| Preempt

    Preempt["触发抢占:<br/>1. 选择低优先级请求<br/>2. 释放其块<br/>3. 重新分配"]

    Preempt --> Allocate

    Allocate --> InitBlock["初始化块:<br/>KVCacheBlock(block_id)"]

    InitBlock --> InsertCache{启用缓存?}

    InsertCache -->|是| CacheInsert["插入缓存:<br/>cached_block_hash_to_block<br/>.insert(block_hash, block)"]
    InsertCache -->|否| AddToReq

    CacheInsert --> AddToReq

    AddToReq --> UpdateTable["更新 Block Table:<br/>block_tables[seq_id]<br/>.append(block_id)"]

    UpdateTable --> End(["块分配完成"])

    style CheckHash fill:#fff9c4
    style Hit fill:#c8e6c9
    style CheckFree fill:#ffcdd2
    style Preempt fill:#ffcdd2
    style End fill:#b2dfdb
```

---

## 5. 调度器核心流程

### Scheduler.schedule() 主循环

```mermaid
flowchart TD
    Start(["schedule() 调用"]) --> Init

    Init --> InitVars["初始化变量:<br/>scheduled_new_reqs = []<br/>scheduled_resumed_reqs = []<br/>scheduled_running_reqs = []<br/>preempted_reqs = []<br/>req_to_new_blocks = {}<br/>token_budget = max_num_batched_tokens"]

    InitVars --> Phase1["阶段 1: 处理等待队列"]

    Phase1 --> WhileWaiting{"waiting 不空<br/>且 token_budget > 0"}

    WhileWaiting -->|是| GetNext["get_next_from_waiting()<br/>根据策略(FCFS/Priority)<br/>获取下一个请求"]

    GetNext --> TrySchedule["尝试调度请求"]

    TrySchedule --> CheckEncoder{需要编码器输入?}

    CheckEncoder -->|是| AllocEncoder["encoder_cache_manager<br/>.allocate()<br/>分配编码器缓存"]
    CheckEncoder -->|否| AllocBlocks

    AllocEncoder --> AllocBlocks["kv_cache_manager<br/>.allocate()<br/>分配 KV Cache 块"]

    AllocBlocks --> CheckAlloc{"分配成功?"}

    CheckAlloc -->|是| AddScheduled["添加到调度列表:<br/>scheduled_new_reqs<br/>.append(req)"]
    CheckAlloc -->|否| NoBudget["token_budget 不足<br/>或块不足"]

    AddScheduled --> UpdateBudget["更新预算:<br/>token_budget -=<br/>num_tokens"]
    UpdateBudget --> WhileWaiting

    NoBudget --> Phase2

    WhileWaiting -->|否| Phase2["阶段 2: 处理运行列表"]

    Phase2 --> ForRunning["遍历 running 列表"]

    ForRunning --> CheckStop{"请求完成?"}

    CheckStop -->|是| RemoveReq["移除请求:<br/>running.remove(req)<br/>kv_cache_manager<br/>.free(req.blocks)"]
    CheckStop -->|否| CheckContinue

    RemoveReq --> ForRunning

    CheckContinue{"可以继续?"}

    CheckContinue -->|否| PreemptReq["抢占请求:<br/>preempted_reqs<br/>.append(req)"]
    CheckContinue -->|是| AddRunning["添加到运行列表:<br/>scheduled_running_reqs<br/>.append(req)"]

    PreemptReq --> ForRunning
    AddRunning --> ForRunning

    ForRunning --> Phase3["阶段 3: 构建输出"]

    Phase3 --> CreateOutput["创建 SchedulerOutput:<br/>- scheduled_new_reqs<br/>- scheduled_resumed_reqs<br/>- scheduled_running_reqs<br/>- req_to_new_blocks<br/>- preempted_reqs"]

    CreateOutput --> PublishEvents["发布 KV Cache 事件:<br/>BlockStored, BlockRemoved"]

    PublishEvents --> End(["返回 SchedulerOutput"])

    style Phase1 fill:#e1f5ff
    style Phase2 fill:#fff4e6
    style Phase3 fill:#e8f5e9
    style CheckAlloc fill:#c8e6c9
    style NoBudget fill:#ffcdd2
    style End fill:#b2dfdb
```

---

## 6. Prefix Caching 实现细节

### 块哈希与缓存查找

```mermaid
sequenceDiagram
    participant Req as Request
    participant Hash as BlockHash
    participant Pool as BlockPool
    participant Cache as cached_block_hash_to_block
    participant Queue as free_block_queue

    Req->>Hash: 计算块哈希<br/>hash_algo: sha256/sha256_cbor
    Hash->>Hash: make_block_hash_with_group_id(<br/>  block_hash, group_id<br/>)

    Req->>Pool: 查找缓存块
    Pool->>Cache: get_one_block(block_hash)

    alt 缓存命中
        Cache-->>Pool: KVCacheBlock
        Pool->>Pool: block.ref_count++
        Pool-->>Req: 返回已缓存的块
    else 缓存未命中
        Cache-->>Pool: None
        Pool->>Queue: popleft() 获取空闲块
        Queue-->>Pool: KVCacheBlock
        Pool->>Cache: insert(block_hash, block)
        Pool-->>Req: 返回新分配的块
    end
```

### 缓存淘汰策略

```mermaid
flowchart TD
    Start(["需要分配新块"]) --> CheckFree{free_block_queue<br/>为空?}

    CheckFree -->|否| Alloc["直接分配<br/>free_block_queue.popleft()"]
    CheckFree -->|是| CheckCache{缓存有块<br/>可淘汰?}

    CheckCache -->|是| SelectVictim["选择淘汰候选:<br/>free_block_queue 中的<br/>LRU 块"]
    CheckCache -->|否| PreemptReq["抢占请求:<br/>选择低优先级运行请求<br/>释放其所有块"]

    SelectVictim --> CheckRef{ref_count == 0?}

    CheckRef -->|是| Evict["淘汰块:<br/>cached_block_hash_to_block<br/>.pop(block_hash, block_id)<br/>block.reset()"]
    CheckRef -->|否| NextVictim["跳过<br/>选择下一个候选"]

    NextVictim --> SelectVictim

    Evict --> Alloc

    PreemptReq --> ReleaseBlocks["释放请求的块:<br/>for block in req.blocks:<br/>  block.ref_count--<br/>  if ref_count == 0:<br/>    evict(block)"]

    ReleaseBlocks --> Alloc

    Alloc --> End(["块分配完成"])

    style CheckFree fill:#fff9c4
    style CheckCache fill:#fff9c4
    style Evict fill:#ffcdd2
    style PreemptReq fill:#ffcdd2
    style End fill:#b2dfdb
```

---

## 7. 完整请求处理时序图

### 从请求到响应的完整流程

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant Scheduler as Scheduler
    participant KVManager as KVCacheManager
    participant BlockPool as BlockPool
    participant Attn as PagedAttention
    participant GPU as GPU KV Cache

    Client->>Scheduler: LLM.generate(prompt)

    Note over Scheduler: 调度阶段
    Scheduler->>Scheduler: schedule()<br/>处理等待队列

    alt 新请求
        Scheduler->>KVManager: allocate(num_tokens)
        KVManager->>BlockPool: 分配块<br/>计算哈希<br/>查找缓存
        BlockPool-->>KVManager: KVCacheBlocks
        KVManager-->>Scheduler: 块列表
        Scheduler->>Scheduler: 添加到 running
    end

    Note over Scheduler: 执行阶段
    Scheduler->>Attn: forward_decode()<br/>forward_prefix()

    alt Prefill 阶段
        Attn->>GPU: 写入 KV Cache<br/>reshape_and_cache()
        Attn->>Attn: context_attention_fwd()
        GPU-->>Attn: Prefill 输出
    else Decode 阶段
        Attn->>Attn: 选择 V1/V2<br/>max_seq_len <= 8192<br/>max_num_partitions == 1<br/>OR num_seqs × heads > 512

        alt V1
            Attn->>GPU: paged_attention_v1()
            GPU-->>Attn: V1 输出
        else V2
            Attn->>Attn: 准备临时存储<br/>tmp_output, exp_sums, max_logits
            Attn->>GPU: paged_attention_v2()
            GPU-->>Attn: V2 输出
        end

        Attn->>GPU: 更新 KV Cache<br/>写入新 token
    end

    Attn-->>Scheduler: output

    Note over Scheduler: 内存管理
    alt 请求完成
        Scheduler->>KVManager: free(blocks)
        KVManager->>BlockPool: 释放块<br/>ref_count--
        BlockPool->>BlockPool: 检查是否淘汰
    end

    Scheduler-->>Client: GeneratedOutput
```

---

## 8. CacheConfig 配置参数

### 配置项详解

```mermaid
mindmap
    root((CacheConfig))
        block_size[块大小<br/>1, 8, 16, 32, 64, 128<br/>CUDA ≤ 32]
        gpu_memory_utilization[GPU 内存利用率<br/>默认 0.9 (90%)<br/>范围 0.0-1.0]
        swap_space[CPU 交换空间<br/>默认 4 GiB<br/>用于块换入换出]
        cache_dtype[KV Cache 数据类型<br/>auto, fp8<br/>fp8_e4m3, fp8_e5m2]
        enable_prefix_caching[启用前缀缓存<br/>V1 默认 true<br/>自动共享公共前缀]
        prefix_caching_hash_algo[哈希算法<br/>sha256<br/>sha256_cbor]
        sliding_window[滑动窗口<br/>限制 KV Cache<br/>只保留最近 N 个 token]
        cpu_offload_gb[CPU 卸载空间<br/>默认 0 GiB<br/>虚拟增加 GPU 内存]
        calculate_kv_scales[计算 FP8 缩放因子<br/>默认 false<br/>动态计算 k_scale, v_scale]
        num_gpu_blocks[GPU 块数量<br/>profiling 后自动设置<br/>可手动覆盖]
        num_cpu_blocks[CPU 块数量<br/>用于 swap<br/>自动计算]
```

### 源码参考

```python
# vllm/config/cache.py:32-124
@config
@dataclass
class CacheConfig:
    block_size: BlockSize = None  # 1, 8, 16, 32, 64, 128
    gpu_memory_utilization: float = 0.9
    swap_space: float = 4  # GiB
    cache_dtype: CacheDType = "auto"
    enable_prefix_caching: Optional[bool] = None
    prefix_caching_hash_algo: PrefixCachingHashAlgo = "sha256"
    sliding_window: Optional[int] = None
    cpu_offload_gb: float = 0
    calculate_kv_scales: bool = False
    # ... 更多配置
```

---

## 9. 内存优化技术对比

### FP8、Sliding Window、CPU Swap

```mermaid
graph LR
    subgraph FP8 ["FP8 量化"]
        FP8In["输入: KV Cache<br/>float16/bfloat16"]
        FP8Scale["计算 k_scale, v_scale"]
        FP8Out["输出: fp8_e4m3/fp8_e5m2<br/>内存占用减少 50%"]
    end

    subgraph SW ["Sliding Window"]
        SWIn["输入: 长序列<br/>10K+ tokens"]
        SWTrunc["截断: 只保留最近<br/>sliding_window 个 token"]
        SWOut["输出: 固定大小 KV Cache<br/>内存占用可控"]
    end

    subgraph Swap ["CPU Swap"]
        SwapIn["输入: GPU 内存不足"]
        SwapMove["移动: 不常用块 → CPU"]
        SwapOut["输出: GPU 有更多空闲块<br/>CPU 作为二级缓存"]
    end

    FP8In --> FP8Scale --> FP8Out
    SWIn --> SWTrunc --> SWOut
    SwapIn --> SwapMove --> SwapOut

    style FP8 fill:#e8f5e9
    style SW fill:#fff4e6
    style Swap fill:#e1f5ff
```

---

## 10. 性能优化关键路径

### 热路径分析

```mermaid
graph TD
    subgraph HotPath ["性能关键路径"]
        direction TB
        Start(["请求处理"])

        Step1["① 块分配<br/>BlockPool.allocate()<br/>O(1) 从 free_block_queue"]

        Step2["② 哈希计算<br/>get_block_hash()<br/>O(block_size) 计算 SHA256"]

        Step3["③ 缓存查找<br/>cached_block_hash_to_block.get()<br/>O(1) 字典查找"]

        Step4["④ Attention 计算<br/>paged_attention_v1/v2<br/>CUDA 核心并行计算"]

        Step5["⑤ KV Cache 更新<br/>reshape_and_cache()<br/>CUDA kernel 写入"]

        Step6["⑥ 块释放<br/>BlockPool.free()<br/>O(1) 返回 free_block_queue"]

        Start --> Step1
        Step1 --> Step2
        Step2 --> Step3
        Step3 --> Step4
        Step4 --> Step5
        Step5 --> Step6
    end

    subgraph Optimizations ["优化点"]
        O1["Prefix Caching<br/>跳过步骤 1-2-3<br/>直接复用缓存块"]
        O2["Continuous Batching<br/>动态调度<br/>最大化 GPU 利用率"]
        O3["FP8 量化<br/>减少内存访问<br/>提升带宽利用率"]
    end

    HotPath -.-> Optimizations

    style Step1 fill:#ffcdd2
    style Step2 fill:#ffcdd2
    style Step3 fill:#ffcdd2
    style Step4 fill:#fff9c4
    style Step5 fill:#fff9c4
    style Step6 fill:#c8e6c9
    style Optimizations fill:#b2dfdb
```

---

## 使用说明

### 在飞书文档中使用

1. **插入图表**：
   - 在飞书文档中输入 `/mermaid`
   - 选择 Mermaid 代码块
   - 复制对应的图表代码

2. **自定义主题**：
   ```mermaid
   %%{init: {
     'theme': 'base',
     'themeVariables': {
       'primaryColor': '#e1f5ff',
       'primaryTextColor': '#000',
       'primaryBorderColor': '#0288d1',
       'lineColor': '#0288d1',
       'fillType0': '#e1f5ff',
       'fillType1': '#fff9c4',
       'fillType2': '#c8e6c9'
     }
   }}%%
   ```

3. **导出图片**：
   - 使用 Mermaid Live Editor: https://mermaid.live/
   - 导出为 PNG/SVG
   - 插入到文档

### 源码参考

所有图表均基于以下源文件：
- `vllm/attention/ops/paged_attn.py` - PagedAttention 实现
- `vllm/config/cache.py` - CacheConfig 配置
- `vllm/v1/core/block_pool.py` - BlockPool 实现
- `vllm/v1/core/sched/scheduler.py` - 调度器实现
- `vllm/attention/layer.py` - Attention 层

这些图表准确反映了 vLLM 的实际实现细节。
