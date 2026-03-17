# PagedAttention 图表 - Mermaid 格式

## 1. PagedAttention 完整工作流程图

```mermaid
flowchart TB
    Start([请求到达]) --> Init[初始化阶段]

    subgraph Init [初始化阶段]
        I1[计算 GPU 可用块数量]
        I2[分配 KV Cache 内存]
        I3[初始化 Block Manager]
    end

    Init --> Prefill[Prefill 阶段]

    subgraph Prefill [Prefill 阶段]
        P1[计算需要的块数量]
        P2[分配物理块]
        P3[写入 KV Cache]
        P4[计算 Attention]
    end

    Prefill --> Decode[Decode 阶段]

    subgraph Decode [Decode 阶段]
        D1{选择 V1 或 V2}
        D2[计算 Attention]
        D3[生成新 token]
    end

    Decode --> Memory[内存管理阶段]

    subgraph Memory [内存管理]
        M1[块分配]
        M2[块释放]
        M3[Prefix Caching]
    end

    Memory --> End([返回响应])

    style Init fill:#e1f5ff
    style Prefill fill:#fff4e6
    style Decode fill:#e8f5e9
    style Memory fill:#f3e5f5
```

## 2. Continuous Batching 工作流程

```mermaid
sequenceDiagram
    participant Scheduler as 调度器
    participant Running as 运行队列
    participant Waiting as 等待队列
    participant BlockMgr as Block Manager
    participant GPU as GPU

    Note over Scheduler: 开始新的迭代
    Scheduler->>Running: 检查已完成的请求

    alt 有请求完成
        Running->>BlockMgr: 释放 KV Cache 块
        BlockMgr-->>GPU: 释放内存
        Running->>Running: 移除已完成请求
    end

    Scheduler->>Waiting: 检查等待队列

    alt 有空闲块且有待处理请求
        Waiting->>BlockMgr: 请求分配块
        BlockMgr-->>GPU: 分配物理块
        BlockMgr-->>Waiting: 返回块号
        Waiting->>Running: 加入运行队列
    end

    Scheduler->>GPU: 执行推理
    GPU-->>Scheduler: 返回结果

    Scheduler->>BlockMgr: 更新 KV Cache
    BlockMgr-->>GPU: 写入新 tokens

    Note over Scheduler: 下一轮迭代
```

## 3. 块分配流程图

```mermaid
flowchart TD
    Start([新请求到达]) --> Calc[计算所需块数量<br/>num_blocks = ⌈num_tokens / block_size⌉]

    Calc --> Check{检查可用块}

    Check -->|有足够块| Alloc[分配物理块]
    Check -->|块不足| Preempt[触发抢占机制]

    Preempt --> Preempt1[选择低优先级请求]
    Preempt1 --> Preempt2[释放其 KV Cache 块]
    Preempt2 --> Alloc

    Alloc --> Update[更新 Block Table]
    Update --> Queue[请求加入运行队列]
    Queue --> End([开始处理])

    Preempt2 -.-> Wait[等待资源释放]
    Wait -.-> Alloc

    style Check fill:#fff9c4
    style Alloc fill:#c8e6c9
    style Preempt fill:#ffcdd2
    style End fill:#b2dfdb
```

## 4. Prefix Caching 工作流程

```mermaid
flowchart LR
    subgraph Requests [请求示例]
        A[请求 A<br/>You are a helpful assistant.<br/>Explain quantum computing]
        B[请求 B<br/>You are a helpful assistant.<br/>What is the capital?]
    end

    subgraph Shared [共享部分]
        S["🔗 共享前缀:<br/>You are a helpful assistant."]
    end

    subgraph Unique [独立部分]
        U1[Explain quantum computing]
        U2[What is the capital?]
    end

    A --> S
    B --> S
    S --> H1[块哈希: SHA256]
    S --> Cache[✓ 命中缓存]
    S --> Share[📦 共享物理块]

    A --> U1
    B --> U2
    U1 --> N1[新增块]
    U2 --> N2[新增块]

    style Shared fill:#c8e6c9
    style Cache fill:#fff9c4
    style Share fill:#b2dfdb
```

## 5. PagedAttention V1/V2 选择逻辑

```mermaid
flowchart TD
    Start([Decode 阶段]) --> CheckLen{max_seq_len<br/>≤ 8192?}

    CheckLen -->|是| CheckBatch{num_seqs × num_heads<br/> > 512?}
    CheckLen -->|否| CalcPart[计算分区数<br/>max_num_partitions =<br/>⌈max_seq_len / 512⌉]

    CalcPart --> CheckPart{num_partitions == 1?<br/>或大批量?}

    CheckBatch -->|是| UseV1[PagedAttention V1]
    CheckBatch -->|否| UseV2[PagedAttention V2]

    CheckPart -->|是| UseV1
    CheckPart -->|否| UseV2

    UseV1 --> V1Feat["✓ 直接计算<br/>✓ 少共享内存<br/>✓ 适合短序列"]
    UseV2 --> V2Feat["✓ 分区归约<br/>✓ 512 tokens/分区<br/>✓ 适合长序列"]

    V1Feat --> End([生成输出])
    V2Feat --> End

    style CheckLen fill:#fff9c4
    style CheckBatch fill:#fff9c4
    style CheckPart fill:#fff9c4
    style UseV1 fill:#c8e6c9
    style UseV2 fill:#ffcdd2
```

## 6. 系统架构总览

```mermaid
flowchart TB
    subgraph Input [输入层]
        Req[用户请求]
        Prompt[提示词]
    end

    subgraph Scheduler [调度器]
        Queue[请求队列]
        Batch[Continuous Batching]
        Schedule[调度策略]
    end

    subgraph Memory [内存管理层]
        BlockMgr[Block Manager]
        BlockPool[物理块池]
        BlockTable[Block Table]
        PrefixCache[Prefix Cache]
    end

    subgraph Compute [计算层]
        Prefill[PagedAttention Prefill]
        Decode[PagedAttention Decode]
        ATTN[Attention 计算]
    end

    subgraph Storage [存储层]
        GPU[GPU KV Cache]
        CPU[CPU Swap Space]
    end

    Req --> Queue
    Prompt --> Queue
    Queue --> Batch
    Batch --> Schedule
    Schedule --> BlockMgr

    BlockMgr --> BlockPool
    BlockMgr --> BlockTable
    BlockMgr --> PrefixCache

    Schedule --> Prefill
    Schedule --> Decode

    Prefill --> ATTN
    Decode --> ATTN

    BlockPool --> GPU
    BlockPool --> CPU

    ATTN --> Output([生成响应])

    style Scheduler fill:#e1f5ff
    style Memory fill:#fff4e6
    style Compute fill:#e8f5e9
    style Storage fill:#f3e5f5
```

## 7. 性能对比图

```mermaid
xychart-beta
    title "PagedAttention 性能提升对比"
    x-axis ["吞吐量", "GPU利用率", "P99延迟", "内存利用率"]
    y-axis "相对值 (%)" 0 --> 400
    line [100, 60, 100, 65]
    line [300, 90, 30, 95]
```

## 使用说明

### 在飞书文档中使用：

1. **插入流程图**：
   - 在飞书文档中输入 `/mermaid` 或选择"插入" → "代码块"
   - 选择 Mermaid 语言
   - 复制上述代码粘贴进去

2. **插入图片**：
   - 将生成的 PNG/PDF 图片上传到飞书文档
   - 拖拽到合适位置

3. **调整样式**：
   - 飞书支持 Mermaid 的所有主题
   - 可以在代码块中添加 `%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#ff0000'}}}%%` 来自定义颜色

### 各图表用途说明：

- **图表1**: 展示 PagedAttention 的完整工作流程，适合放在文档开头
- **图表2**: 展示 Continuous Batching 的时序交互，适合放在批处理章节
- **图表3**: 展示块分配的详细逻辑，适合放在内存管理章节
- **图表4**: 展示 Prefix Caching 的工作原理，适合放在优化技术章节
- **图表5**: 展示 V1/V2 算法选择逻辑，适合放在核心算法章节
- **图表6**: 展示系统整体架构，适合放在架构设计章节
- **图表7**: 展示性能对比数据，适合放在性能测试章节
