# PagedAttention 图表补充

本文档补充了 PagedAttention 技术详解所需的所有时序图和流程图。

---

## 图表 1: PagedAttention 完整工作流程图

### 时序图

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant Scheduler as 调度器
    participant BlockMgr as Block Manager
    participant KVCache as KV Cache
    participant Attention as PagedAttention
    participant GPU as GPU

    Client->>Scheduler: 发送请求
    Scheduler->>BlockMgr: 初始化

    Note over BlockMgr: 初始化阶段
    BlockMgr->>GPU: 计算可用块数量
    BlockMgr->>KVCache: 分配 KV Cache 内存
    BlockMgr-->>Scheduler: 就绪

    Note over Scheduler: Prefill 阶段
    Scheduler->>BlockMgr: 计算需要的块数量
    BlockMgr->>KVCache: 分配物理块
    Scheduler->>KVCache: 写入 KV Cache
    Scheduler->>Attention: 计算 Attention
    Attention->>GPU: 执行计算
    GPU-->>Attention: 返回结果

    Note over Scheduler: Decode 阶段
    loop 每个生成步骤
        Scheduler->>Attention: 选择 V1/V2 算法
        Scheduler->>Attention: 计算 Attention
        Attention->>GPU: 生成新 token
        GPU-->>Scheduler: 返回 token
        Scheduler->>KVCache: 更新 KV Cache
    end

    Note over BlockMgr: 内存管理
    BlockMgr->>KVCache: 块分配/释放
    BlockMgr->>KVCache: Prefix Caching

    GPU-->>Client: 返回完整响应
```

---

## 图表 2: Continuous Batching 工作流程

### 流程图

```mermaid
flowchart TB
    Start([系统启动]) --> Init[初始化 Block Manager]

    Init --> Loop{开始调度循环}

    Loop --> Remove[移除已完成的请求]
    Remove --> Free[释放 KV Cache 块]
    Free --> UpdatePool[更新物理块池]

    UpdatePool --> CheckWait{等待队列<br/>有请求?}

    CheckWait -->|有| CheckBlock{有空闲块?}
    CheckWait -->|无| Execute

    CheckBlock -->|有| Alloc[分配块给新请求]
    CheckBlock -->|无| Execute

    Alloc --> AddQueue[加入运行队列]

    AddQueue --> Execute[批量执行推理]
    Execute --> UpdateKV[更新 KV Cache]

    UpdateKV --> Loop

    style Remove fill:#ffcdd2
    style Alloc fill:#c8e6c9
    style Execute fill:#fff9c4
    style Loop fill:#e1f5ff
```

### 对比图

```mermaid
graph LR
    subgraph Static [Static Batching]
        S1[请求1<br/>10 tokens]
        S2[请求2<br/>100 tokens]
        S3[请求3<br/>50 tokens]
        S1 -.->|等待| S2
        S2 -.->|等待| S3
    end

    subgraph Continuous [Continuous Batching]
        C1[请求1<br/>完成释放]
        C2[请求2<br/>继续处理]
        C3[请求3<br/>动态添加]
    end

    style Static fill:#ffcdd2
    style Continuous fill:#c8e6c9
```

---

## 图表 3: 块分配详细流程

```mermaid
flowchart TD
    Start([新请求到达]) --> Calc1[计算 token 数量]

    Calc1 --> Calc2["计算所需块数<br/>num_blocks = ⌈num_tokens / block_size⌉"]

    Calc2 --> CheckFree{检查块池<br/>空闲块数量}

    CheckFree -->|≥ num_blocks| AllocDirect[直接分配块]
    CheckFree -->|< num_blocks| CheckPreempt{可抢占<br/>请求?}

    CheckPreempt -->|有| Preempt[触发抢占机制]
    CheckPreempt -->|无| Wait[等待资源释放]

    Preempt --> Select[选择低优先级请求]
    Select --> FreeBlocks[释放其 KV Cache]
    FreeBlocks --> AllocDirect

    AllocDirect --> CreateBlock[创建 KVCacheBlocks]
    CreateBlock --> UpdateTable[更新 Block Table]

    UpdateTable --> CheckPrefix{启用<br/>Prefix Cache?}

    CheckPrefix -->|是| LookupHash[查找缓存哈希]
    CheckPrefix -->|否| Assign[分配到请求]

    LookupHash --> Hit{缓存<br/>命中?}

    Hit -->|是| ShareBlock[共享已有块]
    Hit -->|否| Assign

    ShareBlock --> Assign
    Assign --> Queue[加入运行队列]
    Queue --> End([开始处理])

    Wait -.-> CheckFree

    style CheckFree fill:#fff9c4
    style CheckPreempt fill:#ffcdd2
    style AllocDirect fill:#c8e6c9
    style Hit fill:#c8e6c9
    style End fill:#b2dfdb
```

---

## 图表 4: Prefix Caching 工作原理

### 共享机制图

```mermaid
graph TB
    subgraph Before [缓存前]
        Req1[请求 A:<br/>You are helpful.<br/>Explain AI]
        Req2[请求 B:<br/>You are helpful.<br/>What is ML?]
    end

    subgraph Process [处理过程]
        Hash1[计算块哈希]
        Hash2[查找缓存]
        Hash3[匹配前缀]
    end

    subgraph After [缓存后]
        Block1[块 0-5:<br/>You are helpful.]
        Block2[块 6:<br/>Explain AI]
        Block3[块 6':<br/>What is ML?]
    end

    Req1 --> Hash1
    Req2 --> Hash1
    Hash1 --> Hash2
    Hash2 --> Hash3
    Hash3 --> Block1
    Block1 --> Share["🔗 共享物理块"]
    Share --> Block2
    Share --> Block3

    style Block1 fill:#c8e6c9
    style Share fill:#fff9c4
```

### 性能提升图

```mermaid
graph LR
    subgraph Without [无 Prefix Cache]
        W1[请求A<br/>1000 tokens] --> W2[计算全部]
        W3[请求B<br/>1000 tokens] --> W4[重复计算]
    end

    subgraph With [有 Prefix Cache]
        C1[请求A<br/>1000 tokens] --> C2[计算并缓存]
        C3[请求B<br/>1000 tokens] --> C4[复用缓存]
    end

    W2 --> T1[时间: 100ms]
    W4 --> T2[时间: 100ms]

    C2 --> T3[时间: 100ms]
    C4 --> T4[时间: 10ms]

    style Without fill:#ffcdd2
    style With fill:#c8e6c9
```

---

## 图表 5: PagedAttention V1/V2 算法选择

```mermaid
flowchart TD
    Start([Decode 阶段开始]) --> GetParams[获取参数:<br/>max_seq_len, num_seqs, num_heads]

    GetParams --> CheckLen{max_seq_len<br/>≤ 8192?}

    CheckLen -->|是| CheckBatch{num_seqs ×<br/>num_heads<br/> > 512?}
    CheckLen -->|否| CalcPart[计算分区数]

    CalcPart --> CheckPart{num_partitions<br/>== 1?}

    CheckBatch -->|是| UseV1[选择 PagedAttention V1]
    CheckBatch -->|否| CalcPart

    CheckPart -->|是| UseV1
    CheckPart -->|否| UseV2[选择 PagedAttention V2]

    UseV1 --> V1Desc["特点:<br/>• 直接计算 Attention<br/>• 共享内存占用少<br/>• 适合短序列或大批量"]
    UseV2 --> V2Desc["特点:<br/>• 分区归约 (512 tokens/区)<br/>• 需要临时存储<br/>• 适合长序列"]

    V1Desc --> End([执行计算])
    V2Desc --> End

    style CheckLen fill:#fff9c4
    style CheckBatch fill:#fff9c4
    style CheckPart fill:#fff9c4
    style UseV1 fill:#c8e6c9
    style UseV2 fill:#64b5f6
```

---

## 图表 6: 内存架构图

```mermaid
graph TB
    subgraph GPU [GPU 内存]
        GPUBlock["<b>GPU 物理块池</b><br/>num_gpu_blocks: 1000<br/>block_size: 16<br/>利用率: 90%+"]
        GPUKV["<b>KV Cache</b><br/>Shape: [2, num_blocks,<br/>block_size × heads × size]"]
    end

    subgraph CPU [CPU 内存]
        CPUKV["<b>CPU Swap Space</b><br/>num_cpu_blocks: 500<br/>swap_space: 4 GB"]
    end

    subgraph Management [管理结构]
        BlockTable["<b>Block Table</b><br/>映射: 逻辑块 → 物理块<br/>Shape: [batch, max_blocks]"]
        PrefixCache["<b>Prefix Cache</b><br/>Hash → Physical Blocks<br/>自动共享公共前缀"]
    end

    subgraph DataFlow [数据流]
        Alloc[分配]
        Swap[换入换出]
        Share[共享]
    end

    GPUBlock --> GPUKV
    GPUKV --> Alloc
    Alloc --> BlockTable
    BlockTable --> Share
    Share --> PrefixCache
    GPUKV --> Swap
    Swap --> CPUKV

    style GPU fill:#e8f5e9
    style CPU fill:#fff3e0
    style Management fill:#e3f2fd
    style DataFlow fill:#f3e5f5
```

---

## 图表 7: 性能对比汇总

```mermaid
xychart-beta
    title "vLLM vs 传统系统性能对比 (相对值)"
    x-axis ["吞吐量", "GPU利用率", "P99延迟降低", "内存利用率"]
    y-axis "相对性能 (%)" 0 --> 400
    bar [100, 65, 100, 60]
    bar [300, 90, 30, 95]
```

### 详细对比表

| 指标 | HuggingFace | TGI | vLLM | 提升倍数 |
|------|-------------|-----|------|---------|
| **吞吐量** | 1x | 2-3x | **20-30x** | vs HF: 20-30x<br/>vs TGI: 7-10x |
| **GPU 利用率** | 60% | 75% | **90%+** | +50% vs HF<br/>+20% vs TGI |
| **P99 延迟** | 100ms | 80ms | **30-50ms** | -50% vs HF<br/>-40% vs TGI |
| **内存利用率** | 60% | 70% | **95%+** | +58% vs HF<br/>+36% vs TGI |
| **并发请求数** | 32 | 128 | **256+** | 8x vs HF<br/>2x vs TGI |

---

## 使用建议

### 在飞书文档中的位置建议：

1. **文档开头** - 插入图表 1（完整工作流程图）
2. **Continuous Batching 章节** - 插入图表 2
3. **内存管理章节** - 插入图表 3（块分配流程）
4. **Prefix Caching 章节** - 插入图表 4
5. **核心算法章节** - 插入图表 5（V1/V2 选择）
6. **架构设计章节** - 插入图表 6（内存架构）
7. **性能测试章节** - 插入图表 7（性能对比）

### 插入方法：

**方法 1: Mermaid 代码块**
1. 在飞书文档中按 `/`
2. 选择"代码块"或"Mermaid"
3. 复制对应的 Mermaid 代码
4. 粘贴到代码块中

**方法 2: 截图插入**
1. 使用 Mermaid Live Editor (https://mermaid.live/) 渲染代码
2. 导出为 PNG/SVG
3. 插入到飞书文档

### 自定义样式：

如果需要自定义颜色主题，在每个 Mermaid 图表开头添加：

```mermaid
%%{init: {'theme':'base', 'themeVariables': {
  'primaryColor':'#e1f5ff',
  'primaryTextColor':'#000',
  'primaryBorderColor':'#0288d1',
  'lineColor':'#0288d1',
  'fillType0':'#e1f5ff',
  'fillType1':'#fff9c4',
  'fillType2':'#c8e6c9'
}}}%%
```

---

所有图表均采用 Mermaid 格式，飞书文档原生支持，可以直接复制使用。
