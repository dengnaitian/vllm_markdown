# MoeDistributeCombineV2 Combine端执行流程

> **生成时间**: 2026-04-09
> **算子名称**: MoeDistributeCombineV2
> **核心功能**: 将各卡专家计算后的Token数据按路由信息发回原Rank，并加权合并

---

## 1. Combine发送端

Combine算子既充当发送端，又充当接收端。发送端的职责是：将本卡上经过专家FFN计算后的结果，按照Dispatch阶段记录的`assistInfoForCombine`路由信息，**原路发回**给Token所属的Rank。

### 执行流程

```
Process() 入口
  │
  ├─ 1. ReduceScatterTrans()            [可选] TP域内ReduceScatter数据传输
  │     将TP域对端数据写入TP窗口，供后续累加
  │
  ├─ 2. BuffInit()                      发送Buffer初始化
  │     初始化readStateBuf、gmTpSendCountQueue等发送所需UB缓冲区
  │
  ├─ 3. SetWaitTpStatusAndDisPatch()    核心发送流程
  │     │
  │     ├─ 3.1 [可选] TP域同步
  │     │   设置TP对端状态标志 Flag=1
  │     │   轮询等待本卡TP窗口状态 Flag==1
  │     │
  │     └─ 3.2 ExpertAlltoAllDispatchCopyAdd()  EP域AllToAll发送
  │           │
  │           ├─ 3.2.1 读取assistInfoForCombine
  │           │   从GM拷贝辅助路由信息到UB
  │           │
  │           ├─ 3.2.2 遍历每个待发送Token（错位发送避免拥塞）
  │           │   │
  │           │   ├─ 解析路由：toRankId / tokenId / topkId
  │           │   │
  │           │   ├─ ExpertAlltoAllDispatchInnerCopyAdd()
  │           │   │   ├─ 计算目标窗口地址
  │           │   │   │   epOffset = tokenId × (K + sharedExpertNum) + topkId
  │           │   │   │   目标地址 = GetWinAddrByRankId(toRankId) + epOffset × hAlignWinSize_
  │           │   │   │
  │           │   │   ├─ [量化模式] 对expandX进行量化后发送
  │           │   │   ├─ [ReduceScatter模式] TP窗口数据+本地数据累加后发送
  │           │   │   └─ [普通模式] 直接拷贝expandX到目标窗口
  │           │   │
  │           │   └─ 写入接收端状态标志
  │           │       在接收Rank的状态区写入 Flag=1.0
  │           │       地址 = WinStateAddr(toRankId) + tokenId×flagRcvCount×STATE_OFFSET + topkId×STATE_OFFSET
  │           │
  │           └─ 发送完成
  │
  └─ PipeBarrier(PIPE_ALL)             确保发送操作全部完成
```

### 关键步骤说明

| 步骤 | 函数 | 核心动作 |
|------|------|----------|
| 读取路由信息 | `DataCopyPad(expandIdxLocal, expandIdxGM_[...])` | 从GM拷贝`assistInfoForCombine`到UB |
| 解析发送目标 | `expandIdxLocal(baseOffset + 0/1/2)` | 依次读取`toRankId`、`tokenId`、`topkId` |
| 计算窗口偏移 | `epOffset = tokenId * (K + sharedExpertNum) + topkId` | 确定数据在目标窗口中的精确位置 |
| 拷贝数据到窗口 | `DataCopyPad(rankWindow_, expandX[...])` | 将专家计算结果写入目标Rank的窗口 |
| 通知接收端 | `DataCopy(stateGMTensor, statusTensor)` | 在接收端状态区写入1.0，表示数据已就绪 |

---

## 2. Combine接收端

接收端的职责是：等待所有Rank发回的Token数据到齐后，按`expertScales`权重进行加权合并，并叠加共享专家结果，最终输出合并后的Token。

### 执行流程

```
AlltoAllBuffInitAndMaskCal() → LocalWindowCopy()
  │
  ├─ 4. AlltoAllBuffInitAndMaskCal()    接收Buffer初始化 + Mask计算
  │     │
  │     ├─ 4.1 AlltoAllCommBuffInit()   分配UB缓冲区
  │     │   tokenBuf / rowTmpFloatBuf / mulBuf / sumFloatBuf / stateBuf等
  │     │
  │     ├─ 4.2 [可选] TokenMaskCalCnt() 计算一维mask有效token数
  │     │
  │     ├─ 4.3 [可选] ExpertMaskCalCnt() 计算二维mask有效token数
  │     │
  │     └─ 4.4 [可选] MaskSpecialExpert() 过滤特殊专家的mask
  │
  └─ 5. LocalWindowCopy()              核心接收与合并流程
        │
        ├─ 5.1 分核计算：将token均匀分配到各AIV核
        │
        ├─ 5.2 ExpertScaleCopy()        拷贝expertScales权重到UB
        │
        └─ 5.3 while(token未全部收齐)    逐token轮询等待
              │
              ├─ WaitDispatch()          等待数据收齐
              │   │
              │   ├─ 读取状态区全部Flag
              │   ├─ Sum累加所有Flag值
              │   ├─ 判断 Sum ≈ (K + sharedExpertNum)
              │   │   ├─ 是 → 清空状态区，返回true
              │   │   └─ 否 → 返回false，继续等待
              │   │
              │   └─ 判定条件: (target - 0.5) < Sum < (target + 0.5)
              │
              ├─ ProcessExpert()         加权合并（数据收齐后执行）
              │   │
              │   ├─ 初始化累加器 sumFloat = 0
              │   │
              │   ├─ 处理MoE专家 (TopK=0 ~ K-1)
              │   │   │
              │   │   └─ ProcessMoeExpert(tokenIndexOffset, topkId, scaleVal)
              │   │       ├─ 从本卡EP窗口读取专家结果
              │   │       │   wAddr = epWindowGM_ + (tokenIndexOffset + topkId) × hAlignWinSize_
              │   │       ├─ [量化模式] 反量化 DeQuantProcess
              │   │       ├─ 类型转换 Cast(FP16→FP32)
              │   │       ├─ 乘以权重 Muls(rowTmp, scale)
              │   │       └─ 累加 Add(sumFloat, sumFloat, rowTmp)
              │   │
              │   ├─ [可选] 处理特殊专家
              │   │   ├─ 零专家 → 跳过
              │   │   ├─ 拷贝专家 → ProcessCopyExpert(直接拷贝原始Token)
              │   │   └─ 常量专家 → ProcessConstantExpert(计算α加权)
              │   │
              │   ├─ 处理共享专家 (TopK=K ~ K+sharedExpertNum-1)
              │   │   从EP窗口读取共享专家结果，直接累加
              │   │
              │   └─ [可选] 叠加sharedExpertX
              │
              ├─ [可选] AddRmsNorm融合计算
              │   计算 x + residual → RMSNorm → y
              │
              └─ 结果搬出
                  Cast(FP32→FP16) → DataCopyPad → expandOutGlobal_[tokenIndex × H]
```

### 关键步骤说明

| 步骤 | 函数 | 核心动作 |
|------|------|----------|
| 轮询等待 | `WaitDispatch()` | 读取状态区Flag，累加后判断是否≈目标值 |
| 读取窗口数据 | `DataCopyPad(tmpUb, rowTmpGlobal_[...])` | 从本卡EP窗口读取对应专家的结果 |
| 反量化 | `DeQuantProcess()` | [可选] 将INT8/FP8量化数据恢复为FP16 |
| 加权累加 | `Muls + Add` | 乘以expertScales权重并累加到sumFloat |
| 输出 | `Cast + DataCopyPad` | FP32→FP16转换后写回GM |

---

## 3. 举例说明

### 场景配置

```yaml
EP域: 8卡 (Rank 0~7)
moeExpertNum: 16
sharedExpertNum: 1
每卡专家数: 2 (Expert 0,1 在 Rank 0; Expert 2,3 在 Rank 1; ...)
K = 2 (TopK)
H = 4 (隐藏维度，简化示意)
```

**路由规则**: `toRankId = expertId / 2`（每卡2个专家）

### 3.1 输入Token排布（本卡 Rank 0 上经过专家FFN后的 expandX）

| 索引 | Token Id | Expert Id | TopK | 驻留Rank |
| :--: | :------: | :-------: | :--: | :------: |
|  0   |    0     |     5     |  0   |    2     |
|  1   |    0     |     9     |  1   |    4     |
|  2   |    1     |     3     |  0   |    1     |
|  3   |    1     |     12    |  1   |    6     |
|  4   |    2     |     7     |  0   |    3     |
|  5   |    2     |     2     |  1   |    1     |
|  6   |    3     |     5     |  0   |    2     |
|  7   |    3     |     14    |  1   |    7     |

> 每行是一条 `[Bs×K, H]` 的Token向量，经过对应专家FFN计算后的结果

### 3.2 发送端：assistInfoForCombine 路由信息

`assistInfoForCombine` 来自 Dispatch 阶段的 `expandIdxOut`，每3个INT32描述一条路由：

| 字段 | 位置 | 含义 |
| :--: | :--: | :--- |
| **toRankId** | baseOffset + 0 | 数据发回的**目标Rank ID**，即Token原始所属的Rank |
| **tokenId** | baseOffset + 1 | **目标Rank上的Token索引**，标识数据写入目标Rank窗口的第几个Token槽位。用于计算窗口偏移：`epOffset = tokenId × (K + sharedExpertNum) + topkId`；以及状态区偏移：`WinStateAddr(toRankId) + tokenId × flagRcvCount × STATE_OFFSET` |
| **topkId** | baseOffset + 2 | **TopK索引**（0 ~ K-1），标识该数据属于Token的第几个专家结果。用于确定在目标Token槽位中的具体位置 |

| 索引 | toRankId | tokenId | topkId | 含义 |
| :--: | :------: | :-----: | :----: | :--- |
| 0    |    2     |    0    |   0    | expandX[0] → 写入Rank2窗口的第0个Token槽位、TopK=0处 |
| 1    |    4     |    0    |   1    | expandX[1] → 写入Rank4窗口的第0个Token槽位、TopK=1处 |
| 2    |    1     |    0    |   0    | expandX[2] → 写入Rank1窗口的第0个Token槽位、TopK=0处 |
| 3    |    6     |    0    |   1    | expandX[3] → 写入Rank6窗口的第0个Token槽位、TopK=1处 |
| 4    |    3     |    0    |   0    | expandX[4] → 写入Rank3窗口的第0个Token槽位、TopK=0处 |
| 5    |    1     |    0    |   1    | expandX[5] → 写入Rank1窗口的第0个Token槽位、TopK=1处 |
| 6    |    2     |    1    |   0    | expandX[6] → 写入Rank2窗口的第1个Token槽位、TopK=0处 |
| 7    |    7     |    0    |   1    | expandX[7] → 写入Rank7窗口的第0个Token槽位、TopK=1处 |

> **注意**：toRankId 是 Token 原始所属的 Rank，而非专家驻留的 Rank。这里 Rank 0 上的 expandX 对应的是**其他 Rank 发来本卡的 Token** 的专家计算结果，需发回原 Rank。tokenId 不同意味着这些数据在目标Rank上属于不同的Token（例如索引6的tokenId=1表示这是目标Rank2上的第1个Token，而非第0个）。

### 3.3 发送端：窗口数据写入

发送端将 expandX 中的数据写入各目标 Rank 的窗口（`flagRcvCount = K + sharedExpertNum = 3`）：

**Rank 2 的窗口**（收到 Token0(E5) 和 Token3(E5) 的结果）：

| 位置 | epOffset计算 | 写入数据 | 状态Flag |
| :--: | :----------: | :------: | :------: |
| Token0, TopK=0 | 0×3+0=0 | Token0_E5结果 | Flag=1.0 |
| Token0, TopK=1 | 0×3+1=1 | (空) | 0.0 |
| Token0, TopK=2 | 0×3+2=2 | (空) | 0.0 |
| Token3, TopK=0 | 1×3+0=3 | Token3_E5结果 | Flag=1.0 |
| Token3, TopK=1 | 1×3+1=4 | (空) | 0.0 |
| Token3, TopK=2 | 1×3+2=5 | (空) | 0.0 |

**Rank 4 的窗口**（收到 Token0(E9) 的结果）：

| 位置 | epOffset计算 | 写入数据 | 状态Flag |
| :--: | :----------: | :------: | :------: |
| Token0, TopK=0 | 0×3+0=0 | (空) | 0.0 |
| Token0, TopK=1 | 0×3+1=1 | Token0_E9结果 | Flag=1.0 |
| Token0, TopK=2 | 0×3+2=2 | (空) | 0.0 |

**Rank 1 的窗口**（收到 Token1(E3) 和 Token2(E2) 的结果）：

| 位置 | epOffset计算 | 写入数据 | 状态Flag |
| :--: | :----------: | :------: | :------: |
| Token1, TopK=0 | 0×3+0=0 | Token1_E3结果 | Flag=1.0 |
| Token1, TopK=1 | 0×3+1=1 | (空) | 0.0 |
| Token1, TopK=2 | 0×3+2=2 | (空) | 0.0 |
| Token2, TopK=0 | 1×3+0=3 | (空) | 0.0 |
| Token2, TopK=1 | 1×3+1=4 | Token2_E2结果 | Flag=1.0 |
| Token2, TopK=2 | 1×3+2=5 | (空) | 0.0 |

**Rank 7 的窗口**（收到 Token3(E14) 的结果）：

| 位置 | epOffset计算 | 写入数据 | 状态Flag |
| :--: | :----------: | :------: | :------: |
| Token3, TopK=0 | 0×3+0=0 | (空) | 0.0 |
| Token3, TopK=1 | 0×3+1=1 | Token3_E14结果 | Flag=1.0 |
| Token3, TopK=2 | 0×3+2=2 | (空) | 0.0 |

### 3.4 接收端：状态区Flag变化与等待过程

以 **Rank 2** 作为接收端为例，观察 Token0 的数据收齐过程：

Rank 2 的 Token0 在 Dispatch 阶段选择了专家 [E5]，Combine 阶段需等待其他 Rank 发回 E5 的计算结果，加上本卡共享专家结果。

`flagRcvCount = K + sharedExpertNum = 2 + 1 = 3`

| 时间点 | 事件 | Flag[0][0] | Flag[0][1] | Flag[0][2] | Sum | 状态 |
| :----: | :--- | :--------: | :--------: | :--------: | :--: | :--: |
| T0 | 初始状态 | 0.0 | 0.0 | 0.0 | 0.0 | 等待 |
| T1 | Rank0发送Token0(E5)结果，写入Flag[0][0]=1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 等待 |
| T2 | Rank3发送Token3(E5)结果，写入Flag[1][0]=1.0 | (Token3的处理独立) | - | - | - | - |
| T3 | 共享专家结果就绪，写入Flag[0][2]=1.0 | 1.0 | 0.0 | 1.0 | 2.0 | 等待 |
| T4 | 所有Rank 2负责的MoE专家Flag已就绪 | 1.0 | 0.0 | 1.0 | 2.0 | ... |

> 实际判定：`targetCount` 由 `tokenTarget` 值决定（考虑 mask 过滤后的有效专家数）。当 `(target-0.5) < Sum < (target+0.5)` 时判定收齐。

### 3.5 接收端：数据合并输出

数据收齐后，接收端对每个 Token 执行加权合并：

以 **Rank 0** 的 Token0 为例（K=2, expertIds=[5,9], expertScales=[0.7, 0.3]）：

```
sumFloat = [0, 0, 0, 0]     (H=4, 初始化为0)

Step 1: 处理 TopK=0 (Expert 5)
  从窗口读取: winData[0×3+0] = [e5_0, e5_1, e5_2, e5_3]
  反量化(如需): → FP32
  乘以权重:    [e5_0×0.7, e5_1×0.7, e5_2×0.7, e5_3×0.7]
  累加:        sumFloat += [e5_0×0.7, e5_1×0.7, e5_2×0.7, e5_3×0.7]

Step 2: 处理 TopK=1 (Expert 9)
  从窗口读取: winData[0×3+1] = [e9_0, e9_1, e9_2, e9_3]
  反量化(如需): → FP32
  乘以权重:    [e9_0×0.3, e9_1×0.3, e9_2×0.3, e9_3×0.3]
  累加:        sumFloat += [e9_0×0.3, e9_1×0.3, e9_2×0.3, e9_3×0.3]

Step 3: 处理共享专家 (如果有)
  从窗口读取: winData[0×3+2] = [es_0, es_1, es_2, es_3]
  累加:        sumFloat += [es_0, es_1, es_2, es_3]

Step 4: 输出
  Cast FP32 → FP16
  expandOut[0×H] = sumFloat → XOut
```

**输出公式**:
```
XOut[Token0] = expertScales[0] × FFN(E5, Token0)
             + expertScales[1] × FFN(E9, Token0)
             + SharedExpert(Token0)
```

### 3.6 完整数据流示意

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Combine V2 完整数据流                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────── 发送端 (本卡) ─────────────────────────┐ │
│  │                                                                    │ │
│  │  expandX [Bs×K, H]  ──→  读取assistInfoForCombine                 │ │
│  │                           │                                        │ │
│  │                           ├─→ [toRankId=2, tokenId=0, topkId=0]   │ │
│  │                           │     expandX[0] → Rank2窗口[0×3+0]     │ │
│  │                           │     写入 Flag[0][0]=1.0               │ │
│  │                           │                                        │ │
│  │                           ├─→ [toRankId=4, tokenId=0, topkId=1]   │ │
│  │                           │     expandX[1] → Rank4窗口[0×3+1]     │ │
│  │                           │     写入 Flag[0][1]=1.0               │ │
│  │                           │                                        │ │
│  │                           └─→ ... (继续发送)                       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              ↓ AllToAllV RDMA                          │
│  ┌──────────────────────────── 接收端 (本卡) ─────────────────────────┐ │
│  │                                                                    │ │
│  │  本卡EP窗口  ←──  各Rank发来的Token数据                           │ │
│  │                                                                    │ │
│  │  WaitDispatch: 轮询状态区Flag                                      │ │
│  │    └─ Sum(Flag) ≈ K + sharedExpertNum → 数据收齐                  │ │
│  │                                                                    │ │
│  │  ProcessExpert: 加权合并                                           │ │
│  │    sumFloat = 0                                                    │ │
│  │    for topkId in 0..K-1:                                          │ │
│  │      data = 窗口[tokenIdx×(K+shared)+topkId]                      │ │
│  │      sumFloat += expertScales[topkId] × data                      │ │
│  │    for sharedExpert:                                              │ │
│  │      sumFloat += 窗口[sharedExpert数据]                            │ │
│  │                                                                    │ │
│  │  输出: XOut[tokenIndex] = FP16(sumFloat)                          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
