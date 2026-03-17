# Qwen3Attention 模块独立化方案

## 📋 目标

**只抽离 Qwen3Attention 模块**，使其能够独立运行，不依赖 vLLM 框架。

---

## 一、模块分析

### 1.1 当前 Qwen3Attention 结构

```python
class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
    )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor
```

### 1.2 包含的子组件

| 组件 | 功能 | 依赖 |
|------|------|------|
| **QKVParallelLinear** | QKV 融合投影 | vllm.model_executor.layers.linear |
| **RowParallelLinear** | 输出投影 | vllm.model_executor.layers.linear |
| **RMSNorm** | QK 归一化 | vllm.model_executor.layers.layernorm |
| **RotaryEmbedding** | 旋转位置编码 | vllm.model_executor.layers.rotary_embedding |
| **Attention** | 注意力计算 | vllm.attention.layer |

### 1.3 Forward 流程

```
hidden_states [batch, seq_len, hidden_size]
    ↓
QKVParallelLinear → q, k, v
    ↓
QK-Norm (按 head 归一化)
    ↓
RotaryEmbedding → q, k (加入位置编码)
    ↓
Attention → attn_output
    ↓
RowParallelLinear → output [batch, seq_len, hidden_size]
```

---

## 二、依赖简化方案

### 2.1 移除的依赖

| 依赖 | 处理方式 |
|------|----------|
| `get_tensor_model_parallel_world_size()` | 固定为 1（单 GPU） |
| `CacheConfig` | 简化为基本的 KV cache |
| `QuantizationConfig` | 忽略（仅支持 FP16/BF16） |
| `AttentionType.DECODER` | 简化为因果 attention |
| `dual_chunk_attention_config` | 忽略（不使用） |
| `prefix` | 移除（不需要日志） |
| `extract_layer_index` | 移除（不需要 layer index） |

### 2.2 保留的参数

```python
@dataclass
class Qwen3AttentionConfig:
    """Qwen3Attention 配置"""
    # 基础配置
    hidden_size: int              # 隐藏层维度
    num_heads: int                # 注意力头数
    num_kv_heads: int             # KV 头数（GQA）
    head_dim: int | None = None   # 头维度（默认 hidden_size/num_heads）

    # 位置编码
    rope_theta: float = 1000000   # Qwen3 使用较大的 base
    max_position: int = 4096 * 32 # 最大位置

    # 归一化
    rms_norm_eps: float = 1e-6

    # 其他
    qkv_bias: bool = False        # QKV 是否使用 bias
```

---

## 三、核心组件实现

### 3.1 RMSNorm（简化版）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RMSNorm(nn.Module):
    """简化的 RMSNorm 实现"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_size] 或 [batch, num_heads, head_dim]
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
```

### 3.2 QKVParallelLinear（简化版）

```python
class QKVParallelLinear(nn.Module):
    """简化的 QKV 融合线性层（去并行化）"""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        bias: bool = False
    ):
        super().__init__()

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        # 计算输出维度
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        total_size = q_size + 2 * kv_size

        # 单个权重矩阵融合 QKV
        self.weight = nn.Parameter(torch.empty(total_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(total_size)) if bias else None

        # 初始化权重
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # 保存切片位置
        self.q_slice = slice(0, q_size)
        self.k_slice = slice(q_size, q_size + kv_size)
        self.v_slice = slice(q_size + kv_size, total_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_size]
        # weight: [q_size + 2*kv_size, hidden_size]

        # 线性变换
        out = F.linear(x, self.weight, self.bias)  # [batch, seq_len, q_size + 2*kv_size]

        # 切分为 Q, K, V
        q = out[..., self.q_slice]  # [batch, seq_len, num_heads * head_dim]
        k = out[..., self.k_slice]  # [batch, seq_len, num_kv_heads * head_dim]
        v = out[..., self.v_slice]  # [batch, seq_len, num_kv_heads * head_dim]

        return q, k, v
```

### 3.3 RowParallelLinear（简化版）

```python
class RowParallelLinear(nn.Module):
    """简化的行并行线性层（去并行化）"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # 初始化
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, in_features]
        out = F.linear(x, self.weight, self.bias)
        return out  # [batch, seq_len, out_features]
```

### 3.4 RotaryEmbedding（简化版）

```python
class RotaryEmbedding(nn.Module):
    """简化的旋转位置编码（仅支持 default scaling）"""

    def __init__(
        self,
        head_dim: int,
        max_position: int = 4096 * 32,
        base: float = 1000000
    ):
        super().__init__()
        self.head_dim = head_dim
        self.base = base

        # 计算逆频率
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # 预计算缓存
        self.max_position_cached = max_position
        t = torch.arange(max_position, device=inv_freq.device).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: [seq_len] 或 [batch, seq_len]
            q: [batch, seq_len, num_heads, head_dim] 或 [batch, num_heads, seq_len, head_dim]
            k: 同 q
        """
        # 获取 cos/sin
        if positions.dim() == 1:
            cos = self.cos_cached[:, :, positions]  # [1, 1, seq_len, head_dim]
            sin = self.sin_cached[:, :, positions]
        else:
            # positions: [batch, seq_len]
            batch_size, seq_len = positions.shape
            cos = self.cos_cached[:, :, positions.flatten()].view(1, 1, batch_size, seq_len, -1)
            sin = self.sin_cached[:, :, positions.flatten()].view(1, 1, batch_size, seq_len, -1)

        # 应用旋转（假设 q, k 是 [batch, seq_len, num_heads, head_dim]）
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """旋转一半维度"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
```

### 3.5 SimpleAttention（简化版）

```python
class SimpleAttention(nn.Module):
    """简化的 Attention，使用 PyTorch SDPA"""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        scaling: float
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scaling

        # KV head 重复次数（GQA）
        self.repeat_times = num_heads // num_kv_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            q: [batch, seq_len, num_heads, head_dim]
            k: [batch, seq_len, num_kv_heads, head_dim]
            v: [batch, seq_len, num_kv_heads, head_dim]
            mask: [batch, seq_len, seq_len] 或 None

        Returns:
            output: [batch, seq_len, num_heads, head_dim]
        """
        batch_size, seq_len, _, _ = q.shape

        # 如果 KV heads < Q heads，需要重复 KV
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.repeat_times, dim=2)  # [batch, seq_len, num_heads, head_dim]
            v = v.repeat_interleave(self.repeat_times, dim=2)

        # 转置为 [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # PyTorch SDPA
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=True  # 因果 mask
        )

        # 转回 [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2)

        return attn_output
```

---

## 四、完整的 Qwen3Attention 实现

```python
class Qwen3Attention(nn.Module):
    """简化的 Qwen3 Attention 模块"""

    def __init__(self, config: Qwen3AttentionConfig):
        super().__init__()

        # 计算头维度
        head_dim = config.head_dim or (config.hidden_size // config.num_heads)

        # 保存配置
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

        # 子组件
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            bias=config.qkv_bias
        )

        self.o_proj = RowParallelLinear(
            in_features=config.num_heads * head_dim,
            out_features=config.hidden_size,
            bias=False
        )

        self.rotary_emb = RotaryEmbedding(
            head_dim=head_dim,
            max_position=config.max_position,
            base=config.rope_theta
        )

        self.attn = SimpleAttention(
            num_heads=config.num_heads,
            head_dim=head_dim,
            num_kv_heads=config.num_kv_heads,
            scaling=self.scaling
        )

        # QK-Norm
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            positions: [seq_len] 或 [batch, seq_len]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV 投影
        q, k, v = self.qkv_proj(hidden_states)

        # 重塑为 [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # QK-Norm（按 head 归一化）
        q = self.q_norm(q)  # [batch, seq_len, num_heads, head_dim]
        k = self.k_norm(k)  # [batch, seq_len, num_kv_heads, head_dim]

        # 旋转位置编码
        q, k = self.rotary_emb(positions, q, k)

        # Attention
        attn_output = self.attn(q, k, v)  # [batch, seq_len, num_heads, head_dim]

        # 输出投影
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)  # [batch, seq_len, hidden_size]

        return output
```

---

## 五、使用示例

### 5.1 初始化

```python
# 从 config.json 加载配置
import json

with open("config.json", "r") as f:
    hf_config = json.load(f)

config = Qwen3AttentionConfig(
    hidden_size=hf_config["hidden_size"],
    num_heads=hf_config["num_attention_heads"],
    num_kv_heads=hf_config["num_key_value_heads"],
    rope_theta=hf_config.get("rope_theta", 1000000),
    max_position=hf_config["max_position_embeddings"],
    rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6)
)

# 创建模块
attn = Qwen3Attention(config).to("cuda").to(torch.bfloat16)
```

### 5.2 Forward

```python
import torch

# 输入
batch_size = 1
seq_len = 128
hidden_size = config.hidden_size

hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)
positions = torch.arange(seq_len, device="cuda")

# 前向传播
output = attn(hidden_states, positions)

print(output.shape)  # [1, 128, hidden_size]
```

### 5.3 加载权重

```python
from safetensors.torch import load_file

# 加载 checkpoint
checkpoint = load_file("model.safetensors")

# 权重映射
def load_weights(module, checkpoint, layer_idx=0):
    \"\"\"加载权重到模块\"\"\"
    prefix = f"model.layers.{layer_idx}.self_attn"

    # QKV 权重（需要拼接）
    q_weight = checkpoint[f"{prefix}.q_proj.weight"]  # [hidden_size, q_size]
    k_weight = checkpoint[f"{prefix}.k_proj.weight"]  # [hidden_size, kv_size]
    v_weight = checkpoint[f"{prefix}.v_proj.weight"]  # [hidden_size, kv_size]

    # 拼接并转置: [q_size + 2*kv_size, hidden_size]
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=1).T
    module.qkv_proj.weight.data = qkv_weight

    # Output 权重
    o_weight = checkpoint[f"{prefix}.o_proj.weight"].T
    module.o_proj.weight.data = o_weight

    # QK-Norm 权重
    module.q_norm.weight.data = checkpoint[f"{prefix}.q_norm.weight"]
    module.k_norm.weight.data = checkpoint[f"{prefix}.k_norm.weight"]

# 加载权重
load_weights(attn, checkpoint, layer_idx=0)
```

---

## 六、完整代码

将以上所有组件组合到一个文件中：

```python
# qwen3_attention_standalone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

@dataclass
class Qwen3AttentionConfig:
    \"\"\"Qwen3Attention 配置\"\"\"
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: Optional[int] = None
    rope_theta: float = 1000000
    max_position: int = 4096 * 32
    rms_norm_eps: float = 1e-6
    qkv_bias: bool = False

# ... (包含所有上面的类定义)

def create_qwen3_attention_from_hf_config(hf_config: dict) -> Qwen3Attention:
    \"\"\"从 HuggingFace config 创建 Qwen3Attention\"\"\"
    config = Qwen3AttentionConfig(
        hidden_size=hf_config["hidden_size"],
        num_heads=hf_config["num_attention_heads"],
        num_kv_heads=hf_config["num_key_value_heads"],
        head_dim=hf_config.get("head_dim"),
        rope_theta=hf_config.get("rope_theta", 1000000),
        max_position=hf_config["max_position_embeddings"],
        rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6)
    )
    return Qwen3Attention(config)
```

---

## 七、测试验证

### 7.1 单元测试

```python
def test_qwen3_attention():
    # 配置
    config = Qwen3AttentionConfig(
        hidden_size=5120,
        num_heads=40,
        num_kv_heads=8,
        head_dim=128
    )

    # 创建模块
    attn = Qwen3Attention(config)
    attn.eval()  # 测试模式

    # 输入
    batch_size = 2
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    positions = torch.arange(seq_len)

    # 前向传播
    with torch.no_grad():
        output = attn(hidden_states, positions)

    # 验证输出形状
    assert output.shape == (batch_size, seq_len, config.hidden_size)

    print("✅ 测试通过！")

if __name__ == "__main__":
    test_qwen3_attention()
```

### 7.2 对比测试

```python
def compare_with_vllm():
    \"\"\"与 vLLM 输出对比\"\"\"
    # TODO: 加载相同的权重，对比输出是否一致
    pass
```

---

## 八、总结

### 8.1 代码规模

| 组件 | 代码行数 |
|------|----------|
| Qwen3AttentionConfig | ~10 |
| RMSNorm | ~15 |
| QKVParallelLinear | ~40 |
| RowParallelLinear | ~25 |
| RotaryEmbedding | ~50 |
| SimpleAttention | ~40 |
| Qwen3Attention | ~50 |
| **总计** | **~230 行** |

### 8.2 预计工作量

- **实现时间：** 1-2 天
- **测试时间：** 0.5-1 天
- **总计：** 1.5-3 天

### 8.3 关键优势

✅ **极简代码**：只需 ~230 行
✅ **独立运行**：不依赖 vLLM
✅ **易于测试**：接口清晰
✅ **高性能**：使用 PyTorch SDPA
✅ **完整功能**：包含所有 Qwen3 特性（QK-Norm、RoPE、GQA）

### 8.4 使用场景

1. **研究和实验**：快速测试不同的 attention 变体
2. **模型分析**：单独分析 attention 层的行为
3. **教学演示**：展示 attention 机制
4. **独立部署**：只使用 attention 模块

---

## 九、下一步

1. **实现代码**：将上述代码实现到 `qwen3_attention_standalone.py`
2. **单元测试**：编写完整的测试用例
3. **性能测试**：对比 vLLM 的性能
4. **文档完善**：添加使用说明和示例

---

**文档版本：** v1.0
**创建时间：** 2026-03-12
**状态：** 待实施
