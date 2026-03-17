"""
PagedAttention Diagrams Generator
生成时序图和流程图，用于飞书文档
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.font_manager as fm
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'primary': '#3498db',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#9b59b6',
    'light': '#ecf0f1',
    'dark': '#2c3e50'
}

def create_pagedattention_workflow():
    """创建 PagedAttention 完整工作流程图"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    ax.text(5, 9.5, 'PagedAttention 工作流程',
            fontsize=24, fontweight='bold', ha='center', color=COLORS['dark'])

    # 定义阶段
    stages = [
        {'name': '初始化阶段', 'pos': (2, 8), 'color': COLORS['info'],
         'items': ['计算 GPU 可用块数量', '分配 KV Cache 内存', '初始化 Block Manager']},
        {'name': 'Prefill 阶段', 'pos': (5, 8), 'color': COLORS['primary'],
         'items': ['计算需要的块数量', '分配物理块', '写入 KV Cache', '计算 Attention']},
        {'name': 'Decode 阶段', 'pos': (8, 8), 'color': COLORS['success'],
         'items': ['选择 V1 或 V2 算法', '计算 Attention', '生成新 token']},
    ]

    # 绘制阶段
    for i, stage in enumerate(stages):
        # 阶段框
        box = FancyBboxPatch((stage['pos'][0]-0.8, stage['pos'][1]-1.2),
                             1.6, 1.4,
                             boxstyle="round,pad=0.1",
                             edgecolor=stage['color'], facecolor=stage['color'],
                             alpha=0.3, linewidth=2)
        ax.add_patch(box)

        # 阶段名称
        ax.text(stage['pos'][0], stage['pos'][1]+0.3, stage['name'],
               fontsize=14, fontweight='bold', ha='center', va='center',
               color=COLORS['dark'])

        # 步骤列表
        for j, item in enumerate(stage['items']):
            ax.text(stage['pos'][0], stage['pos'][1]-0.1-j*0.3, f"{j+1}. {item}",
                   fontsize=9, ha='center', va='center', color=COLORS['dark'])

    # 绘制内存管理阶段
    memory_box = FancyBboxPatch((4, 5.2), 2, 1.4,
                               boxstyle="round,pad=0.1",
                               edgecolor=COLORS['warning'], facecolor=COLORS['warning'],
                               alpha=0.3, linewidth=2)
    ax.add_patch(memory_box)
    ax.text(5, 6.2, '内存管理阶段', fontsize=14, fontweight='bold',
           ha='center', color=COLORS['dark'])
    ax.text(5, 5.8, '块分配 / 块释放', fontsize=10, ha='center', color=COLORS['dark'])
    ax.text(5, 5.5, 'Prefix Caching', fontsize=10, ha='center', color=COLORS['dark'])

    # 绘制箭头连接
    arrows = [
        ((2.8, 8), (4.2, 8)),
        ((5.8, 8), (7.2, 8)),
        ((5, 7.6), (5, 6.6)),
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(start, end,
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color=COLORS['dark'])
        ax.add_patch(arrow)

    # 添加关键组件说明
    components = [
        ('Block Manager', (1.5, 4), COLORS['primary']),
        ('Block Table', (3.5, 4), COLORS['success']),
        ('KV Cache', (5.5, 4), COLORS['warning']),
        ('物理块池', (7.5, 4), COLORS['info']),
    ]

    for name, pos, color in components:
        box = FancyBboxPatch((pos[0]-0.6, pos[1]-0.3), 1.2, 0.6,
                           boxstyle="round,pad=0.05",
                           edgecolor=color, facecolor=color, alpha=0.5)
        ax.add_patch(box)
        ax.text(pos[0], pos[1], name, fontsize=10, fontweight='bold',
               ha='center', va='center', color=COLORS['dark'])

    # 添加数据流说明
    ax.text(5, 3, '数据流', fontsize=16, fontweight='bold', ha='center',
           color=COLORS['dark'])
    ax.text(5, 2.5, '请求 → 块分配 → KV Cache → Attention → 响应',
           fontsize=12, ha='center', color=COLORS['dark'],
           bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['light'], alpha=0.5))

    # 添加性能指标
    metrics = [
        '内存利用率: 95%+',
        '吞吐量提升: 2-4倍',
        'GPU 利用率: 90%+',
        '延迟降低: 50-70%'
    ]

    ax.text(5, 1.2, '性能提升', fontsize=14, fontweight='bold', ha='center',
           color=COLORS['dark'])
    for i, metric in enumerate(metrics):
        ax.text(1.5 + i*2.2, 0.6, metric, fontsize=10,
               ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['success'], alpha=0.3))

    plt.tight_layout()
    plt.savefig('E:/CodeHUb/vllm-main/pagedattention_workflow.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('E:/CodeHUb/vllm-main/pagedattention_workflow.pdf',
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ PagedAttention 工作流程图已生成")

def create_continuous_batching_flow():
    """创建 Continuous Batching 流程图"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    ax.text(5, 9.5, 'Continuous Batching 工作流程',
            fontsize=24, fontweight='bold', ha='center', color=COLORS['dark'])

    # 定义步骤
    steps = [
        {'name': '移除已完成请求', 'pos': (2, 8), 'color': COLORS['danger'],
         'desc': '释放 KV Cache 块'},
        {'name': '添加新请求', 'pos': (5, 8), 'color': COLORS['primary'],
         'desc': '分配块到等待队列'},
        {'name': '执行推理', 'pos': (8, 8), 'color': COLORS['success'],
         'desc': '批量处理请求'},
        {'name': '更新 KV Cache', 'pos': (5, 6.5), 'color': COLORS['warning'],
         'desc': '写入新 tokens'},
    ]

    # 绘制步骤
    for i, step in enumerate(steps):
        box = FancyBboxPatch((step['pos'][0]-0.8, step['pos'][1]-0.6),
                            1.6, 1.2,
                            boxstyle="round,pad=0.1",
                            edgecolor=step['color'], facecolor=step['color'],
                            alpha=0.3, linewidth=2)
        ax.add_patch(box)

        ax.text(step['pos'][0], step['pos'][1]+0.2, step['name'],
               fontsize=12, fontweight='bold', ha='center', color=COLORS['dark'])
        ax.text(step['pos'][0], step['pos'][1]-0.2, step['desc'],
               fontsize=9, ha='center', color=COLORS['dark'])

    # 绘制循环箭头
    loop_arrow = FancyArrowPatch((3, 8), (2, 8.3),
                                arrowstyle='->', mutation_scale=20,
                                linewidth=2, color=COLORS['dark'], connectionstyle="arc3,rad=.3")
    ax.add_patch(loop_arrow)

    # 绘制连接箭头
    arrows = [
        ((2.8, 8), (4.2, 8)),
        ((5.8, 8), (7.2, 8)),
        ((8, 7.4), (5.8, 6.8)),
        ((5, 6.5), (2.2, 7.6)),
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(start, end,
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color=COLORS['dark'])
        ax.add_patch(arrow)

    # 添加对比表格
    ax.text(5, 5, 'Static Batching vs Continuous Batching',
           fontsize=16, fontweight='bold', ha='center', color=COLORS['dark'])

    # 表格数据
    metrics_data = [
        ['指标', 'Static Batching', 'Continuous Batching'],
        ['吞吐量', '100 req/s', '300+ req/s'],
        ['P99 延迟', '高', '低'],
        ['GPU 利用率', '60-70%', '90%+'],
    ]

    # 绘制表格
    table = ax.table(cellText=metrics_data, cellLoc='center',
                    loc='center', colWidths=[0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 设置表格样式
    for i, row in enumerate(metrics_data):
        for j, cell_text in enumerate(row):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor(COLORS['primary'])
                cell.set_text_props(weight='bold', color='white')
            elif j == 0:
                cell.set_facecolor(COLORS['light'])
                cell.set_text_props(weight='bold')
            elif j == 2:
                cell.set_facecolor(COLORS['success'])
                cell.set_alpha(0.3)

    # 移动表格位置
    table.set_pos([2.5, 1.5, 5, 2.5])

    # 添加关键优势
    advantages = [
        '✓ 动态管理 batch',
        '✓ 及时释放资源',
        '✓ 持续处理请求',
        '✓ 高 GPU 利用率'
    ]

    ax.text(8.5, 3.5, '关键优势', fontsize=12, fontweight='bold',
           ha='center', color=COLORS['dark'])
    for i, adv in enumerate(advantages):
        ax.text(8.5, 3-i*0.4, adv, fontsize=10, ha='center',
               color=COLORS['success'], fontweight='bold')

    plt.tight_layout()
    plt.savefig('E:/CodeHUb/vllm-main/continuous_batching.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('E:/CodeHUb/vllm-main/continuous_batching.pdf',
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Continuous Batching 流程图已生成")

def create_block_allocation_flow():
    """创建块分配流程图"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    ax.text(5, 9.5, '块分配流程',
            fontsize=24, fontweight='bold', ha='center', color=COLORS['dark'])

    # 定义流程节点
    nodes = [
        {'text': '新请求到达', 'pos': (5, 8.5), 'shape': 'diamond', 'color': COLORS['info']},
        {'text': '计算所需块数', 'pos': (5, 7.5), 'shape': 'rect', 'color': COLORS['primary']},
        {'text': '检查可用块', 'pos': (5, 6.5), 'shape': 'diamond', 'color': COLORS['warning']},
        {'text': '分配物理块', 'pos': (3, 5.5), 'shape': 'rect', 'color': COLORS['success']},
        {'text': '触发抢占', 'pos': (7, 5.5), 'shape': 'rect', 'color': COLORS['danger']},
        {'text': '更新 Block Table', 'pos': (3, 4.5), 'shape': 'rect', 'color': COLORS['primary']},
        {'text': '请求加入运行队列', 'pos': (3, 3.5), 'shape': 'rect', 'color': COLORS['success']},
        {'text': '等待资源释放', 'pos': (7, 4.5), 'shape': 'rect', 'color': COLORS['warning']},
    ]

    # 绘制节点
    for node in nodes:
        if node['shape'] == 'diamond':
            # 菱形（判断节点）
            diamond = mpatches.RegularPolygon(node['pos'], 4, radius=0.6,
                                            orientation=np.pi/4,
                                            edgecolor=node['color'], facecolor=node['color'],
                                            alpha=0.3, linewidth=2)
            ax.add_patch(diamond)
        else:
            # 矩形（处理节点）
            box = FancyBboxPatch((node['pos'][0]-0.7, node['pos'][1]-0.3),
                                1.4, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor=node['color'], facecolor=node['color'],
                                alpha=0.3, linewidth=2)
            ax.add_patch(box)

        ax.text(node['pos'][0], node['pos'][1], node['text'],
               fontsize=10, fontweight='bold', ha='center', va='center',
               color=COLORS['dark'])

    # 绘制连接箭头
    arrows = [
        ((5, 7.9), (5, 7.8)),
        ((5, 7.2), (5, 7.1)),
        ((5, 6.2), (5, 6.0)),  # 检查块到判断
        ((4.5, 6.0), (3, 5.8)),  # 有块 → 分配
        ((5.5, 6.0), (7, 5.8)),  # 无块 → 抢占
        ((3, 5.2), (3, 4.8)),  # 分配 → 更新表
        ((3, 4.2), (3, 3.8)),  # 更新表 → 运行队列
        ((7, 5.2), (7, 4.8)),  # 抢占 → 等待
        ((7, 4.2), (3.8, 3.6)),  # 等待后回到分配
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(start, end,
                               arrowstyle='->', mutation_scale=15,
                               linewidth=1.5, color=COLORS['dark'])
        ax.add_patch(arrow)

    # 添加判断标签
    ax.text(4.2, 6.1, '有足够块', fontsize=8, color=COLORS['success'], fontweight='bold')
    ax.text(5.8, 6.1, '块不足', fontsize=8, color=COLORS['danger'], fontweight='bold')

    # 添加关键概念说明
    concepts = [
        ('Block Size', '每个块包含的 token 数\n(1/8/16/32/64/128)', (1.5, 2.5)),
        ('Block Pool', 'GPU/CPU 块池管理\n动态分配释放', (3.5, 2.5)),
        ('Prefix Caching', '自动共享公共前缀\n减少重复计算', (5.5, 2.5)),
        ('Swap Space', 'CPU 交换空间\n处理内存不足', (7.5, 2.5)),
    ]

    for title, desc, pos in concepts:
        box = FancyBboxPatch((pos[0]-0.6, pos[1]-0.4), 1.2, 0.8,
                            boxstyle="round,pad=0.05",
                            edgecolor=COLORS['primary'], facecolor=COLORS['primary'],
                            alpha=0.2, linewidth=1.5)
        ax.add_patch(box)
        ax.text(pos[0], pos[1]+0.15, title, fontsize=9, fontweight='bold',
               ha='center', color=COLORS['dark'])
        ax.text(pos[0], pos[1]-0.15, desc, fontsize=7,
               ha='center', va='center', color=COLORS['dark'])

    # 添加公式
    ax.text(5, 1.2, '块数计算公式', fontsize=12, fontweight='bold',
           ha='center', color=COLORS['dark'])
    ax.text(5, 0.7, 'num_blocks = (num_tokens + block_size - 1) // block_size',
           fontsize=11, ha='center', color=COLORS['primary'],
           bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['light'], alpha=0.7))

    plt.tight_layout()
    plt.savefig('E:/CodeHUb/vllm-main/block_allocation.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('E:/CodeHUb/vllm-main/block_allocation.pdf',
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 块分配流程图已生成")

def create_prefix_caching_flow():
    """创建 Prefix Caching 流程图"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    ax.text(5, 9.5, 'Prefix Caching 工作流程',
            fontsize=24, fontweight='bold', ha='center', color=COLORS['dark'])

    # 示例请求
    ax.text(5, 8.8, '示例场景', fontsize=14, fontweight='bold',
           ha='center', color=COLORS['dark'])

    # 请求 A
    req_a_box = FancyBboxPatch((1, 8), 3.5, 0.5,
                               boxstyle="round,pad=0.1",
                               edgecolor=COLORS['primary'], facecolor=COLORS['primary'],
                               alpha=0.2, linewidth=2)
    ax.add_patch(req_a_box)
    ax.text(2.75, 8.25, '请求 A: "You are a helpful assistant. Explain quantum computing."',
           fontsize=10, ha='center', va='center', color=COLORS['dark'])

    # 请求 B
    req_b_box = FancyBboxPatch((5.5, 8), 3.5, 0.5,
                               boxstyle="round,pad=0.1",
                               edgecolor=COLORS['success'], facecolor=COLORS['success'],
                               alpha=0.2, linewidth=2)
    ax.add_patch(req_b_box)
    ax.text(7.25, 8.25, '请求 B: "You are a helpful assistant. What is the capital?"',
           fontsize=10, ha='center', va='center', color=COLORS['dark'])

    # 共享部分
    shared_box = FancyBboxPatch((3.5, 7.2), 3, 0.5,
                               boxstyle="round,pad=0.1",
                               edgecolor=COLORS['warning'], facecolor=COLORS['warning'],
                               alpha=0.4, linewidth=3)
    ax.add_patch(shared_box)
    ax.text(5, 7.45, '共享前缀: "You are a helpful assistant."',
           fontsize=10, fontweight='bold', ha='center', va='center',
           color=COLORS['dark'])

    # 流程步骤
    steps = [
        {'name': '1. 计算块哈希', 'pos': (2, 6), 'color': COLORS['info'],
         'desc': '为每个 Block 计算内容哈希\n使用 SHA256 或 SHA256_CBOR'},
        {'name': '2. 查找缓存', 'pos': (5, 6), 'color': COLORS['primary'],
         'desc': '在缓存中查找最长匹配前缀\n返回已缓存的块'},
        {'name': '3. 共享块', 'pos': (8, 6), 'color': COLORS['success'],
         'desc': '多个请求指向相同物理块\n无需重复计算'},
        {'name': '4. 新增块', 'pos': (5, 4.5), 'color': COLORS['warning'],
         'desc': '为非共享部分分配新块\n写入 KV Cache'},
    ]

    for step in steps:
        box = FancyBboxPatch((step['pos'][0]-0.8, step['pos'][1]-0.5),
                            1.6, 1.0,
                            boxstyle="round,pad=0.1",
                            edgecolor=step['color'], facecolor=step['color'],
                            alpha=0.3, linewidth=2)
        ax.add_patch(box)
        ax.text(step['pos'][0], step['pos'][1]+0.2, step['name'],
               fontsize=11, fontweight='bold', ha='center', color=COLORS['dark'])
        ax.text(step['pos'][0], step['pos'][1]-0.2, step['desc'],
               fontsize=8, ha='center', va='center', color=COLORS['dark'])

    # 连接箭头
    arrows = [
        ((2.8, 6), (4.2, 6)),
        ((5.8, 6), (7.2, 6)),
        ((8, 5.5), (5.8, 4.8)),
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(start, end,
                               arrowstyle='->', mutation_scale=15,
                               linewidth=2, color=COLORS['dark'])
        ax.add_patch(arrow)

    # 性能提升数据
    ax.text(5, 3.5, '性能提升', fontsize=14, fontweight='bold',
           ha='center', color=COLORS['dark'])

    metrics = [
        ('TTFT 延迟', '3-5倍提升', COLORS['success']),
        ('内存占用', '减少 50-70%', COLORS['primary']),
        ('计算量', '减少重复计算', COLORS['warning']),
    ]

    for i, (metric, value, color) in enumerate(metrics):
        pos_x = 2 + i * 3
        box = FancyBboxPatch((pos_x-0.7, 2.7), 1.4, 0.6,
                            boxstyle="round,pad=0.05",
                            edgecolor=color, facecolor=color,
                            alpha=0.3, linewidth=2)
        ax.add_patch(box)
        ax.text(pos_x, 3.05, metric, fontsize=10, fontweight='bold',
               ha='center', color=COLORS['dark'])
        ax.text(pos_x, 2.85, value, fontsize=9,
               ha='center', color=color, fontweight='bold')

    # 适用场景
    scenarios = [
        '✓ 长系统提示词 (1000+ tokens)',
        '✓ 多问题共享同一文档',
        '✓ 对话历史重放',
        '✓ 批量相似请求处理'
    ]

    ax.text(5, 2, '适用场景', fontsize=12, fontweight='bold',
           ha='center', color=COLORS['dark'])
    for i, scenario in enumerate(scenarios):
        ax.text(1.5 + i * 2.2, 1.4, scenario, fontsize=9,
               ha='center', color=COLORS['success'])

    plt.tight_layout()
    plt.savefig('E:/CodeHUb/vllm-main/prefix_caching.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('E:/CodeHUb/vllm-main/prefix_caching.pdf',
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Prefix Caching 流程图已生成")

def create_attention_version_flow():
    """创建 PagedAttention V1/V2 选择逻辑流程图"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    ax.text(5, 9.5, 'PagedAttention V1/V2 选择逻辑',
            fontsize=24, fontweight='bold', ha='center', color=COLORS['dark'])

    # 定义流程节点
    nodes = [
        {'text': 'Decode 阶段开始', 'pos': (5, 8.5), 'color': COLORS['info']},
        {'text': '检查序列长度', 'pos': (5, 7.5), 'color': COLORS['primary']},
        {'text': 'max_seq_len ≤ 8192?', 'pos': (5, 6.5), 'shape': 'diamond', 'color': COLORS['warning']},
        {'text': 'PagedAttention V1', 'pos': (3, 5.5), 'color': COLORS['success']},
        {'text': '计算分区数', 'pos': (7, 5.5), 'color': COLORS['warning']},
        {'text': 'num_partitions = 1\nOR\nnum_seqs × heads > 512?', 'pos': (7, 4.5), 'shape': 'diamond', 'color': COLORS['warning']},
        {'text': 'PagedAttention V1', 'pos': (5.5, 3.5), 'color': COLORS['success']},
        {'text': 'PagedAttention V2', 'pos': (8.5, 3.5), 'color': COLORS['danger']},
    ]

    # 绘制节点
    for node in nodes:
        shape = node.get('shape', 'rect')
        if shape == 'diamond':
            diamond = mpatches.RegularPolygon(node['pos'], 4, radius=0.7,
                                            orientation=np.pi/4,
                                            edgecolor=node['color'], facecolor=node['color'],
                                            alpha=0.3, linewidth=2)
            ax.add_patch(diamond)
        else:
            box = FancyBboxPatch((node['pos'][0]-0.8, node['pos'][1]-0.4),
                                1.6, 0.8,
                                boxstyle="round,pad=0.05",
                                edgecolor=node['color'], facecolor=node['color'],
                                alpha=0.3, linewidth=2)
            ax.add_patch(box)

        ax.text(node['pos'][0], node['pos'][1], node['text'],
               fontsize=9, fontweight='bold', ha='center', va='center',
               color=COLORS['dark'])

    # 绘制连接箭头
    arrows = [
        ((5, 8.1), (5, 7.9)),
        ((5, 7.1), (5, 7.2)),
        ((4.3, 6.5), (3, 5.9)),  # 短序列 → V1
        ((5.7, 6.5), (7, 5.9)),  # 长序列 → 计算分区
        ((7, 5.1), (7, 4.9)),
        ((6.5, 4.5), (5.5, 3.9)),  # 条件满足 → V1
        ((7.5, 4.5), (8.5, 3.9)),  # 条件不满足 → V2
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(start, end,
                               arrowstyle='->', mutation_scale=15,
                               linewidth=1.5, color=COLORS['dark'])
        ax.add_patch(arrow)

    # 添加判断标签
    ax.text(4, 6.7, '是', fontsize=9, color=COLORS['success'], fontweight='bold')
    ax.text(6, 6.7, '否', fontsize=9, color=COLORS['danger'], fontweight='bold')
    ax.text(6.2, 4.7, '是', fontsize=9, color=COLORS['success'], fontweight='bold')
    ax.text(7.8, 4.7, '否', fontsize=9, color=COLORS['danger'], fontweight='bold')

    # 添加算法特点对比
    ax.text(5, 2.5, '算法特点对比', fontsize=14, fontweight='bold',
           ha='center', color=COLORS['dark'])

    # V1 特点
    v1_box = FancyBboxPatch((1.5, 0.8), 3, 1.5,
                           boxstyle="round,pad=0.1",
                           edgecolor=COLORS['success'], facecolor=COLORS['success'],
                           alpha=0.2, linewidth=2)
    ax.add_patch(v1_box)
    ax.text(3, 2.1, 'PagedAttention V1', fontsize=12, fontweight='bold',
           ha='center', color=COLORS['success'])
    v1_features = [
        '✓ 短序列 (≤8192)',
        '✓ 大批量',
        '✓ 直接计算',
        '✓ 少共享内存',
    ]
    for i, feature in enumerate(v1_features):
        ax.text(3, 1.7 - i*0.25, feature, fontsize=9,
               ha='center', color=COLORS['dark'])

    # V2 特点
    v2_box = FancyBboxPatch((5.5, 0.8), 3, 1.5,
                           boxstyle="round,pad=0.1",
                           edgecolor=COLORS['danger'], facecolor=COLORS['danger'],
                           alpha=0.2, linewidth=2)
    ax.add_patch(v2_box)
    ax.text(7, 2.1, 'PagedAttention V2', fontsize=12, fontweight='bold',
           ha='center', color=COLORS['danger'])
    v2_features = [
        '✓ 长序列 (>8192)',
        '✓ 分区归约',
        '✓ 512 tokens/分区',
        '✓ 多临时存储',
    ]
    for i, feature in enumerate(v2_features):
        ax.text(7, 1.7 - i*0.25, feature, fontsize=9,
               ha='center', color=COLORS['dark'])

    plt.tight_layout()
    plt.savefig('E:/CodeHUb/vllm-main/attention_version_flow.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('E:/CodeHUb/vllm-main/attention_version_flow.pdf',
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ PagedAttention 版本选择流程图已生成")

if __name__ == '__main__':
    print("开始生成 PagedAttention 图表...")
    print()

    create_pagedattention_workflow()
    create_continuous_batching_flow()
    create_block_allocation_flow()
    create_prefix_caching_flow()
    create_attention_version_flow()

    print()
    print("=" * 60)
    print("所有图表生成完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("✓ pagedattention_workflow.png/pdf - PagedAttention 完整工作流程")
    print("✓ continuous_batching.png/pdf - Continuous Batching 流程")
    print("✓ block_allocation.png/pdf - 块分配流程")
    print("✓ prefix_caching.png/pdf - Prefix Caching 工作流程")
    print("✓ attention_version_flow.png/pdf - PagedAttention V1/V2 选择逻辑")
    print("\n所有图表均为高清 PNG 和 PDF 格式，可直接插入飞书文档")
