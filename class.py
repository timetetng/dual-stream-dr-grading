import matplotlib.pyplot as plt
import numpy as np

def generate_ablation_chart():
    # ---------------------------------------------------------
    # 全局学术风格配置 (Academic Style Configuration)
    # ---------------------------------------------------------
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11

    # ---------------------------------------------------------
    # 实验数据 (请替换为您实际实验的输出结果)
    # ---------------------------------------------------------
    strategies = ['Baseline (No Weight)', 'Inverse Weight ($1/N_c$)', 'Inverse Sqrt ($1/\sqrt{N_c}$)']
    
    # 典型场景下的表现模拟：
    # 1. 无权重：整体QWK尚可，但少数类PDR Recall极差
    # 2. 绝对反比：PDR Recall飙升，但由于严重过拟合，导致整体QWK和Macro F1下降
    # 3. 平方根反比(Ours)：在PDR Recall和整体泛化性能间取得最佳平衡
    val_qwk = [0.765, 0.742, 0.841]
    macro_f1 = [0.621, 0.710, 0.805]
    pdr_recall = [0.420, 0.850, 0.795]

    x = np.arange(len(strategies))
    width = 0.25  # 柱子宽度

    # 学术配色
    color_qwk = '#4C72B0'    # 稳重蓝
    color_f1 = '#55A868'     # 护眼绿
    color_recall = '#C44E52' # 砖红色

    fig, ax = plt.subplots(figsize=(9, 6))

    # 绘制分组柱状图
    bar1 = ax.bar(x - width, val_qwk, width, label='Overall Val QWK', color=color_qwk, edgecolor='black', linewidth=1.2)
    bar2 = ax.bar(x, macro_f1, width, label='Macro F1 Score', color=color_f1, edgecolor='black', linewidth=1.2)
    bar3 = ax.bar(x + width, pdr_recall, width, label='PDR (Grade 4) Recall', color=color_recall, edgecolor='black', linewidth=1.2)

    # 设置标签与刻度
    ax.set_ylabel('Performance Metric Score')
    ax.set_title('Ablation Study of Resampling Strategies')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0, 1.05) # 留出顶部空间给数值标签

    # 隐藏右侧和顶部边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 添加图例
    ax.legend(loc='upper left', frameon=False)

    # 在柱子上添加数值标注
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直偏移3个像素
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    add_labels(bar1)
    add_labels(bar2)
    add_labels(bar3)

    plt.tight_layout()
    plt.savefig('resampling_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("图表已生成：resampling_ablation.png")

if __name__ == '__main__':
    generate_ablation_chart()
