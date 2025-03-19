import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# 设置全局字体（这里以黑体为例，可根据实际需求调整，若不需要中文显示可省略这部分）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 12

def load_mosei_data(data_path):
    """
    加载MOSEI数据集的CSV标签文件，分析SEVEN列的分布
    """
    splits = {
        'train': 'labels_emotion/label_file_train.csv',
        'val': 'labels_emotion/label_file_valid.csv',
        'test': 'labels_emotion/label_file_test.csv'
    }

    distributions = {}
    detailed_stats = {}

    for split_name, split_path in splits.items():
        full_path = os.path.join(data_path, split_path)
        if not os.path.exists(full_path):
            print(f"找不到文件: {full_path}")
            continue

        # 读取CSV文件
        df = pd.read_csv(full_path)

        # 获取SEVEN列的连续值
        scores = df['SEVEN'].values

        # 使用a2_parse进行二分类转换
        binary_labels = np.array([a2_parse(score) for score in scores])
        distributions[split_name] = Counter(binary_labels)

        # 详细统计
        total_samples = len(scores)
        neg_samples = np.sum(binary_labels == 0)
        pos_samples = np.sum(binary_labels == 1)

        # 连续值的统计
        detailed_stats[split_name] = {
            'total': total_samples,
            'negative': neg_samples,
            'positive': pos_samples,
            'continuous_stats': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        }

        # 打印详细统计信息
        print(f"\n{split_name}集合统计:")
        print(f"总样本数: {total_samples}")
        print(f"负面样本(0)数量: {neg_samples} ({neg_samples / total_samples * 100:.2f}%)")
        print(f"正面样本(1)数量: {pos_samples} ({pos_samples / total_samples * 100:.2f}%)")
        print("\n连续值分布:")
        print(f"平均值: {detailed_stats[split_name]['continuous_stats']['mean']:.3f}")
        print(f"标准差: {detailed_stats[split_name]['continuous_stats']['std']:.3f}")
        print(f"最小值: {detailed_stats[split_name]['continuous_stats']['min']:.3f}")
        print(f"最大值: {detailed_stats[split_name]['continuous_stats']['max']:.3f}")

        # 打印分布区间统计
        bins = np.linspace(min(scores), max(scores), 10)
        hist, bin_edges = np.histogram(scores, bins=bins)
        for i in range(len(hist)):
            print(f"区间 [{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}]: {hist[i]} 样本 ({hist[i] / total_samples * 100:.2f}%)")

    return distributions, detailed_stats


def a2_parse(a):
    """
    将连续值转换为二分类标签
    """
    if a < 0:
        res = 0
    else:
        res = 1
    return res


def plot_distribution(distributions, detailed_stats):
    """
    绘制数据分布图
    """
    plt.figure(figsize=(15, 10))

    # 创建柱状图
    splits = list(distributions.keys())
    x = np.arange(len(splits))
    width = 0.35  # 由于只有正负两类，将宽度改为0.35

    # 只绘制正负两类的分布
    for i, split in enumerate(splits):
        stats = detailed_stats[split]

        # 绘制负样本和正样本的柱状图，微调颜色透明度等
        plt.bar(x[i] - width / 2, stats['negative'], width,
                label='Negative (0)' if i == 0 else "",
                color='tab:red', alpha=0.7)
        plt.bar(x[i] + width / 2, stats['positive'], width,
                label='Positive (1)' if i == 0 else "",
                color='tab:green', alpha=0.7)

        # 在柱子上方添加具体数值
        plt.text(x[i] - width / 2, stats['negative'], f"{stats['negative']}",
                 ha='center', va='bottom')
        plt.text(x[i] + width / 2, stats['positive'], f"{stats['positive']}",
                 ha='center', va='bottom')

        # 添加百分比
        total = stats['total']
        plt.text(x[i] - width / 2, stats['negative'] / 2,
                 f"{stats['negative'] / total * 100:.1f}%",
                 ha='center', va='center')
        plt.text(x[i] + width / 2, stats['positive'] / 2,
                 f"{stats['positive'] / total * 100:.1f}%",
                 ha='center', va='center')

    # 添加网格线，设置线条样式、颜色和粗细等
    plt.grid(True, linestyle='--', color='gray', alpha=0.5, linewidth=0.5)

    # 设置图表边框，例如只显示底部和左侧边框，设置颜色和粗细
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('gray')
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('gray')
    ax.spines['left'].set_linewidth(0.8)

    plt.xlabel('Dataset Split')
    plt.ylabel('Sample Count')
    plt.title('MOSEI Dataset Binary Label Distribution (SEVEN)',y = 1.05)
    plt.xticks(x, splits)
    plt.legend()

    # 添加总样本数的标注
    for i, split in enumerate(splits):
        total = detailed_stats[split]['total']
        plt.text(x[i], plt.ylim()[1], f'Total: {total}',
                 ha='center', va='bottom')

    # 保存图表到指定路径
    save_path = '/home/mz/demo/MyNet/model/data/datasets/mosei_distribution.png'
    plt.savefig(save_path)
    plt.close()


def main():
    data_path = '/home/mz/demo/TVLT/Dataset/cmumosei'

    print("开始分析MOSEI数据集分布...")

    try:
        # 加载并分析数据
        distributions, detailed_stats = load_mosei_data(data_path)

        # 绘制分布图
        plot_distribution(distributions, detailed_stats)

        # 计算整体的类别不平衡比例
        total_neg = sum(stats['negative'] for stats in detailed_stats.values())
        total_pos = sum(stats['positive'] for stats in detailed_stats.values())

        print("\n整体类别分布:")
        print(f"正负样本比例 (Positive:Negative) = {total_pos / total_neg:.2f}")

        print("\n分析完成! 分布图已保存为 '/home/mz/demo/MyNet/model/data/datasets/mosei_distribution.png'")

    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()