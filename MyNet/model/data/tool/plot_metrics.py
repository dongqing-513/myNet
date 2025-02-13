import re
import matplotlib.pyplot as plt
import numpy as np
from math import ceil


png_name = 'metrics_plot0211.png'


def extract_metrics(log_file):
    # 使用字典来存储每个 epoch 的所有值和最佳值
    epoch_metrics = {}  # 存储每个 epoch 的所有值
    epoch_best_metrics = {}  # 存储每个 epoch 的最佳值
    epoch_losses = {}  # 存储每个 epoch 的损失值
    epoch_f1_metrics = {}  # 存储每个 epoch 的f1值
    current_epoch = None  # 当前正在处理的 epoch
    log_lines = []  # 存储日志文件的所有行

    with open(log_file, 'r') as f:
        log_lines = f.readlines()

    for line in log_lines:
        # 首先尝试获取当前 epoch
        epoch_match = re.search(r"Epoch (\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            if current_epoch not in epoch_metrics:
                epoch_metrics[current_epoch] = []
                epoch_best_metrics[current_epoch] = None
                epoch_losses[current_epoch] = []
                epoch_f1_metrics[current_epoch] = []

        # 检查是否是 best val/the_metric
        val_match = re.search(r"'val/the_metric' reached ([\d.]+)", line)
        if val_match and current_epoch is not None:
            metric_value = float(val_match.group(1))
            # 向上取整到小数点后 4 位
            metric_value = ceil(metric_value * 10000) / 10000
            epoch_metrics[current_epoch].append(metric_value)
            # 更新该 epoch 的最佳值
            if epoch_best_metrics[current_epoch] is None or metric_value > epoch_best_metrics[current_epoch]:
                epoch_best_metrics[current_epoch] = metric_value

        # 检查进度条 100% 时的 mosei/val/accuracy2_epoch
        accuracy_match = re.search(r"mosei/val/accuracy2_epoch tensor\(([\d.]+)", line)
        if accuracy_match and current_epoch is not None:
            metric_value = float(accuracy_match.group(1))
            # 向上取整到小数点后 4 位
            metric_value = ceil(metric_value * 10000) / 10000
            epoch_metrics[current_epoch].append(metric_value)

        # 检查 mosei/val/f1_epoch
        f1_match = re.search(r"mosei/val/f1_epoch tensor\(([\d.]+)", line)
        if f1_match and current_epoch is not None:
            f1_value = float(f1_match.group(1))
            # 向上取整到小数点后 4 位
            f1_value = ceil(f1_value * 10000) / 10000
            epoch_f1_metrics[current_epoch].append(f1_value)

        # 检查 mosei/val/loss_epoch 中的损失值
        loss_match = re.search(r"mosei/val/loss_epoch tensor\(([\d.]+)", line)
        if loss_match and current_epoch is not None:
            loss_value = float(loss_match.group(1))
            # 向上取整到小数点后 4 位
            loss_value = ceil(loss_value * 10000) / 10000
            epoch_losses[current_epoch].append(loss_value)

    # 处理每个epoch的指标
    for epoch in epoch_metrics:
        # 处理accuracy
        if epoch_best_metrics[epoch] is None:
            if epoch_metrics[epoch]:  # 如果有收集到accuracy值
                epoch_best_metrics[epoch] = max(epoch_metrics[epoch])
            else:
                epoch_best_metrics[epoch] = 0

        # 处理f1
        if epoch_f1_metrics[epoch]:
            epoch_f1_metrics[epoch] = max(epoch_f1_metrics[epoch])
        else:
            epoch_f1_metrics[epoch] = 0

        # 处理loss
        if epoch_losses[epoch]:
            epoch_losses[epoch] = min(epoch_losses[epoch])
        else:
            epoch_losses[epoch] = 0

    # 将字典转换为有序列表
    epochs = sorted(epoch_best_metrics.keys())
    val_metrics = [epoch_best_metrics[epoch] for epoch in epochs]
    losses = [epoch_losses[epoch] for epoch in epochs]
    f1_metrics = [epoch_f1_metrics[epoch] for epoch in epochs]
    return epochs, val_metrics, losses, f1_metrics


def plot_metrics(epochs, val_metrics, losses, f1_metrics):
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8')

    # 创建更密集的点以实现平滑效果
    x_smooth = np.linspace(min(epochs), max(epochs), 300)
    y_smooth_val = np.interp(x_smooth, epochs, val_metrics)
    y_smooth_loss = np.interp(x_smooth, epochs, losses)
    y_smooth_f1 = np.interp(x_smooth, epochs, f1_metrics)

    # 绘制原始验证指标数据点
    plt.plot(epochs, val_metrics, 'o', label='Accuracy Data Points',
             color='#e74c3c', markersize=8, alpha=0.5)
    # 绘制平滑验证指标曲线
    plt.plot(x_smooth, y_smooth_val, '-', label='Smoothed Accuracy Curve',
             color='#e74c3c', linewidth=2)
    
    # 绘制原始F1数据点
    plt.plot(epochs, f1_metrics, 'o', label='F1 Data Points',
             color='#2ecc71', markersize=8, alpha=0.5)
    # 绘制平滑F1曲线
    plt.plot(x_smooth, y_smooth_f1, '-', label='Smoothed F1 Curve',
             color='#2ecc71', linewidth=2)

    # 绘制原始损失数据点
    plt.plot(epochs, losses, 'o', label='Loss Data Points',
             color='#3498db', markersize=8, alpha=0.5)
    # 绘制平滑损失曲线
    plt.plot(x_smooth, y_smooth_loss, '-', label='Smoothed Loss Curve',
             color='#3498db', linewidth=2)

    # Customize the plot
    plt.title('Test Metrics and Loss Over Epochs', fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Metric/Loss Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # 设置 y 轴精度为 4 位小数
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))

    # Add some padding to the plot
    plt.margins(x=0.02)

    # Customize the axes
    plt.tick_params(axis='both', which='major', labelsize=10)

    # 在图表中显示准确率和F1的具体数值
    for epoch, metric, f1 in zip(epochs, val_metrics, f1_metrics):
        plt.text(epoch, metric, f'{metric:.4f}', ha='center', va='bottom')
        plt.text(epoch, f1, f'{f1:.4f}', ha='center', va='top')

    # Save the plot
    plt.savefig(png_name, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    log_file = '/home/mz/demo/MyNet/0211fortextlstm.log'
    print("正在从日志文件中提取指标...")
    epochs, val_metrics, losses, f1_metrics = extract_metrics(log_file)

    if not epochs or not val_metrics:
        print("在日志文件中没有找到指标！")
        return

    print("\n每个 epoch 的最佳验证指标值：")
    for epoch, metric in zip(epochs, val_metrics):
        print(f"Epoch {epoch}: {metric:.4f}")

    print("\n每个 epoch 的最佳F1值：")
    for epoch, f1 in zip(epochs, f1_metrics):
        print(f"Epoch {epoch}: {f1:.4f}")

    print("\n每个 epoch 的最小损失值：")
    for epoch, loss in zip(epochs, losses):
        print(f"Epoch {epoch}: {loss:.4f}")

    print(f"\n找到了 {len(epochs)} 个 epoch 的数据")
    print("正在创建图表...")
    plot_metrics(epochs, val_metrics, losses, f1_metrics)
    print("图表已保存为: ", png_name)


if __name__ == '__main__':
    main()