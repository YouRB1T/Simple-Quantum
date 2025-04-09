import math
import matplotlib.pyplot as plt

def plots_graphs(*data_sets, titles, xlabels, ylabels, grid=True, cols=2):
    """
    Метод построения графиков
    :param data_sets: Кортежи вида (x, y)
    :param titles: Список заголовков для каждого графика
    :param xlabels: Список горизонтальных осей для каждого графика
    :param ylabels: Список вертикальных осей для каждого графика
    :param grid: Включение сетки
    :param cols: Кол-во колонок
    """
    num_graphs = len(data_sets)
    rows = math.ceil(num_graphs / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * cols))
    axes = axes.flatten() if num_graphs > 1 else [axes]

    if num_graphs == 1:
        axes = [axes]

    for i, (x, y) in enumerate(data_sets):
        ax = axes[i]
        ax.plot(x, y, marker="o", linestyle="--")
        ax.set_title(titles[i] if titles and i < len(titles) else f"PLot {i + 1}", fontsize=10)
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabels[i])
        if grid:
            ax.grid(True,  linestyle="--", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.show()