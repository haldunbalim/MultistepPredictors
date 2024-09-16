import matplotlib.pyplot as plt

def plot_err_two_sided(means, tights, names, colors, lw = 2, alpha=0.6):
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 20

    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    ax = fig.add_subplot(111)

    custom_lines = []
    for mean, tight, color in zip(means, tights, colors):
        ax.fill_between(range(len(tight)), mean+tight, mean-tight,
                        facecolor=color, alpha=alpha)
        custom_lines.append(plt.Line2D([0], [0], color=color, lw=lw))

    ax.legend(custom_lines, names, loc="lower right", fontsize=14)
    ax.grid()
    ax.set_ylabel("Probabilistic reachable set")
    ax.set_xlabel("Horizon")
    ax.set_ylim([-.1, 2.6])
    ax.set_xticks([0, 5, 10, 15, 20])
    return fig
