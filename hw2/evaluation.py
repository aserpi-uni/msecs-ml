import math
from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt
from matplotlib import ticker
import pandas as pd
import seaborn as sns


def plot_metrics(net: str, history: Path, loss_ylim: Optional[float] = None) -> None:
    metrics = pd.read_csv(history, index_col="epoch")
    metrics.index += 1

    metrics.rename(columns={"acc": "Accuracy",
                            "loss": "Loss",
                            "val_acc": "Validation accuracy",
                            "val_loss": "Validation loss"},
                   inplace=True)  # yapf: disable
    metrics.index.name = "Epoch"
    x_formatter = ticker.MultipleLocator(math.ceil(len(metrics) / 20))

    plt.figure()
    ax = sns.lineplot(data=metrics[["Accuracy", "Validation accuracy"]],
                      dashes=False,
                      markers=True)
    ax.set(xlabel="Epoch", ylabel='Accuracy', ylim=(0, 1))
    ax.xaxis.set_major_locator(x_formatter)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.get_figure().savefig(history.parent / f"{net}_accuracy.pdf", format="pdf")
    plt.clf()

    plt.figure()
    ax = sns.lineplot(data=metrics[["Loss", "Validation loss"]], dashes=False, markers=True)
    ax.set(xlabel="Epoch", ylabel="Loss")
    if loss_ylim:
        ax.set(ylim=(0, loss_ylim))
    ax.xaxis.set_major_locator(x_formatter)
    ax.get_figure().savefig(history.parent / f"{net}_loss.pdf", format="pdf")
    plt.clf()


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = []
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]
