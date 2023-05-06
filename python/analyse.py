import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from autocorrelation import AutoCorrelationWolff


def read_timeseries(filename):
    """read timeseries data from file

    Returns the raw timeseries and the autocorrelation function"""
    df = pd.read_csv(filename, header=None)
    data = df.to_numpy().flatten()

    return data


specs = {
    "MultigridMC": {"extension": "multigridmc", "color": "blue"},
    "SSOR": {"extension": "ssor", "color": "red"},
    "Cholesky": {"extension": "cholesky", "color": "green"},
}

#### Plot timeseries
plt.clf()
gridsizes = (32, 64, 128, 256)
linestyles = {32: "-", 64: "--", 128: "-.", 256: ":"}
n_timeseries = 64
max_lag = 16
fig_timeseries, axs_timeseries = plt.subplots(2, 2)
fig_acorr, ax_acorr = plt.subplots()
j = 0
for nx in gridsizes:
    print("nx = ", nx)
    ax_timeseries = axs_timeseries[j // 2, j % 2]
    for key, val in specs.items():
        data = read_timeseries(f"timeseries{nx:d}x{nx:d}_" + val["extension"] + ".txt")
        axs_timeseries[j // 2, j % 2].plot(
            data[:n_timeseries],
            linewidth=2,
            color=val["color"],
            label=key if nx == 32 else None,
        )
        acw = AutoCorrelationWolff(data)
        acorr = acw._c()
        if key == "MultigridMC":
            tau_int, dtau_int = acw.tau_int()
            print(f"tau_int = {tau_int:5.2f} +/- {dtau_int:5.2f}")
        ax_acorr.plot(
            acorr[:max_lag],
            linewidth=2,
            linestyle=linestyles[nx],
            color=val["color"],
            label=key if nx == 32 else None,
        )
    ax_timeseries.plot(
        [0, n_timeseries], [0, 0], linewidth=2, color="black", linestyle="--"
    )
    ax_timeseries.set_title(rf"${nx:d}\times{nx:d}$", y=0.85)
    ax_timeseries.set_ylim(-3, 8)
    if nx == 128:
        ax_timeseries.set_xlabel("step")
    if nx == 32:
        ax_timeseries.legend(loc="lower right", fontsize=8)
    j += 1
    ax_acorr.plot(
        [],
        [],
        linewidth=2,
        color="black",
        linestyle=linestyles[nx],
        label=rf"${nx:d}\times{nx:d}$",
    )
fig_timeseries.savefig("timeseries.pdf", bbox_inches="tight")

ax_acorr.plot([0, max_lag], [0, 0], linewidth=2, color="black", linestyle="--")
ax_acorr.set_xlabel("lag k")
ax_acorr.set_ylabel("C(k)/C(0)")
ax_acorr.legend(loc="upper right")
fig_acorr.savefig("autocorrelation.pdf", bbox_inches="tight")
