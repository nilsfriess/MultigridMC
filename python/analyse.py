import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def read_timeseries(filename):
    """read timeseries data from file

    Returns the raw timeseries and the autocorrelation function"""
    df = pd.read_csv(filename, header=None)
    data = df.to_numpy().flatten()

    # Mean
    mean = np.mean(data)

    # Variance
    var = np.var(data)

    # Normalized data
    ndata = data - mean

    acorr = np.correlate(ndata, ndata, "full")[len(ndata) - 1 :]
    acorr = acorr / var / len(ndata)

    return data, acorr


if len(sys.argv) != 2:
    print("Usage: python " + sys.argv[0] + " FILENAME")
    sys.exit(-1)
filename = sys.argv[1]

specs = {
    "MultigridMC": {"extension": "multigridmc", "color": "blue"},
    "SSOR": {"extension": "ssor", "color": "red"},
    "Cholesky": {"extension": "cholesky", "color": "green"},
}

#### Plot timeseries
plt.clf()
for key, val in specs.items():
    data, _ = read_timeseries(filename + "_" + val["extension"] + ".txt")
    plt.plot(data[:100], linewidth=2, color=val["color"], label=key)
ax = plt.gca()
ax.set_xlabel("step")
plt.legend(loc="upper right")
plt.savefig("timeseries.pdf", bbox_inches="tight")

#### Plot autocorrelation
plt.clf()
L = 64
for key, val in specs.items():
    _, acorr = read_timeseries(filename + "_" + val["extension"] + ".txt")
    plt.plot(acorr[:L] / acorr[0], linewidth=2, color=val["color"], label=key)
plt.plot([0, L], [0, 0], linewidth=2, color="black", linestyle="--")
ax = plt.gca()
ax.set_xlabel("lag k")
ax.set_ylabel("C(k)/C(0)")
plt.legend(loc="upper right")
plt.savefig("autocorrelation.pdf", bbox_inches="tight")
