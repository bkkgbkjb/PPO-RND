import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from os import path


def plot():
    rlts = []
    # rlt = {}
    for r, d, files in os.walk("./Logs"):
        for file in files:
            assert ".log" in file
            i = 0
            with open(path.join(r, file), "r") as f:
                for rows in f:
                    if "eval/returns: [" not in rows:
                        continue
                    _lid = rows.index("[")
                    _rid = rows.index("]")
                    _c = list(map(float, rows[_lid + 1 : _rid].split(",")))
                    _c = np.array(_c, dtype=np.float32)
                    rlts.append(
                        {
                            "returns": _c,
                            "return_mean": np.mean(_c),
                            "return_std": np.std(_c),
                            "step": i,
                            "seed": file,
                        }
                    )
                    i += 1

    # print(rlts)
    # print(len(rlts))

    df = pd.DataFrame(rlts)
    print(df)

    sns.set_theme()

    sns.lineplot(
        data=df[["step", "return_mean", "seed"]],
        x="step",
        y="return_mean",
    )
    plt.show()
    plt.close()


plot()
