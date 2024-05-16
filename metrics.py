import os
import csv

import matplotlib.pyplot as plt
import pandas
import seaborn


def main() -> None:
    sequences: list[str] = os.listdir("256x144-Dataset/")
    csvs: list[str] = [("256x144-Dataset/" + sequence + "/data_out.csv") for sequence in sequences]

    dataset: list[pandas.DataFrame] = [pandas.read_csv(csv) for csv in csvs]

    to_plot: pandas.DataFrame = pandas.concat(dataset)
    to_plot = to_plot.iloc[:, 0]
    seaborn.histplot(to_plot)

    plt.ylim(0, 1000)
    plt.savefig("throttle.png")


if __name__ == '__main__':
    main()
