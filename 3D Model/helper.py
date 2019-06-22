import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def color_map_conf_matr(conf_matr):
    df_cm = pd.DataFrame(conf_matr, index=["C. Normal", "AD"], columns=["C. Normal", "AD"])
    plt.figure(figsize=(2, 2))
    sn.heatmap(df_cm, annot=True)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()


def plot_graphs(loss, num_epochs, type):
    # plot loss graph
    plt.plot(loss["train"], color='green')
    plt.plot(loss["val"], color='red')
    plt.xticks(np.arange(0, num_epochs, step=5, dtype=np.int))
    plt.legend(['train ' + type, 'validation ' + type], loc='upper left')
    plt.xlabel("Epoch Number")
    plt.ylabel(type)
    plt.title("Tain & Validation " + type + " Change in Each Epoch")
    plt.show()


def calc_metric(y, ystar):
    print(classification_report(y, ystar))
