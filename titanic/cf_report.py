import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pull_data import join_path


def csv_cf_report(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:4]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    cf_df = pd.DataFrame.from_dict(report_data)
    cf_df.to_csv(join_path("report", "classification_report.csv"), index=False)


def plot_cf_report(cf_report, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):
    lines = cf_report.split('\n')

    classes = []
    plotMat = []
    for line in lines[2:4]:
        t = line.split()
        print()
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')

    print(cf_report)
    plt.savefig(join_path("report", "classification_report.png"))
