import seaborn as sns
import matplotlib.pyplot as plt


def plotData(data):
    sns.relplot(
        data=data,
        x="total_bill", y="tip", col="time",
        hue="smoker", style="smoker", size="size",
    )


plotData(sns.load_dataset("tips"))
plt.show()
