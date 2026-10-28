import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Titanic-Dataset.csv")
df = df.drop(columns=("PassengerId"))

print(df.head())
print(df.describe())

df.hist(figsize=(8,6))
plt.suptitle("Histogram of numerical values in Titanic dataset")
plt.tight_layout()
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap = 'coolwarm')
plt.title("Correlation Heatmap")
plt.show()
