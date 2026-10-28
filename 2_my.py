import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Titanic-Dataset.csv")
df.columns = df.columns.str.lower()

print("Missing values : ", df.isnull().sum())

df['fare'] = df['fare'].fillna(df['fare'].median())
df['age'] = df['age'].fillna(df['age'].median())
df = pd.get_dummies(df, columns=['sex'], drop_first=True)
df[['age', 'fare']] = StandardScaler().fit_transform(df[['age', 'fare']])

print("Missing values after cleaning: ", df.isnull().sum().sum())

sns.boxplot(x=df['fare'])
plt.title("Fare (outliers shown in dots)")
plt.show()