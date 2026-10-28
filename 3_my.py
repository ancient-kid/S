import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Titanic-Dataset.csv')
df.columns = df.columns.str.lower()
df = df.select_dtypes(include='number').dropna()
df = df.drop(columns='passengerid', errors='ignore')

n = min(500, len(df))

# 1. Simple random sampling
simple = df.sample(n, random_state=1)

# 2. Stratified sampling using age
df['grp'] = pd.qcut(df['age'], 4, labels=False, duplicates='drop')
strat = df.groupby('grp', group_keys=False).sample(
    frac=n/len(df), random_state=1
)

# 3. Systematic sampling
systematic = df.iloc[[int(i*len(df)/n) for i in range(n)]]

# Compare mean fares
means = {
    'Population': df['fare'].mean(),
    'Simple': simple['fare'].mean(),
    'Stratified': strat['fare'].mean(),
    'Systematic': systematic['fare'].mean()
}

print(means)

plt.bar(means.keys(), means.values())
plt.show()
