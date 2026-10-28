import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('Titanic-Dataset.csv')
df.columns = df.columns.str.lower()

df = df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'sex']].copy()
df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())
df = pd.get_dummies(df, columns=['sex'], drop_first=True, dtype=int)

X = df.drop(columns='survived')
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
print('Decision Tree accuracy:', accuracy_score(y_test, dt.predict(X_test)))
print('Random Forest accuracy:', accuracy_score(y_test, rf.predict(X_test)))

plt.figure(figsize=(12, 6))
plot_tree(
    dt,
    feature_names=list(X.columns),
    class_names=['Did not survive', 'Survived'],
    filled=True
)
plt.title('Decision tree')
plt.show()

plt.bar(X.columns, rf.feature_importances_)
plt.title('Random forest feature importance')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
