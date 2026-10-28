import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

df = pd.read_csv('Titanic-Dataset.csv')
df.columns = df.columns.str.lower()

# Keep the target and simple useful input columns.
df = df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'sex']].copy()
df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())
df = pd.get_dummies(df, columns=['sex'], drop_first=True, dtype=int)

X = df.drop(columns='survived')
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

for name, m in models.items():
    m.fit(X_train, y_train)
    print(name, 'accuracy:', accuracy_score(y_test, m.predict(X_test)))

ConfusionMatrixDisplay.from_estimator(models['KNN'], X_test, y_test)
plt.title('KNN confusion matrix')
plt.show()
