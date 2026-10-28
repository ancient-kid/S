import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv('Titanic-Dataset.csv')
df.columns = df.columns.str.lower()
df = df.select_dtypes(include='number').dropna()
df = df.drop(columns=['passengerid'], errors='ignore')

target = 'fare'
feature = df.drop(columns=target).corrwith(df[target]).abs().idxmax()
print('Feature used:', feature, '| Target:', target)

X = df[[feature]]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print('R2 score:', r2_score(y_test, pred))

order = X_test[feature].to_numpy().argsort()
plt.scatter(X_test[feature], y_test, s=5, alpha=0.4, label='Actual')
plt.plot(X_test[feature].to_numpy()[order], pred[order], color='red', label='Best-fit line')
plt.xlabel(feature)
plt.ylabel(target)
plt.legend()
plt.title('Linear regression')
plt.show()
