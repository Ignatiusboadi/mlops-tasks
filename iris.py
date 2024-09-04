from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

iris = load_iris()
print(iris.keys())
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

print(f'Iris shape: {iris_df.shape}')
print(f'Iris feature names: {iris.feature_names}')
print(f"Iris classes: {iris.target_names}")
print(f'Iris first 5 rows:\n {iris_df.head()}')
print(f'Iris summaries:\n{iris_df[iris.feature_names].describe()}')
sns.heatmap(iris_df.corr())
plt.show()

# logistic regression implementation
X = iris_df.drop(columns=['target'])
y = iris_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Model accuracy: {accuracy}')
print(f"Classification report\n: {classification_report(y_test, y_pred)}")


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()
