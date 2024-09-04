from sklearn.datasets import load_iris
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
