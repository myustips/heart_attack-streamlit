# %%
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# %%
df = pd.read_csv('C:/Users/ASUS/Documents/AE Project/heart.csv')
print(df)

# %%
df.isnull().sum()


# %%
df['output'].value_counts()

# %%
X = df.drop('output', axis=1)
y = df['output']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# %%
model = LogisticRegression(solver='liblinear').fit(X_train, y_train)
res = model.predict(X_test)
print(res)

# %%
model.score(X_test, y_test)

# %%
model.predict_proba(X_test)

# %%



