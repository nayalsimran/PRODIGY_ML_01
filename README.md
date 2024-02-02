# PRODIGY_ML_01
TASK 1-Implement a linear regression model to predict the prices of houses based on their square footage and no of bed rooms and bathrooms

### **Importing Libraries**

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

### **Reading the Dataset**

df = pd.DataFrame(pd.read_csv("/content/House Price.csv"))

### **Data Exploration**

df.head()

df.info()

df.describe()

df.isnull().sum()

varlist =['bedroom','bathroom','floors','sqft_living','street','city','view','waterfront']
def binary_map(x):
  return x.map({'yes': 1,"no":0})
  df[varlist]= df[varlist].apply(binary_map)

df.head()

status=pd.get_dummies(df['waterfront'])

status.head()

df.drop(['waterfront'],axis=1,inplace=True)

df.head()

np.random.seed(0)
df_train, df_test = train_test_split (df, train_size=0.7, test_size=0.3, random_state=100)


from sklearn.preprocessing import StandardScaler

scaler =StandardScaler

scaler =StandardScaler()
num_vars = ['sqft_living','bedrooms','bathrooms','price']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

df_train.head()

plt.figure(figsize=(16,10))
sns.heatmap(df.corr(),annot=True,cmap="YlGnBu")
plt.show()

### **Implementation**

X=df[['bedrooms','bathrooms','sqft_lot']]
y=df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X,y)

### **Evaluation**

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_test)
r2= r2_score(y_test, y_pred)
print("Mean Squared Error:",mse)
print("R -squared:",r2)


### **Prediction**

new_data = [[4, 2 ,5000]]
predicted_price = model.predict(new_data)

print("Predicted Price:", predicted_price[0])


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()

residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Residuals")
plt.title("Residual plot")
plt.show()
