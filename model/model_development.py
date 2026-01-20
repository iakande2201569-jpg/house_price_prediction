# model_development.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# 1. Load Dataset
# Ensure train.csv is in the same folder or update the path
df = pd.read_csv('train.csv')

# 2. Feature Selection
# We selected these 6 features + the target variable 'SalePrice'
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt', 'Neighborhood']
target = 'SalePrice'

X = df[features]
y = df[target]

# 3. Data Preprocessing Setup
# Define numeric and categorical features
numeric_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']
categorical_features = ['Neighborhood']

# Create transformers for preprocessing
# Numeric: Fill missing values with median, then scale (optional but good practice)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical: Fill missing with 'missing', then OneHotEncode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Define the Model (Random Forest)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train the Model
print("Training the Random Forest model...")
model.fit(X_train, y_train)

# 7. Evaluate the Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"MAE: ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# 8. Save the Model
# We create a 'model' folder if it doesn't exist
import os
if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(model, 'model/house_price_model.pkl')
print("\nModel saved to 'model/house_price_model.pkl'")