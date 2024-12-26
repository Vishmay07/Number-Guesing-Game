import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
data = pd.read_csv('cricket_player_performance_data.csv')

# Preprocess the data
data.fillna(0, inplace=True)

# Encode categorical variables
data = pd.get_dummies(data)

# Split features and target variable
X = data.drop('predicted_runs', axis=1)
y = data['predicted_runs']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'cricket_performance_model.pkl')

# Evaluate the model
predictions = model.predict(X_test)
print(predictions)
