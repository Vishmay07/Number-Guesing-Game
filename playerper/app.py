import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Load or create dataset
def create_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'team_strength': np.random.randint(50, 100, num_samples),
        'player_form': np.random.randint(0, 10, num_samples),
        'pitch_conditions': np.random.randint(0, 3, num_samples),
        'previous_meetings': np.random.randint(0, 300, num_samples),
        'total_runs': np.random.randint(100, 400, num_samples)
    }
    return pd.DataFrame(data)


# Load dataset
data = create_synthetic_data(1000)
features = data[['team_strength', 'player_form', 'pitch_conditions', 'previous_meetings']]
target = data['total_runs']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Hyperparameter tuning for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

# Train and evaluate each model
best_model = None
best_score = float('inf')  # Initialize to a high value for MSE

for model_name, model in models.items():
    print(f"Training model: {model_name}")

    if model_name == 'Random Forest':
        grid_search = GridSearchCV(model, rf_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
        grid_search.fit(X_train, y_train)
        best_model_candidate = grid_search.best_estimator_
    else:
        best_model_candidate = model.fit(X_train, y_train)

    # Evaluate model
    cv_scores = cross_val_score(best_model_candidate, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    avg_mse = -np.mean(cv_scores)
    print(f"{model_name} Average MSE: {avg_mse:.2f}")

    if avg_mse < best_score:
        best_score = avg_mse
        best_model = best_model_candidate

if best_model is None:
    print("No model was trained successfully.")
else:
    print(f"Best model: {best_model}")

# Save the best model
joblib.dump(best_model, 'advanced_cricket_score_model.pkl')
print("Model saved as 'advanced_cricket_score_model.pkl'")


# Function for user input and prediction
def user_input_prediction():
    print("Enter the following details to predict total runs:")

    try:
        team_strength = int(input("Team Strength (50-100): "))
        player_form = int(input("Player Form (0-10): "))
        pitch_conditions = int(input("Pitch Conditions (0, 1, or 2): "))
        previous_meetings = int(input("Runs in Previous Meetings (0-300): "))

        # Input validation
        if not (50 <= team_strength <= 100):
            raise ValueError("Team Strength must be between 50 and 100.")
        if not (0 <= player_form <= 10):
            raise ValueError("Player Form must be between 0 and 10.")
        if not (pitch_conditions in [0, 1, 2]):
            raise ValueError("Pitch Conditions must be 0, 1, or 2.")
        if not (0 <= previous_meetings <= 300):
            raise ValueError("Runs in Previous Meetings must be between 0 and 300.")

        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'team_strength': [team_strength],
            'player_form': [player_form],
            'pitch_conditions': [pitch_conditions],
            'previous_meetings': [previous_meetings]
        })

        # Make prediction
        prediction = best_model.predict(input_data)
        print(f"Predicted Total Runs: {prediction[0]:.2f}")

        # Plotting actual vs predicted
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data['total_runs'], y=best_model.predict(features), color='blue', label='Actual Runs')
        plt.scatter(prediction[0], prediction[0], color='red', s=100, label='User Prediction')
        plt.xlabel('Actual Total Runs')
        plt.ylabel('Predicted Total Runs')
        plt.title('Actual vs Predicted Total Runs')
        plt.axline((0, 0), slope=1, color='green', linestyle='--')  # Diagonal line
        plt.legend()
        plt.grid()
        plt.show()

    except ValueError as e:
        print(f"Input error: {e}")


# Run user input prediction
user_input_prediction()
