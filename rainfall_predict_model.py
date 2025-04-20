# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Step 2: Create Synthetic Dataset
np.random.seed(42)
n_samples = 1000

# Generate features
temperature = np.random.uniform(0, 40, n_samples)  # Celsius
humidity = np.random.uniform(0, 100, n_samples)    # Percentage
wind_speed = np.random.uniform(0, 50, n_samples)   # km/h

# Generate target variable (Rainfall: 0 = No, 1 = Yes)
# Let's assume rainfall probability depends on combinations of features
rain_prob = (humidity * 0.3 + (40 - temperature) * 0.2 + wind_speed * 0.1) / 100
rainfall = np.where(rain_prob > np.random.rand(n_samples), 1, 0)

# Create DataFrame
weather_df = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Wind_Speed': wind_speed,
    'Rainfall': rainfall
})

# Step 3: Exploratory Data Analysis (EDA)
# Display first 5 rows
print(weather_df.head())

# Basic statistics
print(weather_df.describe())

# Check class distribution
print("\nClass Distribution:")
print(weather_df['Rainfall'].value_counts())

# Visualizations
plt.figure(figsize=(12, 8))

# Pairplot
sns.pairplot(weather_df, hue='Rainfall')
plt.suptitle('Pairwise Relationships by Rainfall Status', y=1.02)
plt.show()

# Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(weather_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 4: Data Preprocessing
# Handle missing values (though our synthetic data shouldn't have any)
imputer = SimpleImputer(strategy='mean')
X = weather_df.drop('Rainfall', axis=1)
y = weather_df['Rainfall']

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Model Training
# Initialize models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(random_state=42)

# Train Random Forest
rf.fit(X_train_scaled, y_train)

# Train Logistic Regression
lr.fit(X_train_scaled, y_train)

# Step 6: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    sns.heatmap(confusion_matrix(y_test, y_pred), 
                annot=True, fmt='d', cmap='Blues')
    plt.show()

# Evaluate Random Forest
print("Random Forest Performance:")
evaluate_model(rf, X_test_scaled, y_test)

# Evaluate Logistic Regression
print("\nLogistic Regression Performance:")
evaluate_model(lr, X_test_scaled, y_test)

# Step 7: Feature Importance Analysis (for Random Forest)
features = X.columns
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices])
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()

# Step 8: Make Predictions (Example)
new_data = pd.DataFrame({
    'Temperature': [25, 10, 35],
    'Humidity': [80, 60, 45],
    'Wind_Speed': [15, 25, 5]
})

new_data_scaled = scaler.transform(new_data)
rf_predictions = rf.predict(new_data_scaled)
lr_predictions = lr.predict(new_data_scaled)

print("Random Forest Predictions:", rf_predictions)
print("Logistic Regression Predictions:", lr_predictions)