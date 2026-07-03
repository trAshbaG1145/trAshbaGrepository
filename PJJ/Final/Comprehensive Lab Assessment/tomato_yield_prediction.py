import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# ==========================================
# MACHINE LEARNING WORKFLOW
# ==========================================

# 1. Load the dataset
# Ensure 'tomato_yield.csv' is in the same directory as this script
tomato_df = pd.read_csv('tomato_yield.csv')
print("--- Dataset Descriptive Statistics ---")
print(tomato_df.describe()) 

# 2. Separate the independent variables (features, X) and the dependent variable (target, y)
# Select all rows, and columns from index 1 to the second-to-last (excluding Sample_ID and Target)
X = tomato_df.iloc[:, 1:-1] 
# Select all rows, and only the last column (Yield_kg_sqm)
y = tomato_df.iloc[:, -1]   

# 3. Split the dataset into training and test sets
# 70% of data for training, 30% for testing. Random state ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Standardize the dataset
# Neural networks require standardized data for efficient gradient descent
scaler = StandardScaler().fit(X_train) # Generate the standardization rule based on training set
X_train = scaler.transform(X_train)    # Apply rule to standardize the training set
X_test = scaler.transform(X_test)      # Apply the exact same rule to the test set

# 5. Build and train the Multi-Layer Perceptron (MLP) regression model
model = MLPRegressor(
    hidden_layer_sizes=(128, 64), # Two hidden layers with 128 and 64 neurons respectively
    activation="relu",            # ReLU (Rectified Linear Unit) activation function
    shuffle=True,                 # Shuffle samples after each iteration
    solver="adam",                # Adam optimizer for weight updates
    alpha=0.01,                   # L2 regularization penalty to prevent overfitting
    learning_rate="adaptive",     # Adaptive learning rate schedule
    verbose=False,                # Set to True to print progress messages to stdout
    max_iter=800,                 # Maximum number of iterations (epochs)
    random_state=42               # Fixed random seed for reproducible results
)

print("\nTraining the MLP model...")
model.fit(X_train, y_train)

# 6. Test the model and compute evaluation metrics
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Model Evaluation completed.\nModel R² Score: {r2:.4f}")

# ==========================================
# VISUALIZATION
# ==========================================

# Plot 1: Training Loss Curve
# Shows how the error decreases over epochs during training
plt.figure(figsize=(6, 4))
plt.plot(model.loss_curve_, color='green', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Squared Loss')
plt.title('Training Loss Curve (MLP Convergence)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Plot 2: True vs Predicted scatter plot (Test Set)
# Compares the model's predictions against the actual ground truth values
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='white', color='tomato', s=60)
# Draw the ideal diagonal line where predicted equals true
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black', linestyle='--', linewidth=2)
plt.xlabel('True Yield (kg/sqm)')
plt.ylabel('Predicted Yield (kg/sqm)')
plt.title('True vs Predicted Tomato Yield (Test Set)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot 3: Residual Analysis
# Shows the distribution of prediction errors (residuals) across predicted values
residuals = y_test - y_pred
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.7, edgecolors='white', color='purple', s=60)
# Draw a horizontal line at 0 (representing zero error)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Yield (kg/sqm)')
plt.ylabel('Residuals (True - Predicted)')
plt.title('Residual Plot')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()