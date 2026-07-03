# Load Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

def main():
    # 1. Load the dataset
    tea_df = pd.read_csv('TeaData.csv')
    
    # 2. Separate the independent variables (X) and the dependent variable (y).
    # Assuming Column 1 (index 0) is ID, and the last column is tea_yield.
    X = tea_df.iloc[:, 1:-1] # select from the 2nd column to the second-to-last column
    y = tea_df.iloc[:, -1]   # select the last column (tea yield)
    
    # 3. Splitting training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 4. Standardize the dataset
    scaler = StandardScaler().fit(X_train) # generate the standardization rule
    X_train = scaler.transform(X_train)    # standardize the training set
    X_test = scaler.transform(X_test)      # apply the training-set standardization rule to the test set
    
    # 5. Build a Multi-Layer Perceptron model: train - test - evaluate
    # Training model
    model = MLPRegressor(
        hidden_layer_sizes=(100, 100), # two hidden layers, each with 100 neurons
        activation="relu",             # activation function is ReLU
        shuffle=True,                  # whether to shuffle samples after each iteration
        solver="adam",                 # the optimizer
        alpha=0.001,                   # L2 regularization coefficient
        learning_rate="adaptive",      # learning strategy
        verbose=False,                 # whether to print training progress to standard output
        max_iter=500,                  # maximum number of iterations
        random_state=42                # make result reproducible
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # 6. Make predictions and evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    # Output Requirements: Print the model evaluation metric R2, rounded to four decimal places
    print(f"Model Evaluation Metric R2: {r2:.4f}")

if __name__ == "__main__":
    main()