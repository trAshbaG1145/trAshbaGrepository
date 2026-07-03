# 1. Load Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

def main():
    # Load the dataset
    plant_df = pd.read_csv('PlantData.csv')
    
    # 2. Separate the features (X) and label (y)
    X = plant_df.iloc[:, :-1]  # Select all columns except the last one
    y = plant_df.iloc[:, -1]   # Select only the last column (status)
    
    # Note: The labels are already numerical (0, 1, 2), so mapping is not strictly necessary.
    
    # 3. Splitting dataset into training and test sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)  
    
    # 4. Standardize Datasets
    scaler = StandardScaler().fit(X_train) # generate the standardization rule
    X_train = scaler.transform(X_train)    # standardize the training set
    X_test = scaler.transform(X_test)      # apply the training-set standardization rule to the test set
    
    # 5. Training Model
    model = SVC(
        kernel='linear',    # set kernel function
        C=1,                # set regularization hyperparameter
        random_state=42     # set fixed number to make running result reproducible
    )
    model.fit(X_train, y_train)
    
    # 6. Make predictions and evaluate
    y_pred = model.predict(X_test)
    
    # Calculate F1 score
    # Since there are 3 classes (0, 1, 2), we use average='weighted' to account for label imbalance
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Output Requirements: Print the model evaluation metric F1_score, rounded to four decimal places.
    print(f"Model Evaluation Metric F1_score: {f1:.4f}")

if __name__ == "__main__":
    main()