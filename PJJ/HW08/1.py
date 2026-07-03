import numpy as np

def analyze_temperatures(filename):
    try:
        # Remove delimiter='\t' to let NumPy automatically handle all whitespaces (spaces and tabs)
        data = np.loadtxt(filename)
        
        # 1. Calculate the mean and standard deviation for each row
        row_means = np.mean(data, axis=1)
        row_stds = np.std(data, axis=1) 
        
        # Output the statistical information for each row
        for i in range(len(data)):
            print(f"Time {i+1}: mean={row_means[i]:.2f} std={row_stds[i]:.2f}")
            
        # 2. Find the overall maximum and minimum values
        overall_max = np.max(data)
        overall_min = np.min(data)
        
        print(f"Overall max: {overall_max:.2f}")
        print(f"Overall min: {overall_min:.2f}")
        
        # 3. Detect outliers (overall mean ± 2 * overall std)
        overall_mean = np.mean(data)
        overall_std = np.std(data)
        
        lower_bound = overall_mean - 2 * overall_std
        upper_bound = overall_mean + 2 * overall_std
        
        # Count the total number of abnormal points
        outliers = (data < lower_bound) | (data > upper_bound)
        abnormal_count = np.sum(outliers)
        
        print(f"Abnormal points: {abnormal_count}")
        
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")

if __name__ == "__main__":
    analyze_temperatures('Temperature.txt')