import pandas as pd
import matplotlib.pyplot as plt

def plot_yield_percentage(file_path):
    # Fix Chinese character glitch (from hints)
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False    

    # 1. Read the farm_yield.csv file
    df = pd.read_csv(file_path)
    
    # 2. Group the yield into three levels
    bins = [-float('inf'), 200, 400, float('inf')]
    labels = ['Low', 'Medium', 'High']
    df['yield_level'] = pd.cut(df['crop_yield'], bins=bins, labels=labels)
    
    # 3. Count the total number of samples for each level
    # We use reindex to make sure the order stays exactly 'Low', 'Medium', 'High'
    counts = df['yield_level'].value_counts().reindex(labels)
    
    # 4. Draw a pie chart
    colors = ['#ff9999', '#66b3ff', '#99ff99'] # Colors mapping to Low, Medium, High
    
    plt.figure(figsize=(8, 6))
    plt.pie(
        counts, 
        labels=counts.index, 
        autopct='%1.1f%%', 
        colors=colors, 
        startangle=120, 
        shadow=True
    )
    
    # Add title and keep the circle perfectly round
    plt.title('Crop Yield Levels Percentage')
    plt.axis('equal')
    
    # Save the figure as requested
    plt.savefig('output.png', dpi=300)
    print("Pie chart successfully saved as 'output.png'.")

# Run the function
plot_yield_percentage('farm_yield.csv')