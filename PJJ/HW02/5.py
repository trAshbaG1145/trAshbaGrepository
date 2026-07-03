# Read four course scores at once using eval()
a, b, c, d = eval(input())

# Calculate the total score of all four courses
total_score = a + b + c + d

# Step 1: Check admission requirements
# Each individual course must be >= 60 AND the total score must be >= 340
if a >= 60 and b >= 60 and c >= 60 and d >= 60 and total_score >= 340:
    
    # Step 2: Determine funding status for admitted students
    # If the total score is less than 370, they must self-fund
    if total_score < 370:
        print("pay")
    else:
        # If the total score is 370 or higher, it is government-funded
        print("free")
        
else:
    # If any course is below 60 OR the total score is below 340
    print("not")