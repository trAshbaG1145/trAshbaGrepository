# Get the 18-digit ID card number as a string
id_num = input().strip()

# Step 1: Extract the birth date components using string slicing
# The 7th to 14th digits correspond to indices 6 to 14 (not inclusive)
year = id_num[6:10]    # 7th, 8th, 9th, 10th digits
month = id_num[10:12]  # 11th, 12th digits
day = id_num[12:14]    # 13th, 14th digits

# Print the date in YYYY-MM-DD format
print(f"{year}-{month}-{day}")

# Step 2: Extract the gender component
# The 17th digit corresponds to index 16
gender_digit = int(id_num[16])

# Check if the digit is odd (Male) or even (Female)
if gender_digit % 2 != 0:
    print("M")
else:
    print("F")