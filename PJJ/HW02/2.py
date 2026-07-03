# Get a four-digit integer representing the year from the user
year = int(input())

# Determine whether it is a leap year based on the given rules:
# 1. Divisible by 400
# OR
# 2. Divisible by 4 AND NOT divisible by 100
if (year % 400 == 0) or (year % 4 == 0 and year % 100 != 0):
    print("Yes")
else:
    print("No")