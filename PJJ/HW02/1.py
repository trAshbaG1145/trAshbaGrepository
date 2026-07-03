# Get the integer input from the user
# Note: The input prompt includes "number:" to match the input sample.
num = int(input("number:"))

# Check if the number is divisible by 7 or 11
if num % 7 == 0 or num % 11 == 0:
    print("Yes")
else:
    print("No")