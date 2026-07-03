# Read the entire line of characters from the user
text = input()

# Initialize counters for uppercase letters, lowercase letters, and digits
upper_count = 0
lower_count = 0
digit_count = 0

# Loop through each character in the string
for char in text:
    if char.isupper():       # Check if the character is an uppercase letter
        upper_count += 1
    elif char.islower():     # Check if the character is a lowercase letter
        lower_count += 1
    elif char.isdigit():     # Check if the character is a numeric digit (0-9)
        digit_count += 1

# Output the results in the requested order, each on a new line
print(upper_count)
print(lower_count)
print(digit_count)