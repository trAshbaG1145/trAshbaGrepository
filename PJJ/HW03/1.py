# Get a single character from the user with the required prompt
char = input("please input a char:")

# Check if the character is a letter (a-z, A-Z)
if char.isalpha():
    print("alphabet character")

# Check if the character is a number (0-9)
elif char.isdigit():
    print("digital character")

# If it is neither a letter nor a number
else:
    print("others character")