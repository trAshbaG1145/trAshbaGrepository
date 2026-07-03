# Read the input string from the user
text = input()

# Initialize an empty list to store the converted characters
result_chars = []

# Iterate through each character in the string
for char in text:
    if char.islower():
        # Convert lowercase to uppercase and append
        result_chars.append(char.upper())
    elif char.isupper():
        # Convert uppercase to lowercase and append
        result_chars.append(char.lower())
    else:
        # If it is a non-alphabetic character, leave it unchanged
        result_chars.append(char)

# Join the list back into a single string and output it
print("".join(result_chars))