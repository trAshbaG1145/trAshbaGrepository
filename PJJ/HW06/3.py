# Define the function to find the longest word
def maxWord(s):
    # Step 1: Clean the string by removing punctuation marks
    # Replace non-alphabetic characters (except spaces) with a space
    cleaned_chars = []
    for char in s:
        if char.isalpha() or char == " ":
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(" ")
            
    # Reconstruct the cleaned string
    cleaned_string = "".join(cleaned_chars)
    
    # Step 2: Split the cleaned string into a list of words
    words = cleaned_string.split()
    
    # If the list is empty (e.g., input was only punctuation), return an empty string
    if not words:
        return ""
        
    # Step 3: Find the longest word using the built-in max() function with key=len
    longest = max(words, key=len)
    return longest

# Main program execution
# Read the input string from the user
input_string = input()

# Call the function and print the longest word
print(maxWord(input_string))