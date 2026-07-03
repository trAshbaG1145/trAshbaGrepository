# Function to generate odd numbers and count how many times "3" appears
def count_threes_in_odds(n):
    # Step 1: Generate a list of odd numbers in the range [1, n]
    # range(1, n + 1, 2) starts at 1 and steps by 2 (1, 3, 5, 7, ...)
    odd_numbers = list(range(1, n + 1, 2))
    
    # Step 2: Convert the list of numbers into a single combined string
    # This allows us to easily count every single digit "3" (e.g., "33" becomes two "3"s)
    combined_string = "".join(map(str, odd_numbers))
    
    # Step 3: Count the occurrences of the character '3'
    threes_count = combined_string.count('3')
    
    return threes_count

# Main program execution
# Read the positive integer with the required input prompt
num = int(input("number:"))

# Call the function and print the result
print(count_threes_in_odds(num))