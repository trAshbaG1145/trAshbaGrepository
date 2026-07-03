# Read the comma-separated numbers from the user
# Note: The input sample shows a prompt "numbers:", so we include it here.
input_data = input("numbers:")

# Convert the string of numbers into a list of integers
numbers = list(map(int, input_data.split(',')))

# Use a set to store the pairs, which automatically keeps only unique values
unique_pairs = set()

# Nested loop to check every possible pair of numbers
n = len(numbers)
for i in range(n):
    for j in range(i + 1, n):
        # Check if the two numbers add up to 9
        if numbers[i] + numbers[j] == 9:
            # Create a tuple with the smaller number first
            smaller = min(numbers[i], numbers[j])
            larger = max(numbers[i], numbers[j])
            pair = (smaller, larger)
            
            # Add the tuple to our set (duplicates are automatically ignored)
            unique_pairs.add(pair)

# Convert the set of unique pairs back into a list and sort it
# By default, sorting a list of tuples sorts them by their first element
sorted_pairs = sorted(list(unique_pairs))

# Print the final list of sorted pairs
print(sorted_pairs)