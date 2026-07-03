# Read the space-separated words from the input
input_string = input()

# Step 1: Split the string into individual words
words_list = input_string.split()

# Step 2: Remove duplicate words by converting the list into a set
unique_words = set(words_list)

# Step 3: Sort the unique words alphanumerically
sorted_words = sorted(unique_words)

# Step 4: Print the sorted words separated by spaces
print(*sorted_words)