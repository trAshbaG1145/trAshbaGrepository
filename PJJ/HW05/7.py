# Read the input string from the user
text = input()

# Step 1: Count frequencies of each character using a dictionary
frequency_map = {}
for char in text:
    frequency_map[char] = frequency_map.get(char, 0) + 1

# Step 2: Find the maximum frequency value present in the map
# If the input string is empty, max_frequency will default to 0
max_frequency = max(frequency_map.values()) if frequency_map else 0

# Step 3: Collect all characters that match the maximum frequency
most_frequent_chars = []
for char, count in frequency_map.items():
    if count == max_frequency:
        most_frequent_chars.append(char)

# Step 4: Sort the tied characters in ascending ASCII order
most_frequent_chars.sort()

# Step 5: Output each character along with its count on a new line
for char in most_frequent_chars:
    print(f"{char} {max_frequency}")