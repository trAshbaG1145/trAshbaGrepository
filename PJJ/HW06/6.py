# Read all integers from a single line of input and convert them into a list
numbers = list(map(int, input().split()))

# Step 1: Count the occurrences of each integer using a dictionary
frequency_map = {}
for num in numbers:
    frequency_map[num] = frequency_map.get(num, 0) + 1

# Step 2: Extract the unique items (number, count) from the dictionary
unique_items = list(frequency_map.items())

# Step 3: Sort items based on custom multiple rules using a lambda function
# Rule 1: -x[1] -> Sorts by frequency in descending order (higher frequency first)
# Rule 2:  x[0] -> If frequencies are equal, sorts by the number itself in ascending order (smaller first)
sorted_items = sorted(unique_items, key=lambda x: (-x[1], x[0]))

# Step 4: Output each number and its occurrence count on a separate line
for num, count in sorted_items:
    print(num, count)