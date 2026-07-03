# Read the main string and the substring from the user
s = input()
c = input()

# Step 1: Count the occurrences of substring c in string s
count_occurrences = s.count(c)

# Step 2: Delete all instances of c by replacing them with an empty string
# This naturally satisfies the requirement of not considering new 
# occurrences formed after deletion, as .replace() processes the original string.
new_string = s.replace(c, "")

# Print the results
print(count_occurrences)
print(new_string)