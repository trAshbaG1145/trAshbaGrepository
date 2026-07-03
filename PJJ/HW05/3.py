# Read the number of integers (n) from the first line
n = int(input())

# Read the second line, split it by spaces, and convert each string into an integer
numbers = list(map(int, input().split()))

# Sort the list of numbers in ascending order
numbers.sort()

# Print the sorted numbers separated by spaces
print(*numbers)