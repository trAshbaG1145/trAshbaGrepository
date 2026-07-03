# Read the input line and convert into a list of integers
arr = list(map(int, input().split()))

n = len(arr)
# Loop from 0 up to the middle index of the array
for i in range(n // 2):
    # Swap the element at index i with its corresponding element from the back
    # Index (n - 1 - i) targets the elements from the end
    arr[i], arr[n - 1 - i] = arr[n - 1 - i], arr[i]

# Print the elements separated by spaces
print(*arr)