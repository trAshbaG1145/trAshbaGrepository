# Define the function to calculate the factorial of n
def fact(n):
    result = 1
    # Loop from 1 up to n (inclusive) and multiply
    for i in range(1, n + 1):
        result *= i
    return result

# Read the input integer from the user
num = int(input())

# Call the function and print the result
print(fact(num))