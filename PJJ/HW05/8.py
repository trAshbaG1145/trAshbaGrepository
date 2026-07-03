# Helper function to check if a number is prime
def is_prime(num):
    if num < 2:
        return False
    # Check for factors up to the square root of num
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

# Read the even integer n from the user
n = int(input())

# Iterate through every even number from 4 to n (inclusive)
for x in range(4, n + 1, 2):
    # Find two primes that add up to x
    # We loop p1 starting from 2. To ensure p1 <= p2, p1 goes up to x // 2
    for p1 in range(2, (x // 2) + 1):
        p2 = x - p1
        
        # If both p1 and p2 are prime numbers, we found our pair
        if is_prime(p1) and is_prime(p2):
            print(f"{x}={p1}+{p2}")
            break  # Break the inner loop once the first valid pair is found