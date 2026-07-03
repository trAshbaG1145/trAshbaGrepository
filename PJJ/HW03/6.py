# Read the initial height (n) and the number of landings (m)
n = int(input())
m = int(input())

# The first drop travels exactly 'n' meters downward
total_distance = float(n)
current_height = float(n)

# Loop for the subsequent bounces (from the 1st rebound up to the m-th landing)
for i in range(1, m):
    # Calculate the rebound height for this bounce
    current_height = current_height / 4.0
    # The ball goes up and down, so it travels twice the rebound height
    total_distance += current_height * 2

# After the m-th landing, calculate the final rebound height
final_rebound_height = current_height / 4.0

# Output results formatted to two decimal places
print(f"{total_distance:.2f}")
print(f"{final_rebound_height:.2f}")