# Function to check if a list contains duplicate elements
def has_duplicates(lst):
    # Convert the list into a set. 
    # A set cannot have duplicate values, so any duplicates will be removed.
    unique_elements = set(lst)
    
    # If the length of the set is less than the length of the original list,
    # it means duplicate items were present and removed.
    if len(unique_elements) < len(lst):
        return True
    else:
        return False

# Main program execution
# Use eval(input()) to safely parse the keyboard input list, e.g., "[1, 2, 5, 2, 3]"
a = eval(input())

# Call the function and print the return value (True or False)
print(has_duplicates(a))