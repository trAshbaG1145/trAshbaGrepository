def total(xs: list) -> int:
    sum_total = 0
    
    # Iterate through each element in the current list
    for element in xs:
        # If the element is a sub-list, recursively calculate its sum
        if isinstance(element, list):
            sum_total += total(element)
        else:
            # If it's a regular number, add it directly to the total
            sum_total += element
            
    return sum_total

# Main program execution
# Prompts the user for a nested list and evaluates the string input into a list object
ls = eval(input("please input a list:"))

# Print the final calculated sum
print(total(ls))