# Get three numbers in a single line from the user, separated by commas
a, b, c = eval(input("please input three numbers:"))

# Step 1: Check the triangle inequality theorem
# The sum of any two sides must be greater than the third side
if (a + b > c) and (a + c > b) and (b + c > a):
    
    # Step 2: Check for an equilateral triangle (all three sides are equal)
    if a == b == c:
        print("equilateral triangle")
        
    # Step 3: Check for a right triangle using the Pythagorean theorem (a² + b² = c²)
    # Since we don't know which side is the longest, we must check all three possibilities
    elif (a**2 + b**2 == c**2) or (a**2 + c**2 == b**2) or (b**2 + c**2 == a**2):
        print("right triangle")
        
    # Step 4: If it's a valid triangle but neither equilateral nor right
    else:
        print("ordinary triangle")
        
else:
    # If the sides cannot form a triangle
    print("false")