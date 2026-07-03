# Read the single line of ten ratings and convert them into a list of integers
votes = list(map(int, input().split()))

# Loop through every possible rating score from 1 to 9
for rating in range(1, 10):
    # Count how many times the current rating appears in the votes list
    vote_count = votes.count(rating)
    
    # Print the rating scale followed by its total vote count
    print(rating, vote_count)