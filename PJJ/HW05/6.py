# Read the ciphertext from the user
cipher_text = input()

# Initialize an empty list to collect the decoded characters
original_chars = []

# Loop through each character in the ciphertext
for char in cipher_text:
    if 'A' <= char <= 'Z':
        # Decode uppercase letter
        decoded_ascii = ord('Z') - (ord(char) - ord('A'))
        original_chars.append(chr(decoded_ascii))
        
    elif 'a' <= char <= 'z':
        # Decode lowercase letter
        decoded_ascii = ord('z') - (ord(char) - ord('a'))
        original_chars.append(chr(decoded_ascii))
        
    else:
        # Non-letter characters remain completely unchanged
        original_chars.append(char)

# Combine the list of decoded characters back into a string
original_text = "".join(original_chars)

# Output the ciphertext first, then the decoded original text
print(cipher_text)
print(original_text)