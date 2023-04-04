import pickle

# Open the pickle file for reading
with open('/Users/joshuachristiansen/Github Work/AI_Img_Cap/Ai_Image_Captioner/processed_images/wundt_books1.pkl', 'rb') as f:
    # Load the data from the file
    data = pickle.load(f)

# Print the data
print(data)