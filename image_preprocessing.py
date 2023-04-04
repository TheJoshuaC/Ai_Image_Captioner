# Image Preprocessing

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle



"""
Load the input image: Use a Python imaging library like PIL or OpenCV to load the input image into memory. - DONE

Resize the image: Resize the input image to a fixed size that matches the expected input size of the machine 
learning model. For example, if the model expects 224x224 input images, resize the input image to 224x224. - DONE

Normalize the pixel values: Normalize the pixel values of the image to lie between 0 and 1. 
This is typically done by dividing each pixel value by 255. - DONE

Apply data augmentation: Data augmentation techniques such as rotation, flipping, and cropping can 
be used to create additional training examples and improve the robustness of the model. 
Use a library like Keras ImageDataGenerator or OpenCV to apply data augmentation.

Convert the image to a tensor: Convert the preprocessed image to a tensor that can be fed into 
the machine learning model. A tensor is a multi-dimensional array that can be processed 
efficiently by the machine learning model. - DONE

"""

image_path = '/Users/joshuachristiansen/Github Work/AI_Img_Cap/Ai_Image_Captioner/Images to process /wundt_books1.jpg'
file_path = '/Users/joshuachristiansen/Github Work/AI_Img_Cap/Ai_Image_Captioner/processed_images/wundt_books1.pkl'


def preprocess_image(image_path, file_path):
    # Load the input image
    img = cv2.imread(image_path)

    # Create an empty list to store the augmented images
    augmented_images = []

    # Resize the image to 224x224
    resized = cv2.resize(img, (224, 224))
    augmented_images.append(resized.copy())

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    augmented_images.append(gray.copy())

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    augmented_images.append(blurred.copy())

    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    augmented_images.append(thresh.copy())

    # Dilate the image to fill in gaps
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    augmented_images.append(dilated.copy())

    # Horizontal flip
    img = cv2.imread(image_path)
    flipped = cv2.flip(img, 1)
    augmented_images.append(flipped.copy())

    # Vertical flip
    img = cv2.imread(image_path)
    flipped = cv2.flip(img, 0)
    augmented_images.append(flipped.copy())

    # Rotate the image
    img = cv2.imread(image_path)
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 30, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    augmented_images.append(rotated.copy())

    # Crop the image
    img = cv2.imread(image_path)
    (h, w) = img.shape[:2]
    crop_size = 100
    x = np.random.randint(0, w - crop_size)
    y = np.random.randint(0, h - crop_size)
    cropped = img[y:y+crop_size, x:x+crop_size]
    augmented_images.append(cropped.copy())

    # Add noise to the image
    img = cv2.imread(image_path)
    noise = np.random.randint(-10, 10, img.shape)
    noisy = img + noise
    noisy = np.clip(noisy, 0, 255)
    augmented_images.append(noisy.copy())


    # Add brightness to the image
    img = cv2.imread(image_path)
    M = np.ones(img.shape, dtype="uint8") * 75
    brightened = cv2.add(img, M)
    augmented_images.append(brightened.copy())

    # Add contrast to the image
    img = cv2.imread(image_path)
    alpha = 2.0
    contrast = cv2.multiply(img, np.array([alpha]))
    augmented_images.append(contrast.copy())

    # Add saturation to the image
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = img_hsv[:, :, 1] * 1.5
    img_hsv[:, :, 1][img_hsv[:, :, 1] > 255] = 255
    img_hsv[:, :, 2] = img_hsv[:, :, 2] * 1.2
    augmented_images.append(img_hsv.copy())

    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    augmented_images.append(img_rgb.copy())

    # Add sharpness to the image
    img = cv2.imread(image_path)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    augmented_images.append(sharpened.copy())

    # Add emboss to the image
    img = cv2.imread(image_path)
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    embossed = cv2.filter2D(img, -1, kernel)
    augmented_images.append(embossed.copy())

    # Add edge detection to the image
    img = cv2.imread(image_path)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    edges = cv2.filter2D(img, -1, kernel)
    augmented_images.append(edges.copy())

    # Add motion blur to the image
    img = cv2.imread(image_path)
    size = 15
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    motion_blurred = cv2.filter2D(img, -1, kernel_motion_blur)
    augmented_images.append(motion_blurred.copy())

    # Add zoom blur to the image
    img = cv2.imread(image_path)
    size = 15
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), int((size-1)/2)] = 2.0
    box_filter = 1/(size**2)
    kernel = kernel - box_filter
    zoom_blurred = cv2.filter2D(img, -1, kernel)
    augmented_images.append(zoom_blurred.copy())

    # Create a list to store the tensor images
    tensor_images = []

    # Loop through each augmented image and convert it to a tensor
    for image in augmented_images:
        # Normalize the pixel values
        img_array = image / 255.0
    
        # Convert the image to a tensor
        img_tensor = np.expand_dims(img_array, axis=0)
    
        # Append the tensor image to the list
        tensor_images.append(img_tensor)

    i = 1
    while os.path.isfile(file_path):
        file_path = f'{os.path.splitext(file_path)[0]}_{i}.pkl'
        i += 1

    # Save the tensor images into a file
    with open(file_path, 'wb') as f:
        pickle.dump(tensor_images, f)

preprocess_image(image_path, file_path)

    
    






"------------------------------------------------------------------------------------------------------------"
# Playing with pre-processing

# img = cv2.imread("/Users/joshuachristiansen/Github Work/AI_Img_Cap/Ai_Image_Captioner/Images to process /wundt_books1.jpg", 1)


# h, w, c = img.shape
# print("Dimensions of the image is:nnHeight:", h, "pixelsnWidth:", w, "pixelsnNumber of Channels:", c) # Dimensions of the image is:nnHeight: 225 pixelsnWidth: 225 pixelsnNumber of Channels: 3

# print(type(img)) # <class 'numpy.ndarray'>

# # Since the image is an N-dimensional array, we can check the data type of the image:
# print(img.dtype) # uint8

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
# # cv2.imwrite('wundt_books1.jpg', gray)

# cv2.imshow('wundt_books1_greyscale', img)


# def extract_bit_plane(cd):
#     #  extracting all bit one by one 
#     # from 1st to 8th in variable 
#     # from c1 to c8 respectively 
#     c1 = np.mod(cd, 2)
#     c2 = np.mod(np.floor(cd/2), 2)
#     c3 = np.mod(np.floor(cd/4), 2)
#     c4 = np.mod(np.floor(cd/8), 2)
#     c5 = np.mod(np.floor(cd/16), 2)
#     c6 = np.mod(np.floor(cd/32), 2)
#     c7 = np.mod(np.floor(cd/64), 2)
#     c8 = np.mod(np.floor(cd/128), 2)
#     # combining image again to form equivalent to original grayscale image 
#     cc = 2 * (2 * (2 * c8 + c7) + c6) # reconstructing image  with 3 most significant bit planes
#     to_plot = [cd, c1, c2, c3, c4, c5, c6, c7, c8, cc]
#     fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(10, 8), subplot_kw={'xticks': [], 'yticks': []})
#     fig.subplots_adjust(hspace=0.05, wspace=0.05)
#     for ax, i in zip(axes.flat, to_plot):
#         ax.imshow(i, cmap='gray')
#     plt.tight_layout()
#     plt.show()
#     return cc

# # reconstructed_image = extract_bit_plane(gray) # extracting bit planes from the image

# con_img = np.zeros([256, 256])
# con_img[0:32, :] = 40 # upper row
# con_img[:, :32] = 40 #left column
# con_img[:, 224:256] = 40 # right column
# con_img[224:, :] = 40 # lower row
# con_img[32:64, 32:224] = 80 # upper row
# con_img[64:224, 32:64] = 80 # left column
# con_img[64:224, 192:224] = 80 # right column
# con_img[192:224, 32:224] = 80 # lower row
# con_img[64:96, 64:192] = 160 # upper row
# con_img[96:192, 64:96] = 160 # left column
# con_img[96:192, 160:192] = 160 # right column
# con_img[160:192, 64:192] = 160 # lower row
# con_img[96:160, 96:160] = 220
# plt.imshow(con_img)



# # Viewing the image

# # cv2.imshow("/Users/joshuachristiansen/Github Work/AI_Img_Cap/Ai_Image_Captioner/Images to process /wundt_books1.jpg", img)
# # k = cv2.waitKey(0)
# # if k == 27 or k == ord('q'):
# #     cv2.destroyAllWindows()