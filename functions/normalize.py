import os
import cv2
import numpy as np
import matplotlib as plt


def normalize_images(X, target_size, apply_sharpening=False, apply_sobel=False, apply_sobel_x=False, apply_sobel_y=False, sobel_k_size=5):
    normalized_images = []

    for img in X:
        if len(img.shape) == 3:
            # Convert to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img
                
        # Apply a filter to suppress noise
        denoised_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # Detect contours
        _, thresh = cv2.threshold(denoised_img, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(max_contour)

            cropped_img = img[y:y+h, x:x+w]

            resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)
        else:
            # Assign a default value to img if contours are not found
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        # Optionally apply kernel sharpening
        if apply_sharpening:
            kernel_sharpening = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            resized_img = cv2.filter2D(resized_img, -1, kernel_sharpening)

        # Optionally apply Sobel edge detection
        if apply_sobel or apply_sobel_x or apply_sobel_y:
            gray_resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray_resized_img, cv2.CV_64F, 0, 1, ksize=sobel_k_size)
            sobel_y = cv2.Sobel(gray_resized_img, cv2.CV_64F, 1, 0, ksize=sobel_k_size)

            if apply_sobel:
                sobel_OR = cv2.bitwise_or(np.uint8(np.abs(sobel_x)), np.uint8(np.abs(sobel_y)))
                resized_img = sobel_OR

            if apply_sobel_x:
                resized_img = np.uint8(np.abs(sobel_x))

            if apply_sobel_y:
                resized_img = np.uint8(np.abs(sobel_y))

        normalized_images.append(resized_img)
            
    return np.array(normalized_images)