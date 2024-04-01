import cv2
import numpy as np
import streamlit as st
import os
from skimage.metrics import structural_similarity as ssim


def process_image(image, reference):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to a fixed size
    size = (600, 400)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    # Compute the structural similarity index
    ssim_index = ssim(resized, reference)

    # Apply thresholding to detect tampering
    threshold = 0.8
    tampering_detected = ssim_index < threshold

    # Find contours in the image
    contours, hierarchy = cv2.findContours(resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return tampering_detected, ssim_index, contours


def draw_contours(image, contours):
    # Draw the contours on the image
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 100:
            cv2.drawContours(image, contours, i, (0, 0, 255), 2)



   # Define the Streamlit app
def main():
    st.set_page_config(page_title='PAN Card Tampering Detection')

    # Add a title and subtitle
    st.title('PAN Card Tampering Detection')
    st.write('Upload an image of a PAN card to detect tampering and validate the ID.')

    # Load reference images from a folder
    reference_folder = "C:\\Users\\DELL 3400\\OneDrive\\Desktop\\Pan-Card-Tampering-Detection\\reference"
    reference_images = []
    for filename in os.listdir(reference_folder):
        reference_path = os.path.join(reference_folder, filename)
        reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
        if reference_image is not None:
            reference_image_resized = cv2.resize(reference_image, (600, 400), interpolation=cv2.INTER_AREA)
            reference_images.append(reference_image_resized)


    # Load tampered images
    tampered_folder = "C:\\Users\\DELL 3400\\OneDrive\\Desktop\\Pan-Card-Tampering-Detection\\tampered"
    tampered_images = []
    for filename in os.listdir(tampered_folder):
        tampered_path = os.path.join(tampered_folder, filename)
        tampered_image = cv2.imread(tampered_path, cv2.IMREAD_GRAYSCALE)
        if tampered_image is not None:
            tampered_image_resized = cv2.resize(tampered_image, (600, 400), interpolation=cv2.INTER_AREA)
            tampered_images.append(tampered_image_resized)


    # Add a file uploader to get the image from the user
    uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'jpeg', 'png'])

    # Process the uploaded image and display the results
    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # Initialize variables
        tampering_detected = False
        best_ssim_index = 0

        # Iterate through each reference image
        for reference_image_resized in reference_images:
            # Process the image
            curr_tampering_detected, ssim_index, contours = process_image(image, reference_image_resized)

            # Check if current similarity index is better than previous best
            if ssim_index > best_ssim_index:
                tampering_detected = curr_tampering_detected
                best_ssim_index = ssim_index

            # Break loop if perfect match is found
            if ssim_index == 1:
                tampering_detected = False
                break

        # Draw the contours on the image
        draw_contours(image, contours)

        # Display the image and results
        st.image(image, channels='BGR', use_column_width=True)
        if tampering_detected:
            st.error('Tampering detected!')
        else:
            st.success('No tampering detected.')
        st.write(f'Best structural similarity index: {best_ssim_index:.2f}')

if __name__ == '__main__':
    main()
