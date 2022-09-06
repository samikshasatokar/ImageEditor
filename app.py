import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os


@st.cache
def load_image(img):
    im = Image.open(img)
    return im


def main():
    """ Image Editor """

    st.title(" Image Editing App")

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)


    enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Gray-Scale",
                                                     "Contrast", "Brightness", "Blur"])

    if enhance_type == 'Gray-Scale':
        new_img = np.array(our_image.convert('RGB'))
        img = cv2.cvtColor(new_img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        st.image(img)

    if enhance_type == 'Contrast':
        c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
        enhancer = ImageEnhance.Contrast(our_image)
        img_output = enhancer.enhance(c_rate)
        st.image(img_output)

    if enhance_type == 'Brightness':
        c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
        enhancer = ImageEnhance.Brightness(our_image)
        img_output = enhancer.enhance(c_rate)
        st.image(img_output)

    if enhance_type == 'Blur':
        new_img = np.array(our_image.convert('RGB'))
        blur_rate = st.sidebar.slider("Blur Rate", 0.5, 3.5)
        img = cv2.cvtColor(new_img, 1)
        img = cv2.GaussianBlur(img, (5, 5), blur_rate)
        st.image(img)

    filter = st.sidebar.radio("Filter", ["Original", "Emboss",
                                         "Sepia", "Sharpen"])

    if filter == 'Emboss':
        img = np.array(our_image.convert('RGB'))
        emboss_kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
        emboss_effect_img = cv2.filter2D(src=img, kernel=emboss_kernel, ddepth=-1)
        st.text("Embossed Image")
        st.image(emboss_effect_img)

    if filter == 'Sepia':
        img = np.array(our_image.convert('RGB'))
        sepia_kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
        sepia_effect_img = cv2.filter2D(src=img, kernel=sepia_kernel, ddepth=-1)
        st.text("Sepia Effect Image")
        st.image(sepia_effect_img)

    if filter == 'Sharpen':
        img = np.array(our_image.convert('RGB'))
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen_effect_img = cv2.filter2D(src=img, kernel=sharpen_kernel, ddepth=-1)
        st.text("Sharpened Image")
        st.image(sharpen_effect_img)

    edits = st.sidebar.radio("Edit Image", ["Original", "Resize"])

    if edits == 'Resize':
        img = np.array(our_image.convert('RGB'))
        st.text("Enter the scaling percentage")
        scale_percent = int(st.text_input('Scaling Percentage', '0'))
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        st.text("Resized Image")
        st.image(resized_img)









if __name__ == '__main__':
    main()
