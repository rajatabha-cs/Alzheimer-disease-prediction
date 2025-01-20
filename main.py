import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image, UnidentifiedImageError
import os


st.set_page_config(
    page_title="ALZHEIMER'S DISEASE RECOGNITION SYSTEM",
    page_icon="logo.png",
    layout="wide"
)
def predict_single_image(model_path, image_path, class_mapping, target_size=(150, 150)):
    """
    Predict the class of a single MRI image.
    
    Args:
        model_path (str): Path to the trained model file.
        image_path (str): Path to the uploaded image file.
        class_mapping (dict): Mapping of class indices to class names.
        target_size (tuple): Target size for image resizing.
    
    Returns:
        str: Predicted class label.
    """
    # Load the trained model
    model = load_model(model_path)

    # Load and preprocess the image
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict the class
    prediction = model.predict(image_array)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]  # Get index of max probability
    predicted_class_label = class_mapping[predicted_class_idx]

    return predicted_class_label

# Add this function to validate if the uploaded image is likely an MRI image
def validate_mri_image(image):
    """
    Validates if the uploaded image is likely an MRI.
    
    Args:
        image (PIL.Image): The uploaded image.
        
    Returns:
        tuple: (bool, str) - Whether the image is valid and a validation message.
    """
    # Check the resolution
    if image.size[0] < 128 or image.size[1] < 128:
        return False, "The resolution of the image is too low to be an MRI."
    
    # Check the color mode (MRI images are typically grayscale)
    if image.mode != "L":
        return False, "The image does not appear to have the appropriate format for an MRI (grayscale or RGB)."
    
    # Optional: Add any additional heuristic checks here
    
    return True, "The uploaded image appears to be valid."

#st.logo('logo.png',size="large")
def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

my_logo = add_logo(logo_path="logo.png", width=90, height=90)
st.sidebar.image(my_logo)

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("ALZHEIMER'S DISEASE RECOGNITION SYSTEM", divider=True)
    image_path = "home_page.jfif"
    st.image(image_path,use_column_width=True)
    #st.image(image_path,width=900)
    st.markdown("""
    Welcome to the Alzheimer Disease Recognition System! 
    
    Our mission is to help in identifying Alzheimer's disease efficiently. Upload an MRI image of a brain, and our system will analyze it to detect any signs of diseases. Together, let's protect our elders and ourselves from Alzheimer's Disease!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an MRI image of a brain with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases and its severeness.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes CNN learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Alzheimer's Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)
#About Project
elif(app_mode=="About"):
    st.header("About",divider=True)
    st.markdown("""
        #### About Dataset
This dataset comprises a mix of real and synthetic axial MRIs and was developed to rectify the class imbalance in the original Kaggle Alzheimer's dataset, which featured four categories: "No Impairment", "Very Mild Impairment", "Mild Impairment", and "Moderate Impairment". Each category had 100, 70, 28, and 2 patients, respectively, and each patient's brain was sliced into 32 horizontal axial MRIs.
The MRI images were acquired using a 1.5 Tesla MRI scanner with a T1-weighted sequence. The images have a resolution of 128x128 pixels and are in the “.jpg” format. All images have been pre-processed to remove the skull.
However, it is important to note that the synthetic MRIs were not verified by a radiologist. Therefore, any results or indications from this dataset may or may not resemble real-world patient's symptoms or patterns. Moreover, there are no threats to privacy as these synthetic MRIs do not resemble any real-world patients
            """)   
    st.subheader("Content", divider=True) 
    st.markdown(""" 
        1. train (10240 images)
        2. test (1279 images)
              """)   
    st.markdown("""The train and test directories have 4 subdirectories which classifies the stages of Alzheimer Disease:
                """)
    st.markdown("""
        1. Mild Impairment
        2. Moderate Impairment
        3. No Impairment
        4. Very Mild Impairment""")
    st.subheader("Team", divider=True) 
    
    col1, col2, col3, col4 = st.columns(4,gap="medium", vertical_alignment="top")
    style_heading = 'text-align: center'
    style_image = 'display: block; margin-left: auto; margin-right: auto;'
    with col1:
        st.markdown(f"<h4 style='{style_heading}'>Rajatabha Das</h4>", unsafe_allow_html=True)
        #st.markdown(f"<img src='Rajatabha.png' style='{style_image}'/>", unsafe_allow_html=True)
        #st.subheader("Rajatabha Das")
        st.image("Rajatabha.png")

    with col2:
        st.markdown(f"<h4 style='{style_heading}'>Piyasha Kanrar</h4>", unsafe_allow_html=True)
        #st.subheader("Piyasha Kanrar")
        st.image("Piyasha.png")

    with col3:
        st.markdown(f"<h4 style='{style_heading}'>Anwesha Das Gupta</h4>", unsafe_allow_html=True)
        #st.subheader("Anwesha Das Gupta")
        st.image("Anwesha.png") 
    with col4:
        st.markdown(f"<h4 style='{style_heading}'>Swapna Mondal</h4>", unsafe_allow_html=True)
        #st.subheader("Swapna Mondal")
        st.image("swapna.png") 
               
#Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition", divider=True)
    test_image = st.file_uploader("Choose an Image:", type=["png", "jpg", "jpeg"])

    if test_image is not None:
        try:
            # Open the uploaded image
            image = Image.open(test_image)
            
            # Validate if it's an MRI image
            is_valid, validation_message = validate_mri_image(image)
            if not is_valid:
                st.error(validation_message)
            else:
                st.success(validation_message)
                
                # Save the file temporarily
                temp_file_path = "temp_uploaded_image.jpg"
                with open(temp_file_path, "wb") as f:
                    f.write(test_image.getbuffer())
                
                if st.button("Show Image"):
                    st.image(image, use_column_width=True)
                
                if st.button("Predict"):
                    with st.spinner("Predicting Results..."):
                        model_path = "alzheimer_cnn_model_20thnov.h5"
                        class_mapping = {
                            0: "Mild Demented",
                            1: "Moderate Demented",
                            2: "Non Demented",
                            3: "Very Mild Demented",
                        }
                        predicted_label = predict_single_image(model_path, temp_file_path, class_mapping)
                        st.success(f"### Prediction: {predicted_label}")
                
                # Optionally delete the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        except UnidentifiedImageError:
            st.error("The uploaded file is not a valid image.")