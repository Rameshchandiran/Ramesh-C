import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the VGG16 and CNN model
model_vgg = load_model('corn_VGG_model.keras')
model_cnn = load_model('corn_CNN_model.keras')

# Class labels (update this to match your model's output labels)
class_labels = ['Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# Function to show the home page with text and images in two columns
def show_home_page():
    st.title("Corn Plant Disease Detection")
    st.subheader("Plant Diseases and Their Symptoms")

    # Creating two columns: col1 for text, col2 for images
    col1, col2 = st.columns(2)

    # Common Rust disease description and image
    with col1:
        st.write("""
        **Common Rust**:
        - **What it is**: Common Rust is a disease caused by a fungal pathogen _Puccinia sorghi_, which infects corn leaves and stalks.
        - **How it happens**: The fungus produces reddish-orange pustules that release spores which spread to other plants.
        - **Symptoms**: Yellow to brown spots on leaves, leaf curling, reduced plant growth.
        """)
    with col2:
        # You can add an image for Common Rust here (ensure the image is in the same directory or provide the correct path)
        img_common_rust = Image.open("common_rust_image.jpg")
        st.image(img_common_rust, caption="Common Rust on Corn")

    # Gray Leaf Spot disease description and image
    st.write("---")  # Horizontal line to separate diseases
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        **Gray Leaf Spot**:
        - **What it is**: Gray Leaf Spot is caused by the fungus _Cercospora zeae-maydis_ and affects corn plants.
        - **How it happens**: The fungus thrives in humid conditions and spreads via wind and rain, attacking the leaves.
        - **Symptoms**: Lesions with gray or tan centers and dark borders on leaves.
        """)
    with col2:
        # You can add an image for Gray Leaf Spot here (ensure the image is in the same directory or provide the correct path)
        img_gray_leaf_spot = Image.open("gray_leaf_spot_image.jpg")
        st.image(img_gray_leaf_spot, caption="Gray Leaf Spot on Corn")

    # Healthy Corn plant description and image
    st.write("---")  # Horizontal line to separate diseases
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        **Healthy**:
        - **What it is**: Healthy corn plants are free of diseases and have green, vibrant leaves.
        - **Symptoms**: Uniform green color with no signs of disease or damage.
        """)
    with col2:
        # You can add an image for Healthy Corn here (ensure the image is in the same directory or provide the correct path)
        img_healthy = Image.open("healthy_corn_image.jpg")
        st.image(img_healthy, caption="Healthy Corn Plant")

    # Prevention and control section
    st.write("---")  # Horizontal line to separate diseases
    st.write("""
    **Prevention and Control**:
    - Crop rotation, resistant varieties, and fungicide application can help manage these diseases.
    """)

# Function to show the prediction page
def show_prediction_page():
    st.title("Disease Prediction Page")
    
    # Upload image widget
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image.")
        st.write("Classifying...")

        # Image preprocessing
        img = img.resize((224, 224))  # Resize image to match the model's input size
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Rescale image to [0, 1]

        # Dropdown for model selection
        model_name = st.selectbox("Select Model", ['VGG', 'CNN'])

                # Select the model based on user input
        if model_name == 'VGG':
            model = model_vgg
        else:
            model = model_cnn

        # Predict button
        if st.button("Predict"):
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions)]
            confidence = np.max(predictions) * 100
            st.write(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}%")

# Function to handle login
def login():
    # Default login credentials
    username = "Admin"
    password = "123"
    
    st.title("Login Page")
    
    # Username and password input fields
    user_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if user_input == username and password_input == password:
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Invalid credentials, please try again.")

# Main function to switch between pages
def main():
    # Initialize session state for login
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Display login page if not logged in
    if not st.session_state.logged_in:
        login()
    else:
        # Create a navigation sidebar
        page = st.sidebar.selectbox("Select a page", ["Home", "Prediction"])
        
        if page == "Home":
            show_home_page()
        elif page == "Prediction":
            show_prediction_page()

if __name__ == "__main__":
    main()
