import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import numpy as np

# Cache the model loading process
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/trained_model.h5')
    return model

# TensorFlow model prediction
def model_prediction(test_image):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

st.markdown("""
        <style>
               .block-container {
                    padding-top: 3rem;
                }
        </style>
        """, unsafe_allow_html=True)



app_mode = option_menu(
    menu_title = None,
    options = ["Home", "About Me", "Diagnosis"],
    icons=["house","info-circle","heart"],
    orientation = "horizontal",
    default_index=2
)


# Homepage
if(app_mode == "Home"):
    st.title("Welcome to Diabetic Retinopathy Detection")

    # Display an image of diabetic retinopathy
    # st.image("path_to_your_image/diabetic_retinopathy.jpg", use_column_width=True)  # Update the image path

    st.markdown("""
    Diabetic retinopathy is a serious eye condition that affects individuals with diabetes, potentially leading to vision loss. Our application leverages cutting-edge machine learning techniques to **detect diabetic retinopathy early**, enabling timely intervention and effective management.
    """)

    # Display the image
    st.image("img/stageDR.jpg", caption="Stages of Diabetic Retinopathy", use_column_width=True)


    st.markdown("""
    ## How It Works
    1. **Image Upload**: Use the file uploader to upload a retinal image.
    2. **Prediction**: Click the 'Predict' button to start the analysis.
    3. **Results**: Receive instant feedback on the presence and severity of diabetic retinopathy.

    ## Why Early Detection Matters
    Early diagnosis of diabetic retinopathy can significantly reduce the risk of severe vision impairment. Regular screenings and prompt interventions can help manage the disease effectively and protect your vision.
    Start your journey towards better eye health today!

    ### Ready to Get Started?
    - **Upload your retinal image now** and see how our tool can help you!
    """)

    # Additional Call to Action with a friendly reminder
    st.markdown("""

    🥳 **Let’s begin your journey to healthier eyes!** \n
    If you're ready to analyze your retinal image, head over to the **`Diagnosis`** page for predictions!
    """)



# About Me Page
elif app_mode == "About Me":
    st.header("About Me")
    st.markdown(
        """
        ## **Divyansh Kumar Singh**git
        **Final-Year Student | B.E. - Computer Science & Engineering**

        Hello! I’m Divyansh Kumar Singh, currently in my final year at the University Institute of Engineering and Technology, Panjab University. As a dedicated and passionate learner, I have a strong foundation in **software development**, particularly in **Java**, **Kotlin**, and **Android development** using the Android SDK. My focus has always been on leveraging technology to solve real-world problems, and I'm currently exploring **Artificial Intelligence** (AI) to further expand my skill set and knowledge.

        ### **Project: Diabetic Retinopathy Detection**
        This app represents my commitment to utilizing technology for impactful causes. Diabetic Retinopathy Detection is designed to aid early diagnosis of diabetic retinopathy using machine learning, with the goal of supporting healthcare professionals in detecting early signs of retinal damage for better patient outcomes.

        ### **Professional Links**
        - **LinkedIn**: [Divyansh Kumar Singh](https://www.linkedin.com/in/divyansh-kumar-singh-645311201/)
        - **GitHub**: [singhdivyanshdishu](https://github.com/singhdivyanshdishu)
        - **Email**: [singhbohdivydishu@gmail.com](mailto:singhbohdivydishu@gmail.com)

        ### **Education**
        - **B.E. - Computer Science & Engineering**
          University Institute of Engineering and Technology, Panjab University
          **2021 - 2025** | CGPA: 6.95 / 10

        ### **Certifications**
        - **Building an Android App with Jetpack Libraries**
        - **Complete Guide to Power BI for Data Analysts**

        ### **Skills**
        - **Programming Languages**: Java, Kotlin, Python
        - **Tools & Technologies**: Android SDK, Firebase, Power BI, SQL
        - **Soft Skills**: Problem-Solving, Communication, Leadership

        ### **Current Interests**
        My current interest lies in exploring AI and its applications in healthcare, with a vision to contribute to innovative solutions in diagnostics and patient care.

        Thank you for visiting my page!
        """
    )




# Prediction Page
elif app_mode == "Diagnosis":
    st.header("Diabetic Retinopathy Detection")

    # Section for downloading the test dataset
    st.markdown("### Download the Test Dataset")
    st.write("""
    Enhance your experience with the Diabetic Retinopathy Detection tool by downloading our comprehensive test dataset.
    This dataset contains a variety of retinal images, useful for practicing and testing the diagnostic capabilities
    of our models. Click the button below to download the dataset in `.rar` format.
    """)

    # Download button for the test dataset
    with open("test_dataset.rar", "rb") as f:
        bytes_data = f.read()

    st.download_button(label="Download Test Dataset",
                       data=bytes_data,
                       file_name="test_dataset.rar",
                       mime="application/octet-stream",
                       help="Click here to download the test dataset")

    st.markdown("---")  # Add a horizontal line for better separation

    # Image uploader with size limit (in bytes)
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"],
                                   help="Upload an image file (max size: 2MB)",
                                   label_visibility="collapsed")


    # Automatically display the uploaded image
    if test_image is not None:
        # Check the file size
        if test_image.size > 2 * 1024 * 1024:  # 2MB limit
            st.error("File size exceeds 2MB. Please upload a smaller image.")
            test_image = None
        else:
            # Display the image
            st.image(test_image, caption='Original Image', use_column_width=0.5)


            # Make prediction using the AI model
            if st.button("Predict", help="Make a prediction on the processed image"):
                with st.spinner("Processing... Please wait..."):
                    st.write("Our Prediction:")
                    result_index = model_prediction(test_image)

                    # Define class names
                    class_name = ['Mild', 'Moderate', 'Normal', 'Proliferate', 'Severe']
                    st.success(f"The model predicts it's a: **{class_name[result_index]}**")



    else:
        st.info("✨ **Please upload your retinal image for prediction.**  \n"
                 "To get started, simply select an image from your device.")
