# ALZHEIMER'S DISEASE RECOGNITION SYSTEM

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
    
#### About Dataset
This dataset comprises a mix of real and synthetic axial MRIs and was developed to rectify the class imbalance in the original Kaggle Alzheimer's dataset, which featured four categories: "No Impairment", "Very Mild Impairment", "Mild Impairment", and "Moderate Impairment". Each category had 100, 70, 28, and 2 patients, respectively, and each patient's brain was sliced into 32 horizontal axial MRIs.
The MRI images were acquired using a 1.5 Tesla MRI scanner with a T1-weighted sequence. The images have a resolution of 128x128 pixels and are in the “.jpg” format. All images have been pre-processed to remove the skull.
However, it is important to note that the synthetic MRIs were not verified by a radiologist. Therefore, any results or indications from this dataset may or may not resemble real-world patient's symptoms or patterns. Moreover, there are no threats to privacy as these synthetic MRIs do not resemble any real-world patients.

1. train (10240 images)
2. 2. test (1279 images)

The train and test directories have 4 subdirectories which classifies the stages of Alzheimer Disease:
1. Mild Impairment
2. Moderate Impairment
3. No Impairment
4. Very Mild Impairment

The model gives an accuracy of 90.5%.
