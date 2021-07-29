import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
from model import FacialExpressionModel


# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
#eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

model = FacialExpressionModel("model.json", "model_weights.h5")

# Defining a function that will do the detections
def detect(color,gray):
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(color, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = color[y:y+h, x:x+w]
      
    return color,faces,roi_gray





def about():
	st.write(
		'''
		**Haar Cascade** is an object detection algorithm.
		It can be used to detect objects in images or videos. 

		The algorithm has four stages:

			1. Haar Feature Selection 
			2. Creating  Integral Images
			3. Adaboost Training
			4. Cascading Classifiers



View Dev-k web :point_right: http://dev-k-copyright.herokuapp.com/
		''')
Text=""
def prediction(roi_gray):
    roi=cv2.resize(roi_gray,(48,48))
    # Let us make the Prediction of Emotion present in the Image.
    prediction = model.predict_emotion(
        roi[np.newaxis, :, :, np.newaxis])

    # Custom Symbols to print with text of emotion.
    Symbols = {"Happy": ":)", "Sad": ":}", "Surprise": "!!",
                "Angry": "?", "Disgust": "#", "Neutral": ".", "Fear": "~"}

    # Defining the Parameters for putting Text on Image
    Text = str(prediction) + Symbols[str(prediction)]
    return Text


def main():
    st.title("Face Detection And Facial Expression recognition :sunglasses: ")
    st.write("**Model:Haar cascade and Deep Neural Network**")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)
    if choice == "Home":

    	st.write("Go to the About section from the sidebar to learn more about it.")
        # You can specify more file types below if you want
    	image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
    	if image_file is not None:
            image = Image.open(image_file) 
            if st.button("Process"):
                
                # result_img is the image with rectangle drawn on it (in case there are faces detected)
                # result_faces is the array with co-ordinates of bounding box(es)
                image=np.array(image.convert('RGB'))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                result_img, result_faces ,roi_gray= detect(image,gray)
                Text=prediction(roi_gray)
                st.image(result_img, use_column_width = True)
                st.success("Expression: {} ".format(Text))

    elif choice == "About":
    	about()




if __name__ == "__main__":
    main()


