from tensorflow.keras.models import model_from_json
import cv2
import numpy as np
from PIL import Image
import streamlit as st

#directory="C:/Users/USER/Desktop/deploy_fr/"
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from Kaggle server")

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


# Defining a function that will do the detections
def detect(color,gray):
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(color, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = color[y:y+h, x:x+w]
      
    return color,faces,roi_gray

def prediction(roi_gray):
    roi=cv2.resize(roi_gray,(48,48))
    prediction = loaded_model.predict(roi[np.newaxis, :, :, np.newaxis])
    # Custom Symbols to print with text of emotion.
    # Symbols = {"Happy": ":)", "Sad": ":}", "Surprise": "!!","Angry": "?", "Disgust": "#", "Neutral": ".", "Fear": "~"}

    # Defining the Parameters for putting Text on Image
    #Text = str(prediction) + Symbols[str(prediction)]
    return label_map[np.argmax(prediction[0])]


def main():
    st.title("Face Detection And Facial Expression recognition :sunglasses: ")
    st.write("**Model:Haar cascade and Deep Neural Network**")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)
    if choice == "Home":
        st.write("Go to the About section from the sidebar to learn more about it.")
        image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

        if image_file is not None:
            img = Image.open(image_file)
            if st.button("Process"):
                color_img=np.array(img.convert('RGB'))
                gray_img = cv2.imread(directory+'images.jpg',0)
                result_img, result_faces ,roi_gray= detect(color_img,gray_img)
                roi=cv2.resize(roi_gray,(48,48))
                prediction = loaded_model.predict(roi[np.newaxis, :, :, np.newaxis])
                Text=label_map[np.argmax(prediction[0])]
                st.image(result_img, use_column_width = True)
                st.success("Expression: {} ".format(Text))
    elif choice == "About":
    	about()

if __name__ == "__main__":
    main()

