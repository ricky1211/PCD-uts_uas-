![headergithub](https://github.com/ricky1211/PCD-uts_uas-/assets/115614668/c835f740-78c3-4b65-ae69-073004d52c60)  
# UAS-UTS Pengolahan Citra Digital
Teknik Informatika 22 A4

|               NAMA              |    NIM    |
| ------------------------------- | --------- |
| Mikael Rivaldo                  | 312210378 |
| Ricky Alfian Saputra            | 312210279 |
| Rafi Maulana Firdaus            | 312210382 |
| Dimas Adi Nugraha               | 312210409 |

------------------------------------------------

## UTS (Udemy)
### Face Eye Detection
Program untuk mendeteksi wajah dan mata menggunakan OpenCV dan Streamlit.  
Berikut Kode program nya :

``` py
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np


face_cascade = cv2.CascadeClassifier('./detector/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./detector/haarcascade_eye.xml')

def detect_faces(our_image):
    new_img = np.array(our_image.convert("RGB"))
    faces = face_cascade.detectMultiScale(new_img, 1.1, 8)
    for (x, y, w, h) in faces:
        cv2.rectangle(new_img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    return new_img, faces

def detect_eye(our_image):
    new_img = np.array(our_image.convert("RGB"))
    eyes = eye_cascade.detectMultiScale(new_img, 1.1, 20)
    for (x, y, w, h) in eyes:
        cv2.rectangle(new_img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    return new_img, eyes

def Cartoonize_image(our_image):
    new_img = np.array(our_image.convert("RGB"))
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADDAPTIVE_TRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(new_img, 9, 300, 300)
    Cartoon = cv2.bitwise_and(color, color, mask = edges)
    return Cartoon
def main():
    st.title('EDIT GAMBAR HERE')
    st.text('GASKAAANNN')

    activities = ['Detection', 'About']
    choice = st.sidebar.selectbox('Select Activity', activities)

    if choice == 'Detection':
        st.subheader('Face Detection')
        image_file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text('Original Image')
            st.image(our_image)

            enhance_type = st.sidebar.radio("Enhance type", ['Original', 'Gray-scale', 'Contrast', 'Brightness', 'Blurring', 'Sharpness'])

            if enhance_type == 'Gray-scale':
                img = np.array(our_image.convert('RGB'))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                st.image(gray, channels="GRAY")
            elif enhance_type == "Contrast":
                rate = st.sidebar.slider("Contrast", 0.5, 6.0)
                enhancer = ImageEnhance.Contrast(our_image)
                enhanced_img = enhancer.enhance(rate)
                st.image(enhanced_img)
            elif enhance_type == "Brightness":
                rate = st.sidebar.slider("Brightness", 0.0, 8.0)
                enhancer = ImageEnhance.Brightness(our_image)
                enhanced_img = enhancer.enhance(rate)
                st.image(enhanced_img)
            elif enhance_type == "Blurring":
                rate = st.sidebar.slider("Blurring", 0.0, 7.0)
                blurred_img = cv2.GaussianBlur(np.array(our_image), (15, 15), rate)
                st.image(blurred_img)
            elif enhance_type == "Sharpness":
                rate = st.sidebar.slider("Sharpness", 0.0, 14.0)
                enhancer = ImageEnhance.Sharpness(our_image)
                enhanced_img = enhancer.enhance(rate)
                st.image(enhanced_img)
            elif enhance_type == "Original":
                st.image(our_image, width = 300)
            else :
                st.image(our_image, width = 300)
        tasks = ["Faces", "Eyes", "Cartoonize", "Cannize"]
        feature_choice = st.sidebar.selectbox("Find features", tasks)
        if st.button("Process"):
            if feature_choice == "Faces":
                result_img, result_face = detect_faces(our_image)
                st.image(result_img)
                st.success("{} Wajah Terdeteksi".format(len(result_face)))

            if feature_choice == "Eyes":
                result_img, result_eye = detect_eye(our_image)
                st.image(result_img)
                st.success("{} Mata Terdeteksi".format(len(result_eye)))

            elif feature_choice == "Cartoonize":
                result_img, Cartoonize_img = detect_cartoonize(our_image)
                st.image(result_img)
                st.success("{} Mata Terdeteksi".format(len(result_eye)))
                
    elif choice == 'About':
        st.subheader('About Developer')
        st.markdown('Built with Streamlit by MIKEL')
        st.text('My name is Mikael Rivaldo I AM GENJI.')

if __name__ == "__main__":
    main()
```
# Face Detect :  
Disini kami menggunakan foto Ghozali sebagai objek yang kami coba.

<img src="https://github.com/MikaelRivaldo/PCD-Streamlit/assets/115770247/d91d2449-c895-4101-a8fa-fc5712eae378" alt="WhatsApp Image 2024-07-03 at 20 00 24" width="300">
  
Berikut adalah hasil kode diatas:  
<img src="https://github.com/ricky1211/PCD-uts_uas-/assets/115770247/57db3967-cc45-4fe0-815c-46d6b4e5b381" alt="image" width="300">

-----------------------------  
# Hasil detect Mata :  
Setelah deteksi wajah, kemudian eye detect.  
![image](https://github.com/ricky1211/PCD-uts_uas-/assets/115770247/33f00fa4-caa3-485a-9082-2ff583b6e5a1)









## UAS (Image segmentation with K-means Clustering)
Program untuk mendeteksi warna dengan program numpy dan matplotlib.

Berikut adalah Codingan programnya:


![image](https://github.com/ricky1211/PCD-uts_uas-/assets/115770247/303286c1-3992-4b14-a655-eb5f8e2b3dbf)

![image](https://github.com/ricky1211/PCD-uts_uas-/assets/115770247/abe5666b-9b74-4b87-9cfe-e07ab12f6ded)

![image](https://github.com/ricky1211/PCD-uts_uas-/assets/115770247/b99e6c15-f6ed-4c79-b533-98475764d239)

![image](https://github.com/ricky1211/PCD-uts_uas-/assets/115770247/5395de5c-ebe0-4a76-9d10-df1afb3f3995)






