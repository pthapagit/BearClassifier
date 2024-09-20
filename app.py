import streamlit as st
from fastai.vision.all import *

st.title("Classify your Bear")

learn = load_learner('export.pkl')

labels = learn.dls.vocab


def Predict(image):
    img = PILImage.create(image)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


col1, col2 = st.columns(2)

with col1:
    # File uploader for image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    submit_button = st.button("Submit")

    if uploaded_image:
        predictions = Predict(uploaded_image)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Default view when no image is uploaded
    else:
        st.write("Upload an image to see the progress bars.")

with col2:
    # Create the progress bars
    latest_iteration = st.empty()
    progress_bar_black = st.progress(0, text="Black Bear")
    progress_bar_grizzly = st.progress(0, text="Grizzly Bear")
    progress_bar_teddy = st.progress(0, text="Teddy Bear")

    if submit_button:
        progress_bar_black.progress(predictions['black'], text=f"Black Bear: {predictions['black'] * 100:.2f}%")
        progress_bar_grizzly.progress(predictions['grizzly'], text=f"Grizzly Bear: {predictions['grizzly'] * 100:.2f}%")
        progress_bar_teddy.progress(predictions['teddy'], text=f"Teddy Bear: {predictions['teddy'] * 100:.2f}%")
