from pathlib import Path
import streamlit as st
import numpy as np

from PIL import Image
from sklearn.neighbors import NearestNeighbors

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array

BASE_DIR = Path(__file__).resolve().parent


st.set_page_config(page_title="Jewelry Search")

TOP_K = 25


@st.cache_resource
def load_model():

    model = MobileNetV2(weights="imagenet",include_top=False,pooling="avg")
    model.trainable = False
    return model



@st.cache_resource
def load_search_index():

    data = np.load("jewelry.npz")

    embeddings = data["embeddings"]
    image_paths = data["image_paths"]

    NN = NearestNeighbors(n_neighbors=TOP_K)
    NN.fit(embeddings)
    return embeddings, image_paths, NN



def extract_embedding(model, Img):

    Img = Img.convert("RGB")
    Img = Img.resize((224, 224))
    Img_Array = img_to_array(Img)

    Img_Array = np.expand_dims(Img_Array , axis=0)

    Img_Array = preprocess_input(Img_Array)

    embedding = model.predict(Img_Array)
    return embedding



model = load_model()
embeddings, image_paths, NN = load_search_index()


st.title("Jewelry Search")

st.write("Upload a jewelry image ","to find visually similar jewelry")


source = st.radio("Choose image source:",["Upload", "Camera"])
Query_Image = None


if source == "Upload":
    uploaded_file = st.file_uploader("Upload an image",type=["jpg", "png"])

    if uploaded_file is not None:
        Query_Image = Image.open(uploaded_file)

else:

    camera_file = st.camera_input("Camera")

    if camera_file is not None:
        Query_Image = Image.open(camera_file)


if Query_Image is not None:

    st.subheader("Query Image")
    st.image(Query_Image)

    Threshold = st.slider("Minimum Similarity",min_value=0.0,max_value=1.0,)


    Query_Embedding = extract_embedding(model,Query_Image)


    distances, indices = NN.kneighbors(Query_Embedding)

    matches = []

    for distance, idx in zip(distances[0],indices[0]):

        similarity = 1 - distance

        if similarity >= Threshold:

            matches.append((image_paths[idx],similarity))



    if len(matches) == 0:
        st.warning("Try another image")
    else:
        st.subheader("Similar Jewelry Results")

