### Script for CS329s ML Deployment Lec 
import os
import json
import requests
import SessionState
import streamlit as st
import tensorflow as tf
from utils import load_and_prep_image, classes_and_models, update_logger, predict_json

#tf.enable_eager_execution()

# Setup environment credentials (you'll need to change these)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "virtual-signer-319900-7022eb9969cd.json" # change for your GCP key
PROJECT = "virtual-signer-319900" # change for your GCP project
REGION = "us-east4" # change for your GCP region (where your model is hosted)

### Streamlit code (works as a straigtht-forward script) ###
st.title("IDC Detection")
st.header("Identify histopathology images")

@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(image, model, class_names):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    image = load_and_prep_image(image)
    # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    #image = tf.expand_dims(image, axis=0)
    preds = predict_json(project=PROJECT,
                         region=REGION,
                         model=model,
                         instances=image)
    # print('preds...', preds)
    pred_class = class_names[tf.argmax(preds[0])]
    #pred_conf = preds[0]
    pred_conf = tf.reduce_max(preds[0])
    print('preds...', preds, pred_conf)

    return image, pred_class, pred_conf

# Pick the model version
choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1 (DenseNet)", 
     "Model 2 (AlexNet)", 
     "Model 3 (Xception)") 
)

# Model choice logic
if choose_model == "Model 1 (DenseNet)":
    CLASSES = classes_and_models["model_1"]["classes"]
    MODEL = classes_and_models["model_1"]["model_name"]
elif choose_model == "Model 2 (AlexNet)":
    CLASSES = classes_and_models["model_2"]["classes"]
    MODEL = classes_and_models["model_2"]["model_name"]
else:
    CLASSES = classes_and_models["model_3"]["classes"]
    MODEL = classes_and_models["model_3"]["model_name"]

# Display info about model and classes
if st.checkbox("Show classes"):
    st.write(f"You chose {MODEL}, these are the classes of IDC (1 or 0) it can identify:\n", CLASSES)

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload an image of food",
                                 type=["png", "jpeg", "jpg"])

# Setup session state to remember state of app so refresh isn't always needed
# See: https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11 
session_state = SessionState.get(pred_button=False)

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True 

# And if they did...
if session_state.pred_button:
    session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(session_state.uploaded_image, model=MODEL, class_names=CLASSES)
    print("USING *****", MODEL, CLASSES)
    st.write(f"Prediction: {session_state.pred_class} , \
               Confidence: {session_state.pred_conf:.4f} ,  with {MODEL}")

# TODO: code could be cleaned up to work with a main() function...
# if __name__ == "__main__":
#     main()