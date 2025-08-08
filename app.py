import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import requests
import torch
from ultralytics.nn.tasks import DetectionModel

# Allow this class to be unpickled
torch.serialization.add_safe_globals([DetectionModel])

# Load environment variables
api_key = st.secrets["SPOONACULAR_API_KEY"]

# Load your model
model = YOLO('best_v2.pt')

st.title("Fridge Ingredient Detector 🍽️")
st.write("Upload a picture of your fridge (or use your camera) to get recipe ideas!")

# Choose input method
input_method = st.radio("Choose image input method:", ["Upload Image", "Use Camera"])

img = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

elif input_method == "Use Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        img = cv2.imdecode(np.frombuffer(camera_image.getvalue(), np.uint8), cv2.IMREAD_COLOR)

# Proceed if an image is loaded
if img is not None:
    # Run YOLO inference
    results = model(img)
    confidence_threshold = 0.7
    detected_classes = []

    # Get original image for display (convert BGR to RGB)
    input_img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get annotated image (drawn boxes/labels)
    annotated_img = results[0].plot()  # Returns image with boxes and labels

    # Filter boxes and extract class names
    for result in results:
        high_conf_boxes = [box for box in result.boxes if float(box.conf) > confidence_threshold]
        for box in high_conf_boxes:
            class_idx = int(box.cls)
            class_name = model.names[class_idx]
            detected_classes.append(class_name)

    # Display original and YOLO image side-by-side
    st.subheader("Input & Detected Image")
    col1, col2 = st.columns(2)
    with col1:
        st.image(input_img_display, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.image(annotated_img, caption="Detected Objects", use_column_width=True)

    # Display detected ingredients
    if detected_classes:
        st.success(f"Detected ingredients: {', '.join(set(detected_classes))}")
    else:
        st.warning("No ingredients detected with high confidence.")

    # Call Spoonacular API
    if detected_classes and api_key:
        url = "https://api.spoonacular.com/recipes/findByIngredients"
        params = {
            "ingredients": ",".join(detected_classes),
            "ignorePantry": True,
            "ranking": 2,
            "number": 5,
            "apiKey": api_key
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            st.subheader("Recommended Recipes 🍳")

            # Initialize index in session state
            if "recipe_index" not in st.session_state:
                st.session_state.recipe_index = 0

            # Navigation buttons
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous"):
                    st.session_state.recipe_index = (st.session_state.recipe_index - 1) % len(data)
            with col3:
                if st.button("Next ➡️"):
                    st.session_state.recipe_index = (st.session_state.recipe_index + 1) % len(data)

            # Current recipe
            recipe = data[st.session_state.recipe_index]
            st.image(recipe['image'], width=300)
            st.markdown(f"### {recipe['title']}")
            st.markdown(f"**Used Ingredients**: {', '.join([i['name'] for i in recipe['usedIngredients']])}")
            st.markdown(f"**Missing Ingredients**: {', '.join([i['name'] for i in recipe['missedIngredients']])}")

            # Optional: Show position indicator
            st.caption(f"Recipe {st.session_state.recipe_index + 1} of {len(data)}")
        else:
            st.error("Failed to fetch recipes. Check your API key and usage quota.")
    elif not api_key:
        st.error("Missing Spoonacular API key. Make sure it's set in your environment or secrets.")

