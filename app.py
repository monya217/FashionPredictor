from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
from io import BytesIO

# Load trained model
model = load_model("fashion_cnn_model.keras")

# Class names (Fashion MNIST)
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

app = FastAPI()

@app.get("/")
def home():
    return HTMLResponse("""
    <h2>Fashion MNIST Classifier 👗👟</h2>
    <form action="/predict" enctype="multipart/form-data" method="post">
    <input name="file" type="file">
    <input type="submit">
    </form>
    """)

from io import BytesIO

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()   # read file bytes
    img = image.load_img(BytesIO(contents), target_size=(28,28), color_mode="grayscale")
    
    # Convert to array + preprocess
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array, verbose=0)
    pred_class = np.argmax(pred, axis=1)[0]
    
    return {"filename": file.filename, "prediction": class_names[pred_class]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


