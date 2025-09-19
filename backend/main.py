from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image as PILImage
import tensorflow as tf

IMG_SIZE = 224

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("cat_classifier.h5")

CLASS_NAMES = [
    "Abyssinian", "american_bulldog", "american_pit_bull_terrier", "basset_hound",
    "beagle", "Bengal", "Birman", "bombay", "boxer", "British_Shorthair",
    "chihuahua", "Egyptian_Mau", "english_cocker_spaniel", "english_setter",
    "german_shorthaired", "great_pyrenees", "havanese", "japanese_chin",
    "keeshond", "leonberger", "Maine_Coon", "miniature_pinscher", "newfoundland",
    "Persian", "pomeranian", "pug", "Ragdoll", "Russian_Blue", "saint_bernard",
    "samoyed", "scottish_terrier", "shiba_inu", "Siamese", "Sphynx",
    "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = PILImage.open(BytesIO(contents)).convert("RGB").resize((IMG_SIZE,IMG_SIZE))
        x = np.expand_dims(np.array(img), axis=0) / 255.0
        

        preds = model.predict(x)
        pred_idx = np.argmax(preds, axis=1)[0]
        species = CLASS_NAMES[pred_idx]
        species = species.replace("_", " ").capitalize()
        probability = float(np.max(preds) * 100)

        return {"species": species, "probability": round(probability, 2)}
    except Exception as e:
        return {"error": str(e)}