import json
from fastapi import FastAPI, HTTPException, File, UploadFile
from keras.preprocessing.image import load_img, img_to_array
from pydantic import BaseModel

from caption import predict

app = FastAPI()

class ImageInput(BaseModel):
    img_path: str

@app.post("/predict")                        
def predict_caption(image_input: ImageInput):
    img_path = image_input.img_path
    caption = predict(img_path)
    if caption:
        start = "<start>"
        end = "<end>"

        start_idx = caption.find(start) + len(start)
        end_idx = caption.find(end)

        result = caption[start_idx:end_idx].strip()

        return {"caption": result}
        
    else:
        raise HTTPException(status_code=404, detail="Not found image")   
