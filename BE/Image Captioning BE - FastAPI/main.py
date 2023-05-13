import json
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from caption import predict

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)