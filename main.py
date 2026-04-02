from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from model import process_image_bytes

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    result_image = process_image_bytes(image_bytes)

    return StreamingResponse(result_image, media_type="image/png")
