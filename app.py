from typing import List, IO, Annotated

from fastapi import FastAPI, UploadFile, File
from starlette.responses import FileResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from dataframe import get_sample_from_files
from model import predict_class

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("public/index.html")


@app.post("/predict")
async def upload_files(files: List[UploadFile] = File(...)):
    files = [x.file for x in files]
    sample = get_sample_from_files(files)

    return {'result': str(predict_class(sample))}
