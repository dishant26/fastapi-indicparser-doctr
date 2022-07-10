from fastapi import FastAPI, Query, Path, File, UploadFile, Form, Depends
from fastapi import Depends, FastAPI, File, Form
from enum import Enum
from use_doctr import doctr
from pydantic import BaseModel, validator, BaseSettings, Json
from starlette.requests import Request
from pydantic import BaseModel
from typing import Optional
import os
import cv2
import numpy as np
import json
import ast
try:
 from PIL import Image
except ImportError:
 import Image
import shutil
from indicparser import indic_parser

class model_name(str, Enum):
    pytesseract = 'Pytesseract'
    doctr = 'DocTR'
    none = None


app = FastAPI()

@app.post('/indicparser')
async def upload_file(file: UploadFile, 
                      inference: str = Form(...), 
                      lang: str = Form(...), 
                      model: Optional[str] = Form(None), 
                      confidence_threshold: Optional[float] = Form(None)):
    print(file.filename)
    with open("temporary.png", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    img = cv2.imread("temporary.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("temporary.png", cv2.IMREAD_COLOR)
    pil_file=Image.fromarray(img)
    output = indic_parser(inference, lang, pil_file, file.filename, model, confidence_threshold, img2)
    return output
 
@app.post('/doctr')
async def choose_model(model: model_name, File: UploadFile):
    if model == model_name.pytesseract:
        return f"You've selected {model} for OCR. Please upload the .jpg, .jpeg or .png file."
    elif model == model_name.doctr:
        FILE = doctr(File)
        return FILE.ocr()
        # return f"You've selected {model} for OCR. Please upload the .jpg, .jpeg or .png or .pdf file."

    else:
        return f"Please select a model to OCR the image."
 