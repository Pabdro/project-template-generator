from fastapi import FastAPI, Depends
from starlette.middleware.cors import CORSMiddleware
from src.llm_service import TemplateLLM
from src.prompts import ProjectParams
from src.parsers import ProjectIdeas
import io
import cv2
from fastapi import (
    FastAPI, 
    UploadFile, 
    File, 
    HTTPException, 
    status,
    Depends
)
from fastapi.responses import Response
import numpy as np
from PIL import Image
from datetime import datetime as dt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Any
import numpy as np
from ultralytics import YOLO

FACE_MODEL_PATH = "src/blaze_face_short_range.tflite"
    
class FaceDetector:
    def __init__(self, model_path=FACE_MODEL_PATH):
        base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE
            )
        self.model = vision.FaceDetector.create_from_options(options)
    def predict_image(self, image_array: np.ndarray):
        mp_image= mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
        detection = self.model.detect(mp_image)
        results = []
        for detection in detection.detections:
            bbox = detection.bounding_box
            detection_dict = {
                "bbox": [bbox.origin_x, bbox.origin_y, bbox.width, bbox.height],
                "keypoints": [(kp.x, kp.y) for kp in detection.keypoints]
            }
            results.append(detection_dict)
        return results

app = FastAPI()

face_detector = FaceDetector()

def get_face_detector():
    return face_detector

def predict_uploadfile(predictor, file):
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 
            detail="No es una imagen"
        )
    img_obj = Image.open(img_stream)
    img_array = np.array(img_obj)
    return predictor.predict_image(img_array), img_array

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_llm_service():
    return TemplateLLM()


@app.post("/generate")
def generate_project(params: ProjectParams, service: TemplateLLM = Depends(get_llm_service)) -> ProjectIdeas:
    return service.generate(params)

@app.post("/faces", responses={
    200: {"content": {"image/jpeg": {}}}
})
def detect_faces(
    file: UploadFile = File(...), 
    predictor: FaceDetector = Depends(get_face_detector)
) -> Response:
    results, img = predict_uploadfile(predictor, file)
    for result in results:
        bbox = result['bbox']
        keypoints = result['keypoints']
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                            (0, 255, 0), 2)
        for i in range(2):  
            x = int(keypoints[i][0] * img.shape[1])  
            y = int(keypoints[i][1] * img.shape[0])  
            img = cv2.circle(img, (x, y), 5, (0, 255, 0), 2)
        if len(keypoints) >= 3:  
            x = int(keypoints[2][0] * img.shape[1]) 
            y = int(keypoints[2][1] * img.shape[0]) 
            img = cv2.circle(img, (x, y), 5, (0, 255, 0), 2)
        if len(keypoints) >= 4:  
            x = int(keypoints[3][0] * img.shape[1])  
            y = int(keypoints[3][1] * img.shape[0])  
            img = cv2.circle(img, (x, y), 5, (0, 255, 0), 2)
    img_pil = Image.fromarray(img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.get("/")
def root():
    return {"status": "OK"}
