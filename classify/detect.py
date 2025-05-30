import cv2
from ultralytics import YOLO
import math

import torch
from torch import nn
import torchvision
from torchvision import transforms
import ast
from PIL import Image
from typing import Tuple, List
import numpy as np
from queue import Queue
from threading import Thread
import time
from .vit_model_config import model_vit, dict_classes, pretrained_vit_transforms, device

model = YOLO("models/best_flower_2.onnx")


def predict_flower_v2(image:Image)->Tuple[str, float]:
    image = pretrained_vit_transforms(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    model_vit.eval()
    with torch.no_grad():
        preds:torch.Tensor = model_vit(image)
        pred_label_encoded = preds.argmax(dim=1)
        # get confidence
        # conf = torch.nn.functional.softmax(preds, dim=1)[0][preds.item()].item()
        # print(conf)
        conf = torch.nn.functional.softmax(preds, dim=1)[0][pred_label_encoded.item()].item()
        pred_label_decoded = dict_classes[pred_label_encoded.item()]
    
    return pred_label_decoded, conf

def handle_flower_recognition(args):
    image_original, bbox, result_queue = args
    x1, y1, x2, y2 = bbox
    image = image_original[y1:y2, x1:x2]
    image = Image.fromarray(image)  
    pred_label_decoded, conf = predict_flower_v2(image)
    result_queue.put((pred_label_decoded, conf, x1, y1, x2, y2))
    pass

def predict_all_flowers(bboxes:list, image_original:np.ndarray)->List[Tuple[str, float, int, int, int, int]]:
    results = []
    result_queue = Queue()
    threads = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        # image = image_original[y1:y2, x1:x2]
        # image = Image.fromarray(image)
        # pred_label_decoded, conf = predict_flower_v2(image)
        # results.append((pred_label_decoded, conf, x1, y1, x2, y2))
        thread = Thread(target=handle_flower_recognition, args=((image_original, bbox, result_queue),), daemon=True)
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    while not result_queue.empty():
        results.append(result_queue.get())

    return results