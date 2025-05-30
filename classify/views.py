from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import FileUploadForm
from requests import post
from .models import DetectFile
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
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
import cv2
from ultralytics import YOLO
from .detect import predict_all_flowers
import os
from django.conf import settings

model = YOLO("models/best_flower_2.onnx")

#  {0: 'tulip', 1: 'Lotus', 2: 'Lily', 3: 'hibiscus', 4: 'water_lily', 5: 'bougainvillea', 6: 'sunflower', 7: 'Hoa Phuong Do', 8: 'dandelion', 9: 'Lantana', 10: 'rose', 11: 'yellow_apricot_blossom', 12: 'orchid', 13: 'common_daisy'}
list_flower_en = {
    'tulip': 'Tulip',
    'common_daisy': 'Common Daisy',
    'Lotus': 'Lotus',
    'Lily': 'Lily',
    'hibiscus': 'Hibiscus',
    'water_lily': 'Water Lily',
    'bougainvillea': 'Bougainvillea',
    'sunflower': 'Sunflower',
    'Hoa Phuong Do': 'Royal poinciana',
    'dandelion': 'Dandelion',
    'Lantana': 'Lantana',
    'rose': 'Rose',
    'yellow_apricot_blossom': 'Yellow Apricot Blossom',
    'orchid': 'Orchid'
}

list_flower_vn = {
    'tulip': 'Tulip',
    'common_daisy': 'Cúc họa mi',
    'Lotus': 'Hoa sen',
    'Lily': 'Hoa ly',
    'hibiscus': 'Hoa dâm bụt',
    'water_lily': 'Hoa súng',
    'bougainvillea': 'Hoa giấy',
    'sunflower': 'Hoa hướng dương',
    'Hoa Phuong Do': 'Hoa phượng đỏ',
    'dandelion': 'Bồ công anh',
    'Lantana': 'Nguyệt quế',
    'rose': 'Hoa hồng',
    'yellow_apricot_blossom': 'Mai vàng',
    'orchid': 'Lan'
}

# Create your views here.
def index(request):
    form = FileUploadForm()
    uploaded_files = DetectFile.objects.all()
    uploaded_files = [file.file.name.split('/')[-1] for file in uploaded_files][::-1][:50]
    return render(request, 'classify/index.html', {'form': form, 'images': uploaded_files})

# def detect(request):
#     if request.method == 'POST':
#         form = FileUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             form.save()
#             URL = "http://localhost:8000/plants/detect2/"
#             files = {'files': open('media/images/' + str(request.FILES['file']), 'rb')}
#             response = post(URL, files=files)
#             # Print result in response to console
#             print(response.json())
#             return render(request, 'classify/result.html', {'image_url': str(request.FILES['file']), 'result': response.json().get('result'), 'status': response.json().get('status')})
#         else:
#             print(form.errors)
    
def image_view(request, file_name):
    try:
        with open(f"media/images/{file_name}", 'rb') as file:
            response = HttpResponse(file.read(), content_type="image/png")
            return response
    except:
        return HttpResponse("File Not Found")

def detect(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            input2 = request.FILES.get('file')
            print("Name image:",input2)
            temp_file_path = "image.jpg"
            with open(temp_file_path, 'wb') as temp_file:
                for chunk in input2.chunks():
                    temp_file.write(chunk)

            image_flower = cv2.imread(temp_file_path)
            image_flower = cv2.resize(image_flower, (224, 224))
            
            results = model(image_flower, stream=True)
            
            bboxes = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

                    bboxes.append((x1, y1, x2, y2))

                results = predict_all_flowers(bboxes=bboxes, image_original=image_flower)
                print(results)
                for result in results:
                    pred_label_decoded, conf, x1, y1, x2, y2 = result
                    cv2.rectangle(image_flower, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(image_flower, pred_label_decoded, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite("media/images/output.jpg", image_flower)
            
            labels = []
            for result in results:
                pred_label_decoded, conf, x1, y1, x2, y2 = result
                if 'language' in request.POST:
                    language = request.POST['language']
                    if language == 'vn':
                        pred_label_decoded = list_flower_vn.get(pred_label_decoded, pred_label_decoded)
                    elif language == 'en':
                        pred_label_decoded = list_flower_en.get(pred_label_decoded, pred_label_decoded)
                else:
                    pred_label_decoded = list_flower_vn.get(pred_label_decoded, pred_label_decoded)
                    
                labels.append(pred_label_decoded)
            os.remove(temp_file_path)

            return render(request, 'classify/result.html', {'image_url': str(request.FILES['file']), 'result': labels, 'status': "success", 'img_result': 'output.jpg'})
        
class DetectMobileView(APIView):
    def post(self, request):
        form = FileUploadForm(request.POST, request.FILES)
        if not form.is_valid():
            return Response({'status': 'error', 'message': 'Form không hợp lệ'}, status=status.HTTP_400_BAD_REQUEST)

        input_file = request.FILES.get('file')
        if not input_file:
            return Response({'status': 'error', 'message': 'Không có tệp được tải lên'}, status=status.HTTP_400_BAD_REQUEST)

        temp_file_path = os.path.join(settings.MEDIA_ROOT, 'temp_image.jpg')
        try:
            with open(temp_file_path, 'wb') as temp_file:
                for chunk in input_file.chunks():
                    temp_file.write(chunk)

            image_flower = cv2.imread(temp_file_path)
            if image_flower is None:
                return Response({'status': 'error', 'message': 'Tệp ảnh không hợp lệ'}, status=status.HTTP_400_BAD_REQUEST)

            image_flower = cv2.resize(image_flower, (224, 224))
            results = model(image_flower, stream=True)
            bboxes = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    bboxes.append((x1, y1, x2, y2))

            prediction_results = predict_all_flowers(bboxes=bboxes, image_original=image_flower)
            print(prediction_results)

            labels = []
            for result in prediction_results:
                pred_label_decoded, conf, x1, y1, x2, y2 = result
                cv2.rectangle(image_flower, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(image_flower, pred_label_decoded, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # If request has a language parameter, use it to determine the label
                if 'language' in request.data:
                    language = request.data['language']
                    if language == 'vn':
                        pred_label_decoded = list_flower_vn.get(pred_label_decoded, pred_label_decoded)
                    elif language == 'en':
                        pred_label_decoded = list_flower_en.get(pred_label_decoded, pred_label_decoded)
                else:
                    pred_label_decoded = list_flower_vn.get(pred_label_decoded, pred_label_decoded)

                labels.append(pred_label_decoded)

            output_path = os.path.join(settings.MEDIA_ROOT, 'images', 'output.jpg')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image_flower)

            response_data = {
                'image_url': 'classify/image/' + input_file.name,
                'result': labels,
                'status': 'success',
                'img_result': 'classify/image/output.jpg'
            }
            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'status': 'error', 'message': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)