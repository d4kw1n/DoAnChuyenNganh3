from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index_classify"),
    path('image-detect/', views.detect, name='detect_image'),
    path('image/<str:file_name>', views.image_view, name='image_view'),
    path('image/mobile/detect/', views.DetectMobileView.as_view(), name='detect_mobile'),
]