from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index_csv"),
    path('upload/', views.upload_file, name='upload_file'),
    path('success/', views.upload_success, name='upload_success'),
    path('analyze/<int:file_id>/', views.analyze_csv, name='analyze_csv'),
    path('analyze/<int:file_id>/analysis', views.analysis_model, name='analysis_model'),
    path('media/uploads/<str:file_name>', views.serve_file, name='serve_file'),
    path('view/uploads/<str:file_name>', views.view_file, name='view_file'),
    path('delete/<int:file_id>', views.delete_file, name='delete_file'),
    path('image/<str:file_name>', views.load_image, name='view_image'),
    path('analyze/<int:file_id>/normalize', views.normalize_data, name='normalize_data'),
]