from django.shortcuts import render, redirect
from .forms import FileUploadForm
from .models import UploadedFile
import pandas as pd
from django.core.paginator import Paginator
from django.http import JsonResponse, HttpResponse
from django.template.loader import render_to_string

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import numpy as np
import seaborn as sns
import datetime

def index(request):
    form = FileUploadForm()
    uploaded_files = UploadedFile.objects.all()
    return render(request, 'analysisCSV/index.html', {'form': form, 'uploaded_files': uploaded_files})

def upload_file(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        print(form)
        try:
            if form.is_valid():
                form.save()
                return redirect('upload_success')
        except:
            return render(request, 'error.html')
        
    else:
        return render(request, 'error.html')
        

def upload_success(request):
    return render(request, 'analysisCSV/upload_success.html')

def analyze_csv(request, file_id):
    uploaded_file = UploadedFile.objects.get(pk=file_id)
    df = pd.read_csv(uploaded_file.file)
    explore_data = df.describe()
    explore_html = explore_data.to_html()

    data_preview_html = df.head(10).to_html(index=False)
    
    df_lines = "{:,}".format(len(df))
    paginator = Paginator(df, 10)
    page_number = request.GET.get('page')
    data_all_rows = paginator.get_page(page_number)
    
    columns = df.columns
    
    if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
        return JsonResponse({'table_html': render_to_string('analysisCSV/table.html', {'data_all_rows': data_all_rows})})
    
    return render(request, 'analysisCSV/analyze_csv.html', {'uploaded_file': uploaded_file, 'explore_html': explore_html, 'data_preview_html': data_preview_html, 'df_lines': df_lines, 'data_all_rows': data_all_rows, 'columns': columns, 'file_id': file_id})

def analysis_model(request, file_id):
    if request.method == 'POST':
        target_variable = request.POST.get('target')
        input_columns = request.POST.getlist('input')
        model = request.POST.get('model')
        file_id = request.POST.get('file_id')
        type_clean = request.POST.get('type-clean')
        
        try:
            result = train_model(model, file_id, target_variable, input_columns, type_clean)
        except Exception as e:
            return JsonResponse({'error': str(e)})
        if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
            return JsonResponse({'result_form': render_to_string('analysisCSV/analysis_model.html', {"result": result})})
        
    else:
        return render(request, 'analysisCSV/index.html')

def train_model(model_type, file_id, target, selected_variables, type_clean):
    uploaded_file = UploadedFile.objects.get(pk=file_id)
    df = pd.read_csv(uploaded_file.file)
    # df = df.dropna(subset=selected_variables + [target])  # Ensure all selected variables and target have no NaN
    if type_clean == 'delete':
        df = df.dropna(subset=selected_variables + [target])
    elif type_clean == 'replace':
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
            
            # Thay thế các giá trị thiếu trong các cột số bằng giá trị trung bình
        imputer_numeric = SimpleImputer(strategy='mean')
        df[numeric_cols] = imputer_numeric.fit_transform(df[numeric_cols])
            
            # Thay thế các giá trị thiếu trong các cột chuỗi bằng giá trị xuất hiện nhiều nhất
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])
    # df = (df - df.min()) / (df.max() - df.min())

    categories = [variable for variable in selected_variables if df[variable].dtype == "object"]
    numericals = [variable for variable in selected_variables if variable not in categories]

    # Impute categorical variables
    imputer = SimpleImputer(strategy='most_frequent')
    df[categories] = imputer.fit_transform(df[categories])
    
    # Encode categorical variables
    for category in categories:
        le = LabelEncoder()
        df[category] = le.fit_transform(df[category])

    # Impute and scale numerical variables
    df[numericals] = df[numericals].fillna(df[numericals].mean())
    scaler = StandardScaler()
    df[numericals] = scaler.fit_transform(df[numericals])

    X = df[selected_variables]
    y = df[target]

    # Encode target if it's categorical
    if df[target].dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(df[target])

    print(f"Shapes - X: {X.shape}, y: {y.shape}")

    # Check if the dimensions match
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Inconsistent numbers of samples: X has {X.shape[0]} samples, y has {y.shape[0]} samples.")

    # Split the data
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=27)

    
    # Implement model based on type
    results = {}

    if model_type == "linear":
        model = LinearRegression()
        model.fit(trainX, trainy)

        # Đánh giá mô hình
        y_pred = model.predict(testX)
        rmse = mean_squared_error(testy, y_pred)
        # rmse = np.sqrt(mean_squared_error(testy, y_pred))
        accuracy = model.score(testX, testy)
        # Lưu kết quả
        results['equation'] = str(model.coef_)  
        results['accuracy'] = f"{accuracy:.2f}"

        # Vẽ biểu đồ
        plt.scatter(testy, y_pred, color='blue')
        plt.plot(testy, testy, color='red', linewidth=1)  # Đường thẳng y = x
        plt.title('Linear Regression')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        random_file_name = uploaded_file.file.name.split("/")[-1].split(".")[0] + "-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        plt_path = f"static/img/{random_file_name}"
        plt.savefig(plt_path)
        plt.close()

        results['visualization'] = random_file_name       

        return results
    if model_type == "logistic":
        clf = OneVsRestClassifier(LogisticRegression())
        clf.fit(trainX, trainy)
        y_scores = clf.predict_proba(testX)
        fpr, tpr, _ = roc_curve(testy, y_scores[:, 1], pos_label=1)  
        roc_auc = auc(fpr, tpr)
        # auc_score = roc_auc_score(testy, y_scores[:, 1])
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="best")
        random_file_name = uploaded_file.file.name.split("/")[-1].split(".")[0] + "-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        plt_path = f"static/img/{random_file_name}"
        
        # Lưu hình ảnh
        plt.savefig(plt_path, dpi=300)
        plt.close()

        results = {}
        results['visualization'] = random_file_name
        results['equation'] = None
        results['accuracy'] = f"{roc_auc:.2f}"

        return results

    if model_type == "knn":
        k = 5
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(trainX, trainy)
        y_pred = knn.predict(testX)
        accuracy = accuracy_score(testy, y_pred)
        
        # Tính toán confusion matrix
        cm = confusion_matrix(testy, y_pred)

        # Vẽ biểu đồ Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        random_file_name = uploaded_file.file.name.split("/")[-1].split(".")[0] + "-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        plt_path = f"static/img/{random_file_name}"
        plt.savefig(plt_path)
        plt.close()

        results = {}
        results['equation'] = None
        results['accuracy'] = f"{accuracy:.2f}"
        results['visualization'] = random_file_name

        return results
    # Decision Tree
    if model_type == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        from sklearn import tree
        dt = DecisionTreeClassifier()
        dt.fit(trainX, trainy)
        y_pred = dt.predict(testX)
        accuracy = accuracy_score(testy, y_pred)
        # Plot Decision Tree
        plt.figure(figsize=(16, 20))
        tree.plot_tree(dt, filled=True, feature_names=X.columns, class_names=np.unique(y).astype(str), fontsize=6)
        random_file_name = uploaded_file.file.name.split("/")[-1].split(".")[0] + "-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        plt_path = f"static/img/{random_file_name}"
        plt.savefig(plt_path)
        plt.close()
        results = {}
        results['equation'] = None
        results['accuracy'] = f"{accuracy:.2f}"
        results['visualization'] = random_file_name
        return results
    
    # Random Forest
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn import tree
        
        rf = RandomForestClassifier()
        rf.fit(trainX, trainy)
        y_pred = rf.predict(testX)
        accuracy = accuracy_score(testy, y_pred)
        # Plot Decision Tree
        plt.figure(figsize=(16, 20))
        tree.plot_tree(rf.estimators_[0], filled=True, feature_names=X.columns, class_names=np.unique(y).astype(str), fontsize=6)
        random_file_name = uploaded_file.file.name.split("/")[-1].split(".")[0] + "-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        plt_path = f"static/img/{random_file_name}"
        plt.savefig(plt_path)
        plt.close()
        results = {}
        results['equation'] = None
        results['accuracy'] = f"{accuracy:.2f}"
        results['visualization'] = random_file_name
        return results


def serve_file(request, file_name):
    file_path = f'media/uploads/{file_name}'
    with open(file_path, 'rb') as file:
        response = HttpResponse(file.read(), content_type="application/csv")
        response['Content-Disposition'] = f'attachment; filename={file_name}'
        return response

def view_file(request, file_name):
    try:
        file_path = f'media/uploads/{file_name}'
        with open(file_path, 'r') as file:
            response = HttpResponse(file.read(), content_type="text/txt")
            return response
    except:
        return HttpResponse("File not found")

def delete_file(request, file_id):
    uploaded_file = UploadedFile.objects.get(pk=file_id)
    uploaded_file.delete()
    return redirect('index_csv')

def load_image(request, file_name):
    try:
        with open(f"static/img/{file_name}", 'rb') as file:
            response = HttpResponse(file.read(), content_type="image/png")
            return response
    except:
        return HttpResponse("File Not Found")
    
def normalize_data(request, file_id):
    uploaded_file = UploadedFile.objects.get(pk=file_id)
    df = pd.read_csv(uploaded_file.file)
    df = df.dropna()
    df = (df - df.min()) / (df.max() - df.min())
    df.to_csv(uploaded_file.file.name, index=False)
    return redirect('analyze_csv', file_id=file_id)

def cleanData(request, file_id):
    type_clean = request.POST.get('type_clean')
    if type_clean == 'delete':
        uploaded_file = UploadedFile.objects.get(pk=file_id)
        df = pd.read_csv(uploaded_file.file)
        df = df.dropna()
        df.to_csv(uploaded_file.file.name, index=False)
    elif type_clean == 'replace':
        uploaded_file = UploadedFile.objects.get(pk=file_id)
        df = pd.read_csv(uploaded_file.file)
        df = df.fillna(0)
        df.to_csv(uploaded_file.file.name, index=False)
    return redirect('analyze_csv', file_id=file_id)