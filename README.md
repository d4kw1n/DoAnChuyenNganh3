# machine-final-web

 Training a Machine Learning Model to classify orchids & Analysis CSV deployed on the Web Platform

## Run

- Create a new virtual enviroment: **python -m venv <Virtual Env Name>**
  
```bash
python -m venv venv
```

- To access this virtual enviroment
  - Window: **<Virtual Env Name>/Scripts/activate** . Ex: **venv/Scripts/activate**
  - Mac/Linux: **source <Virtual Env Name>/bin/active** . Ex: **source venv/bin/active**

- Download all libraries on venv
  
```bash
pip install -r requirements.txt
```

- To deacactive venv: **deactivate**
- To migrate:

```bash
python manage.py makemigrations <App name>
python manage.py migrate
```

- To run this project, please run the commands below

```bash
pip install -r requirements.txt

python manage.py runserver 8080
```

- Then, please visit [http://127.0.0.1:8080](http://127.0.0.1:8080) to access website

## List Dataset Flowers

- Hoa tulip
- Hoa sen
- Hoa loa kèn
- Cây dâm bụt
- Hoa giấy
- Hoa hướng dương
- Bồ công anh
- Lantana (Ngũ sắc)
- Hoa hồng
- Hoa mai
- Phong lan
- Hoa cúc
