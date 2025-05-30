def train_model(model, file_id, target, selected_variables):
    uploaded_file = UploadedFile.objects.get(pk=file_id)
    df = pd.read_csv(uploaded_file.file)
    
    categories = []
    nums = []
    for variable in selected_variables:
        if df[variable].dtypes == "object":
            categories.append(variable)
        else:
            nums.append(variable)
    df_numericals = df[nums]
    imputer = SimpleImputer(strategy='most_frequent')
    df_categories = df[categories]
    df_categories = pd.DataFrame(imputer.fit_transform(df_categories), columns=df_categories.columns)
    for cate in categories:
        lb_encoder = LabelEncoder()
        df_categories[cate] = lb_encoder.fit_transform(df[cate])
    
    for num in nums:
        mean = df_numericals[num].mean()
        df_numericals.fillna(mean, inplace=True) 
    X = pd.concat([df_numericals, df_categories], axis=1)

    if df[target].dtypes == "object":
        lb_encoder = LabelEncoder()
        y = lb_encoder.fit_transform(df[target])
    else:
        y = df[target]


    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
    scaler = StandardScaler()
    trainX = scaler.fit_transform(trainX)
    testX = scaler.transform(testX)

    results = {}

    if model == "linear":
        model = LinearRegression()
        model.fit(trainX, trainy)
        y_pred = model.predict(testX)
        rmse = mean_squared_error(testy, y_pred, squared=False)
        results['equation'] = str(model.coef_)  
        results['accuracy'] = f"{rmse:.2f}"  
        plt.scatter(testy, y_pred, color='blue')
        plt.title('Linear Regression')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        random_file_name = uploaded_file.file.name.split("/")[-1].split(".")[0] + "-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        plt_path = f"static/img/{random_file_name}"
        plt.savefig(plt_path)
        plt.close()
        results['visualization'] = random_file_name       
        return results

    if model == "logistic":
        clf = OneVsRestClassifier(LogisticRegression())
        ns_probs = [0 for _ in range(len(testy))]
        clf.fit(trainX, trainy)
        pred = clf.predict(testX)
        # y_scores = clf.predict_proba(testX)
        lr_probs = clf.predict_proba(testX)
        lr_probs = lr_probs[:, 1]
        ns_auc = roc_auc_score(testy, ns_probs)
        lr_auc = roc_auc_score(testy, lr_probs)
        
        ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
        # fpr, tpr, _ = roc_curve(testy, y_scores[:, 1], pos_label=1)  
        # roc_auc = auc(fpr, tpr)
        # plt.figure()
        # lw = 2
        # plt.plot(fpr, tpr, color='darkorange',
        #         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc="best")
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        random_file_name = uploaded_file.file.name.split("/")[-1].split(".")[0] + "-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        plt_path = f"static/img/{random_file_name}"
        
        # Lưu hình ảnh
        plt.savefig(plt_path)
        plt.close()

        results = {}
        results['visualization'] = random_file_name

        return results

    if model == "knn":
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