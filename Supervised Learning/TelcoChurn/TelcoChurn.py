##############################################################
# Telco Churn
##############################################################
##############################################################
# İş Problemi
##############################################################

#Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi
# beklenmektedir.
###############################################################
# Veri Seti Hikayesi
###############################################################

# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve
# İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir.
# Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.

#CustomerId : Müşteri İd
#Gender: Cinsiyet
# SeniorCitizen: Müşterinin yaşlı olup olmadığı (1, 0)
# Partner: Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
# Dependents: Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır
# tenure: Müşterinin şirkette kaldığı ay sayısı
# PhoneService: Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines: Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService: Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity: Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup: Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection: Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport: Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV: Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingMovies: Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Contract: Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling: Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod: Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges: Müşteriden tahsil edilen toplam tutar

# Churn — Whether the customer churned or not (Yes or No)

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import math
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split, cross_validate

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

###############################################################
# GÖREV-1: Keşifçi Veri Analizi
###############################################################
df_telco = pd.read_csv('TelcoChurn/Telco-Customer-Churn.csv')
df_telco.head()
df_telco.shape
df_telco.info()
df_telco.isnull().sum()
df_telco.describe().T
# Tüm sütun adlarını büyük harfe dönüştür
df_telco.columns = df_telco.columns.str.upper()

#Services Each Customer Has to Sign Up

# PhoneService — Whether the customer has a phone service or not (Yes or No)
# MultipleLines — Whether the customer has multiple lines or not (Yes, No, No phone service)
# InternetService — A type of internet service the customer has (DSL, Fiber Optic, No)
# OnlineSecurity — Whether the customer has online security or not (Yes, No, No Internet Service)
# OnlineBackup — Whether the customer has online backup or not (Yes, No, No Internet Service)
#DeviceProtection — Whether the customer has device protection or not (Yes, No, No Internet Service)
#TechSupport — Whether the customer has tech support or not (Yes, No, No Internet Service)
#StreamingTV—Whether the customer has a streaming TV (Yes, No, No Internet Service)
#StreamingMovies — Whether the customer has a streaming movie (Yes, No, No Internet Service)

#Customer Account Information

#Tenure — How long customer has stayed in the company
#Contract — The type of contract the customer has (Month-to-Month, One year, Two years)
#PaperlessBilling — Whether the customer has a paperless billing (Yes, No)
#PaymentMethod — payment method used by the customer (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
#MonthlyCharges — Amount charged to the customer monthly
#TotalCharges — The total amount charged to the customer

#Customer Demographic Info

#CustomerID — Unique value for each customer
#gender — The type of gender each customer (Female, Male)
#SeniorCitizen — Whether the customer is a senior citizen (Yes, No)
#Partner — Whether the customer has a partner or not (Yes, No)
#Dependents — Whether the customer has a dependent or not (Yes, No)

# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
#Bu kolon float olması gerekirken object kalmış
df_telco['TOTALCHARGES'] = pd.to_numeric(df_telco['TOTALCHARGES'], errors='coerce')


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
        Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f"  -> İsimleri: {cat_cols}")
    print(f'num_cols: {len(num_cols)}')
    print(f"  -> İsimleri: {num_cols}")
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f"  -> İsimleri: {cat_but_car}")
    print(f'num_but_cat: {len(num_but_cat)}')
    print(f"  -> İsimleri: {num_but_cat}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df_telco)

#Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
for column in cat_cols:
    print(df_telco[column].value_counts()) #TotalCharges kolonu bir üstte değiştirildi.

#Payment kolonunda automatic ifades uzun kalıyor.
df_telco["PAYMENTMETHOD"].unique()
df_telco["PAYMENTMETHOD"] = df_telco["PAYMENTMETHOD"].str.replace(" (automatic)", "", regex=False)


#Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz
# Sayısal değişkenler için histogramları çiz
plt.figure(figsize=(15, 5))
for i, col in enumerate(num_cols):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df_telco[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show() # Grafiği göstermek için


def histogram_plots(df, numerical_values, target):
    number_of_columns = 2
    number_of_rows = math.ceil(len(numerical_values) / 2)

    fig = plt.figure(figsize=(12, 5 * number_of_rows))

    for index, column in enumerate(numerical_values, 1):
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)
        ax = sns.kdeplot(df[column][df[target] == "Yes"], fill=True)
        ax = sns.kdeplot(df[column][df[target] == "No"], fill=True)
        ax.set_title(column)
        ax.legend(["Churn", "No Churn"], loc='upper right')
    plt.savefig("numerical_variables.png", dpi=300)
    return plt.show()

histogram_plots(df_telco,num_cols, "CHURN")


# Kategorik değişkenler için çubuk grafikleri çiz
n_rows = (len(cat_cols) + 2) // 3 # Satır sayısını ayarla
plt.figure(figsize=(18, 5 * n_rows))
for i, col in enumerate(cat_cols):
    plt.subplot(n_rows, 3, i + 1)
    # Hata uyarısını gidermek için 'hue' parametresini ekle ve 'legend=False' yap
    sns.countplot(data=df_telco, x=col, hue=col, palette='viridis', legend=False)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right') # Etiketleri daha iyi okunabilirlik için döndür
plt.tight_layout()
plt.show()

#NOT:--> No internet service durumundan dolayı 6 adet kolon da 1500 kişi için No olmuş. Ne yapılabilinir?
#PhoneService ve SeniorCitizen değişkenlerinde bir imbalanced var denilebilir.

#Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
def plot_categorical_to_target(df, categorical_values, target):
    number_of_columns = 3
    number_of_rows = math.ceil(len(categorical_values) / 3)

    fig = plt.figure(figsize=(20, 5 * number_of_rows))

    for index, column in enumerate(categorical_values, 1):
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)
        ax = sns.countplot(x=column, data=df, hue=target, palette="Blues")
        ax.set_title(column)
    return plt.show()

plot_categorical_to_target(df_telco,cat_cols, "CHURN")

#Adım 5: Aykırı gözlem var mı?

df_telco.describe().T
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df_telco, col))

#Adım 6: Eksik gözlem var mı inceleyiniz.
df_telco.isnull().sum()
df_telco[df_telco['TOTALCHARGES'].isnull()] # Bu değerler object formatında No iken float formata çevirince NaN hale gelmiş. Bunların hepsi Tenure= o olan yeni müşteriler

###############################################################
# GÖREV-2: Feature Engineering
###############################################################
#Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
#Aykırı gözlem yoktu--> TotalCharges değerlerinde yer alan NaN değer 0 olarak değiştirildi.
df_telco.fillna(0, inplace=True) #Bir önceki satırdaki tespite göre değerler verilir.

#Adım 2: Yeni değişkenler oluşturunuz

#TotalServiceCount (Toplam Hizmet Sayısı):# Services Each Customer Has to Sign Up kolonlarında
# kullanılan servislerin sayıları çıkartılabilir.

# Hizmet sütunlarını tanımla
service_columns = [
    'PHONESERVICE', 'MULTIPLELINES', 'INTERNETSERVICE', 'ONLINESECURITY',
    'ONLINEBACKUP', 'DEVICEPROTECTION', 'TECHSUPPORT', 'STREAMINGTV', 'STREAMINGMOVIES'
]

# Her müşteri için toplam hizmet sayısını hesapla
df_telco['TOTAL_SERVICES_COUNT'] = 0

for col in service_columns:
    if col in ['PHONESERVICE', 'MULTIPLELINES','ONLINESECURITY','ONLINEBACKUP','ONLINEBACKUP','DEVICEPROTECTION','TECHSUPPORT','STREAMINGTV', 'STREAMINGMOVIES']:
        # 'Yes' olanları say
        df_telco['TOTAL_SERVICES_COUNT'] += (df_telco[col] == 'Yes').astype(int)
    elif col == 'INTERNETSERVICE':
        # 'No' olmayanları say (yani 'DSL' veya 'Fiber optic')
        df_telco['TOTAL_SERVICES_COUNT'] += (df_telco[col] != 'No').astype(int)

#HasInternetService (İnternet Hizmeti Var Mı?): InternetService sütununda 'No' dışında bir değer olan müşteriler için 1,
# aksi takdirde 0 değeri alacak bir ikili değişken. İnternet hizmetinin churn üzerindeki etkisi büyük olabilir.

df_telco['HASINTERNETSERVICE'] = df_telco['INTERNETSERVICE'].apply(lambda x: 1 if x != 'No' else 0)


#Başka Düşünülebilcek Yeni Değişkenler:
    #AverageMonthlyCharge (Ortalama Aylık Ücret): TotalCharges ve Tenure sütunlarını kullanarak 
    # müşterinin ortalama aylık ödediği tutarı hesaplayabiliriz. Bu, MonthlyCharges'dan farklı olarak, 
    # müşterinin tüm abonelik süresi boyunca ödediği ortalama maliyeti yansıtır.

    #IsMonthToMonthContract (Aylık Sözleşmeli Mi?): Contract sütunundan türetilebilir. 'Month-to-month'
    #   ise 1, diğer sözleşme türleri için 0 değeri alacak bir ikili değişken. Genellikle aylık sözleşmeli
    #   müşterilerin churn etme olasılığı daha yüksektir.

#Adım 3: Encoding işlemlerini gerçekleştiriniz.

# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_names(df_telco,5,20)

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df_telco.columns if df_telco[col].dtypes == "O" and df_telco[col].nunique() == 2]

for col in binary_cols:
    df_telco = label_encoder(df_telco, col)

# One-Hot Encoding İşlemi
# cat_cols listesinin güncelleme işlemi
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["CHURN"]]

cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df_telco = one_hot_encoder(df_telco, cat_cols, drop_first=True)

df_telco.info()
df_telco.shape

#Adım 4: Numerik değişkenler için standartlaştırma yapınız.
num_cols
scaler = StandardScaler()
df_telco[num_cols] = scaler.fit_transform(df_telco[num_cols])

df_telco.head()
df_telco.shape

###############################################################
# GÖREV-3: Modelleme
###############################################################

#Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
#X--> CUSTOMERID ve CHURN yok

#Correlation btw the target and features
df_telco.drop('CUSTOMERID', axis=1, inplace=True)
plt.figure(figsize=(10,6))
df_telco.corr()["CHURN"].sort_values(ascending=False).plot(kind="bar")
plt.savefig("correlation.png", dpi=300)
plt.show()

#Train and test split
X = df_telco.drop(columns = "CHURN")
y = df_telco["CHURN"]

#stratify y diyere y'deki 0-1 değerleri dengeli olarak dağılır.
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#Confusion Matrix Fonksiyonu
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

#ROC graph and calculation
def plot_roc_curve(estimator, X_test, y_test, title='ROC Curve', color='r--', figsize=(8, 6)):
    """
    Plots the ROC Curve for a given classifier and test data.

    Parameters:
    - estimator: trained model object with predict_proba or decision_function
    - X_test: features of the test set
    - y_test: true labels of the test set
    - title: title of the plot (default: 'ROC Curve')
    - color: line style for the baseline (default: 'r--')
    - figsize: size of the plot (default: (8, 6))
    """
    #plt.figure(figsize=figsize)
    RocCurveDisplay.from_estimator(estimator=estimator, X=X_test, y=y_test)
    plt.plot([0, 1], [0, 1], color, label='Random Classifier')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()



# Logistic Regression
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression().fit(X_train,y_train)
y_pred_logreg = log_model.predict(X_test)
y_pred_logreg_proba = log_model.predict_proba(X_test)[:, 1]


plot_confusion_matrix(y_test, y_pred_logreg)
print(classification_report(y_test, y_pred_logreg)) # scikit learn library'den

# Accuracy --> 0.81
# Precision Pozitif sınıf (1) tahminlerinin basari oranıdır: TP / (TP+FP) --> 0.68
# Recall Pozitif sınıf (1) tahminlerinin basari oranıdır: TP / (TP+FP) --> 0.57

#ROC
plot_roc_curve(log_model, X_test, y_test)

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_pred_knn_proba = knn.predict_proba(X_test)[:,1]

plot_confusion_matrix(y_test, y_pred_knn)
print(classification_report(y_test, y_pred_knn)) # scikit learn library'den

# Accuracy --> 0.77
# Precision Pozitif sınıf (1) tahminlerinin basari oranıdır: TP / (TP+FP) --> 0.58
# Recall Pozitif sınıf (1) tahminlerinin basari oranıdır: TP / (TP+FP) --> 0.54

#ROC --> 0.8
plot_roc_curve(knn, X_test, y_test)

# Decision Tree

# SVM


#Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile modeli tekrar kurunuz.

# KNN için hiperparametre grid'ini tanımla
knn_params = {"n_neighbors": range(2, 40)}

knn_gs_best = GridSearchCV(knn, knn_params, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
knn_gs_best.fit(X_train, y_train)

knn_gs_best.best_params_

#Final Modeli
knn_final = knn.set_params(**knn_gs_best.best_params_).fit(X, y)
cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean() #0.798
cv_results['test_f1'].mean() #0,5916
cv_results['test_roc_auc'].mean() #0.834
