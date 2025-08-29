##############################################################
# Feature Engineering
##############################################################
##############################################################
# İş Problemi
##############################################################

#Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir
# makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli olan veri
# analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır.
# ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima
# Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir.
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz konsantrasyonu
# BloodPressure: Kan basıncı (mm Hg)
# SkinThickness: Cilt kalınlığı (mm)
# Insulin: İnsülin seviyesi (mu U/ml)
# BMI: Vücut Kitle İndeksi (kg/m²)
# DiabetesPedigreeFunction: Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
# Age: Yaş (yıl)
# Outcome: Sonuç (hedef değişken). 1 diyabetli, 0 diyabetsiz anlamına gelir.

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

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)



###############################################################
# GÖREV-1: Keşifçi Veri Analizi
###############################################################
#Adım 1: Genel resmi inceleyiniz.
df_diabetes = pd.read_csv('diabetes/diabetes.csv')
df_diabetes.shape
df_diabetes.info()
df_diabetes.isnull().sum()

df_diabetes.head()
df_diabetes["Outcome"].value_counts()
#Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=10):
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

cat_cols, num_cols, cat_but_car = grab_col_names(df_diabetes)

df_diabetes.head()

#Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.

# NUMERİK DEĞİŞKEN ANALİZİ:
#Pregnancies--> Max=17?
#Glucose?
print(df_diabetes[num_cols].describe().T.round(2))
# Box Plotlar (Aykırı Değerler)
plt.figure(figsize=(18, 10))
sns.boxplot(data=df_diabetes[num_cols], orient='h')
plt.title('Numerik Değişkenler için Aykırı Değer Analizi (Box Plot)', fontsize=16)
plt.show()

# HEDEF DEĞİŞKEN ANALİZİ
print(df_diabetes["Outcome"].value_counts())

plt.figure(figsize=(8, 5))
sns.countplot(x="Outcome", data=df_diabetes)
plt.title(f'{"Outcome"} Dağılımı', fontsize=16)
plt.show()

#Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
summary_stats = df_diabetes.groupby("Outcome")[num_cols].agg(['mean', 'median', 'std']).round(2)
print(summary_stats)

# Her bir numerik sütun için ayrı bir violin plot çiz
for col in num_cols:
    plt.figure(figsize=(10, 7))
    sns.violinplot(x="Outcome", y=col, data=df_diabetes, inner='quartile')
    # inner='quartile' kutu grafiğindeki çeyrekleri gösterir

    plt.title(f"'{col}' Dağılımının '{"Outcome"}'a Göre Karşılaştırılması", fontsize=16)
    plt.xlabel("Outcome (0: Diyabet Değil, 1: Diyabetli)", fontsize=12)
    plt.ylabel(f"{col} Değeri", fontsize=12)
    plt.show()

#Adım 5: Aykırı gözlem analizi yapınız.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
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

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

outlier_thresholds(df_diabetes, "Age")
grab_outliers(df_diabetes, "Age")

for col in num_cols:
    print(col, check_outlier(df_diabetes, col))

# Adım 6: Eksik gözlem analizi yapınız.
# eksik gozlem var mı yok mu sorgusu
df_diabetes.isnull().values.any()
# degiskenlerdeki eksik deger sayisi
df_diabetes.isnull().sum() #Eksik değerimiz yok

# Adım 7: Korelasyon analizi yapınız.
correlation_matrix = df_diabetes.corr()
plt.figure(figsize=(14, 12))
# Seaborn kullanarak ısı haritasını oluştur
# cmap='coolwarm' pozitif korelasyonları sıcak (kırmızı), negatifleri soğuk (mavi) renklerle gösterir.
# annot=True her bir hücreye korelasyon katsayısını yazar.
# fmt='.2f' sayıları iki ondalık basamakla formatlar.
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Değişkenler Arası Korelasyon Isı Haritası', fontsize=20)
plt.show()

###############################################################
# GÖREV-1: Feature Engineering
###############################################################
#Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem
# bulunmamakta ama Glikoz, Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksik
# değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır.
# Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında
# eksik değerlere işlemleri uygulayabilirsiniz.

# Convert 0 values to nan Values
cols_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print((df_diabetes[cols_to_clean] == 0).sum())
df_diabetes[cols_to_clean] = df_diabetes[cols_to_clean].replace(0, np.nan)
df_diabetes.isnull().sum()

# Aykırı değerler problemini çözme--> Baskılama yöntemi
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # Sütunun orijinal veri tipini al
    original_dtype = dataframe[variable].dtype

    # Eşik değerlerini bu orijinal tipe dönüştür.
    # Bu, hatayı gideren anahtar adımdır.
    low_limit = original_dtype.type(low_limit)
    up_limit = original_dtype.type(up_limit)

    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df_diabetes, col))

for col in num_cols:
    replace_with_thresholds(df_diabetes, col)

for col in num_cols:
    print(col, check_outlier(df_diabetes, col))


# Null değerleri sorunu çözümü
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    # Eksik değer sayısı ve oranını tek bir DataFrame'de birleştirir.
    # 'keys' parametresi sütun isimlerini belirler ('n_miss', 'ratio').
    # 'np.round(ratio, 2)' oranları 2 ondalık basamağa yuvarlar.
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    # Eğer 'na_name' True ise, eksik değer içeren sütunların listesini döndürür.
    # Bu, fonksiyonun hem tabloyu yazdırmasını hem de sütun listesini geri vermesini sağlar.
    if na_name:
        return na_columns

missing_values_table(df_diabetes)

na_columns = [col for col in df_diabetes.columns if df_diabetes[col].isnull().sum() > 0]

# Eksiklik bayrağı ile hedef değişken arasındaki ilişkiyi inceleyen fonksiyon.
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    # Her bir eksik sütun için 'NA_FLAG' (Eksiklik Bayrağı) adında yeni bir ikili (binary)
    # değişken oluşturur. Eğer sütunda değer eksikse 1, değilse 0 atanır.
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    # Yeni oluşturulan tüm 'NA_FLAG' sütunlarını seçer.
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    # Her bir 'NA_FLAG' sütunu için döngü yapar.
    for col in na_flags:
        # İlgili 'NA_FLAG' sütununa göre gruplandırma yapar (yani eksik olanlar ve olmayanlar).
        # Her gruptaki hedef değişkenin (target) ortalamasını ve gözlem sayısını hesaplar.
        # Bu, eksik olmanın hedef değişkenin ortalamasını etkileyip etkilemediğini gösterir.
        # Eğer eksik olan grubun hedef ortalaması, eksik olmayan grubun hedef ortalamasından belirgin şekilde farklıysa,
        # eksiklik rastgele değildir (MAR veya MNAR) ve bu bilgi modellemede kullanılmalıdır.
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df_diabetes, "Outcome", na_columns)

#Median kullanılacaklar: Glucose, BMI, BloodPressure

cols_median =["Glucose", "BMI", "BloodPressure"]
for col in cols_median:
    df_diabetes[col].fillna(df_diabetes[col].median(), inplace=True)
    
#KNN ÖNCESİ EKSİK DEĞER DURUMU
df_diabetes.isnull().sum()

# Veriyi Ölçeklendirme ---
scaler = MinMaxScaler()

# Veriyi ölçeklendir (sonuç bir numpy array döner)
# Sütun isimlerini kaybetmemek için geçici bir DataFrame'e çeviriyoruz
scaled_data = scaler.fit_transform(df_diabetes)
df_scaled = pd.DataFrame(scaled_data, columns=df_diabetes.columns)

from sklearn.impute import KNNImputer
# KNNImputer nesnesini oluşturalım (en yakın 5 komşuya bakacak)
# n_neighbors, modelin performansını etkileyen bir hiperparametredir. 5 iyi bir başlangıçtır.
imputer = KNNImputer(n_neighbors=5)

# Ölçeklenmiş veri üzerindeki eksik değerleri doldur
imputed_scaled_data = imputer.fit_transform(df_scaled)
df_imputed_scaled = pd.DataFrame(imputed_scaled_data, columns=df_diabetes.columns)

# scaler'ın 'inverse_transform' metodunu kullanarak veriyi eski haline getir
original_scale_data = scaler.inverse_transform(df_imputed_scaled)
df_final = pd.DataFrame(original_scale_data, columns=df_diabetes.columns)

df_final.isnull().sum()

df_final.describe().T
#Adım 2: Yeni değişkenler oluşturunuz.

df_final['Is_High_BloodPressure'] = np.where(df_final['BloodPressure'] > 90, 1, 0)
df_final.groupby("Is_High_BloodPressure")["Outcome"].count()

# Glikoz Seviyesini Kategorize Etme
# Klinik olarak anlamlı glikoz eşik değerlerini kullanalım.
glucose_bins = [0, 99, 125, 300] # Normal, Prediyabet, Diyabet
glucose_labels = ['Normal_Glucose', 'Prediabetes', 'Diabetic_Glucose']
df_final['Glucose_Category'] = pd.cut(x=df_final['Glucose'], bins=glucose_bins, labels=glucose_labels)

# Yaşı Kategorize Etme
# Yaş grupları oluşturarak modelin doğrusal olmayan yaş etkilerini yakalamasına yardımcı olalım.
age_bins = [20, 30, 45, 65]
age_labels = ['Young_Adult', 'Middle_Aged', 'Senior']
df_final['Age_Category'] = pd.cut(x=df_final['Age'], bins=age_bins, labels=age_labels, right=False)

df_final.head()

#Adım 3: Encoding işlemlerini gerçekleştiriniz.
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    # Verilen DataFrame ve kategorik sütun listesini One-Hot Encoding'e tabi tutar.
    # 'drop_first' parametresi dışarıdan kontrol edilebilir.
    #dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

ohe_cols = [col for col in df_final.columns if 10 >= df_final[col].nunique() > 2]

df = one_hot_encoder(df_final, ohe_cols)

df.head()
df.shape
"""
#Yüzde birin altında olan kolonları sırala--> yokmuş
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
"""


#Adım 4: Numerik değişkenler için standartlaştırma yapınız.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape


# Adım 5: Model oluşturunuz.
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

#############################################
# Hiç bir işlem yapılmadan elde edilecek skor?
#############################################
y = df_diabetes["Outcome"]
X = df_diabetes.drop([ "Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# Yeni ürettiğimiz değişkenlerin etkileri

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)