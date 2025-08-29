################################################
################################################
## Scoutium- Makine Öğrenmesi ile Yetenek Avcılığı Sınıflandırma
################################################
################################################

# Business Problem:

# Bir futbol oyuncusunun yeteneklerini ve potansiyelini değerlendirmek için makine learning
# tekniklerini kullanarak oyuncu sınıflandırması yapmak.

#Veri Seti Hikayesi:

#Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre 
# scoutların değerlendirdikleri futbolcuların, maç içerisinde puanlanan 
# özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.

################################################
# 1. Import Libraries
################################################
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 2. EDA (Exploratory Data Analysis)
################################################

attributes_df = pd.read_csv("scoutium_attributes.csv", sep=";")
labels_df = pd.read_csv("scoutium_potential_labels.csv", sep=";")

attributes_df.head()
labels_df.head()
labels_df.shape

# Adım 2: CSV dosyalarını birleştirme
df = pd.merge(
    attributes_df,
    labels_df,
    on=["task_response_id", "match_id", "evaluator_id", "player_id"],
    how="inner"
)

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    #print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    numeric_df = dataframe.select_dtypes(include='number')
    print(numeric_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


#df["player_id"].nunique()  # Pozisyonların benzersiz sayısı

# Birleştirilmiş veri setinin ilk 5 satırını göster
print(df.head())
df.shape 

# Adım 3: Kaleci (position_id == 1) olan satırları kaldır
df = df[df["position_id"] != 1] # 700 değer gider

#Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.
# ( below_average sınıfı tüm verisetinin %1'ini oluşturur)
df["potential_label"].value_counts()
df = df[df["potential_label"] != "below_average"]

# Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir 
# tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu olacak şekilde 
# manipülasyon yapınız.

# Adım A: Indekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” 
# ve değerlerde scout’ların oyunculara verdiği puan “attribute_value” olacak şekilde pivot table’ı oluşturunuz.
pivot_df = df.pivot_table(
    index=["player_id", "position_id", "potential_label"],
    columns="attribute_id",
    values="attribute_value"
).reset_index()

print(pivot_df.head())

#Adım B: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve 
# “attribute_id” sütunlarının isimlerini stringe çeviriniz.
pivot_df = pivot_df.rename(columns={col: str(col) for col in pivot_df.columns})
print(pivot_df.head())

print(pivot_df.shape)

#Adım 6: LabelEncoder fonksiyonunu kullanarak “potential_label” kategorilerini 
# (average,highlighted) sayısal olarak ifade ediniz.
# potential_label kategorilerini sayısal olarak ifade et
#average: 0, highlighted: 1
le = LabelEncoder()
pivot_df["potential_label_encoded"] = le.fit_transform(pivot_df["potential_label"])

print(pivot_df[["potential_label", "potential_label_encoded"]].head(20))

pivot_df.info()
# Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
def grab_col_names(dataframe, cat_th=3, car_th=400):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

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

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

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
    num_cols = [col for col in num_cols if col not in num_but_cat and "id" not in col.lower()]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(pivot_df)

#Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki 
# veriyi ölçeklendirmek için StandardScaler uygulayınız.
X_scaled = StandardScaler().fit_transform(pivot_df[num_cols])
pivot_df[num_cols] = pd.DataFrame(X_scaled, columns=pivot_df[num_cols].columns)

pivot_df.head()

# Split X and y
y = pivot_df["potential_label_encoded"]
X = pivot_df[num_cols]

check_df(pivot_df)

#Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel 
# etiketlerini tahmin eden bir makine öğrenmesi modeli geliştiriniz. 
# (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)


#Base Models
def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


metrics = ["roc_auc", "f1", "precision", "recall", "accuracy"]

for metric in metrics:
    base_models(X, y, scoring=metric)


#pivot_df["potential_label_encoded"].value_counts()

"""
Base Models.... ROC AUC is a widely used metric for evaluating the performance of binary classifiers, especially for imbalanced datasets. It measures the ability of a classifier to distinguish between classes across all possible classification thresholds.
roc_auc: 0.8167 (LR) 
roc_auc: 0.7719 (KNN) 
roc_auc: 0.8618 (SVC) 
roc_auc: 0.7252 (CART) 
roc_auc: 0.8939 (RF) -----
roc_auc: 0.8258 (Adaboost) 
roc_auc: 0.8564 (GBM) 
roc_auc: 0.849 (XGBoost) 
roc_auc: 0.8774 (LightGBM) 
Base Models.... The F1-Score is the harmonic mean of Precision and Recall
f1: 0.5728 (LR) 
f1: 0.4784 (KNN) 
f1: 0.3998 (SVC) 
f1: 0.5138 (CART) 
f1: 0.5489 (RF) 
f1: 0.5903 (Adaboost) 
f1: 0.5917 (GBM) 
f1: 0.5966 (XGBoost) ----
f1: 0.5824 (LightGBM) 
Base Models....
precision: 0.7492 (LR) 
precision: 0.9583 (KNN) 
precision: 1.0 (SVC) ----
precision: 0.5474 (CART) 
precision: 0.7852 (RF) 
precision: 0.7036 (Adaboost) 
precision: 0.7132 (GBM) 
precision: 0.6838 (XGBoost) 
precision: 0.6996 (LightGBM) 
Base Models....
recall: 0.4815 (LR) 
recall: 0.3226 (KNN) 
recall: 0.2505 (SVC) 
recall: 0.5185 (CART) 
recall: 0.4464 (RF) 
recall: 0.5341 (Adaboost) 
recall: 0.5351 (GBM) -----
recall: 0.5341 (XGBoost) 
recall: 0.499 (LightGBM) 
Base Models....
accuracy: 0.856 (LR) 
accuracy: 0.8561 (KNN) 
accuracy: 0.845 (SVC) 
accuracy: 0.7972 (CART) 
accuracy: 0.8708 (RF) -----
accuracy: 0.8488 (Adaboost) 
accuracy: 0.8598 (GBM) 
accuracy: 0.8524 (XGBoost) 
accuracy: 0.8524 (LightGBM)
"""

# Adım 9 Devamı: hyperparameter optimization
knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}


classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

metrics = ["roc_auc", "f1", "precision", "recall", "accuracy"]

for metric in metrics:
    print(f"\nHyperparameter Optimization for {metric}:\n")
    best_models = hyperparameter_optimization(X, y, scoring=metric)

"""
Hyperparameter Optimization for roc_auc:

Hyperparameter Optimization....
########## KNN ##########
roc_auc (Before): 0.7719
roc_auc (After): 0.7719
KNN best params: {'n_neighbors': 5}

########## CART ##########
roc_auc (Before): 0.7243
roc_auc (After): 0.7236
CART best params: {'max_depth': 1, 'min_samples_split': 2}

########## RF ##########
roc_auc (Before): 0.8849
roc_auc (After): 0.8861
RF best params: {'max_depth': 8, 'max_features': 5, 'min_samples_split': 15, 'n_estimators': 300}

########## XGBoost ##########
roc_auc (Before): 0.849
roc_auc (After): 0.8361
XGBoost best params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100}

########## LightGBM ##########
roc_auc (Before): 0.8774
roc_auc (After): 0.837

Hyperparameter Optimization for f1:

Hyperparameter Optimization....
########## KNN ##########
f1 (Before): 0.4784
f1 (After): 0.4784
KNN best params: {'n_neighbors': 5}

########## CART ##########
f1 (Before): 0.5913
f1 (After): 0.5913
CART best params: {'max_depth': 1, 'min_samples_split': 2}

########## RF ##########
f1 (Before): 0.5778
f1 (After): 0.598
RF best params: {'max_depth': 15, 'max_features': 5, 'min_samples_split': 15, 'n_estimators': 200}

########## XGBoost ##########
f1 (Before): 0.5975
f1 (After): 0.5975
XGBoost best params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100}

########## LightGBM ##########
f1 (Before): 0.606
f1 (After): 0.606
LightGBM best params: {'learning_rate': 0.01, 'n_estimators': 300}

Hyperparameter Optimization for precision:

Hyperparameter Optimization....
########## KNN ##########
precision (Before): 0.9583
precision (After): 0.9583
KNN best params: {'n_neighbors': 5}

########## CART ##########
precision (Before): 0.9
precision (After): 0.9
CART best params: {'max_depth': 1, 'min_samples_split': 2}

########## RF ##########
precision (Before): 0.9185
precision (After): 0.8857
RF best params: {'max_depth': 8, 'max_features': 5, 'min_samples_split': 20, 'n_estimators': 300}

########## XGBoost ##########
precision (Before): 0.8631
precision (After): 0.8631
XGBoost best params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100}

########## LightGBM ##########
precision (Before): 0.7678
precision (After): 0.7678
LightGBM best params: {'learning_rate': 0.01, 'n_estimators': 300}

Hyperparameter Optimization for recall:

Hyperparameter Optimization....
########## KNN ##########
recall (Before): 0.3226
recall (After): 0.3226
KNN best params: {'n_neighbors': 5}

########## CART ##########
recall (Before): 0.461
recall (After): 0.461
CART best params: {'max_depth': 1, 'min_samples_split': 2}

########## RF ##########
recall (Before): 0.4259
recall (After): 0.461
RF best params: {'max_depth': 15, 'max_features': 5, 'min_samples_split': 15, 'n_estimators': 300}

########## XGBoost ##########
recall (Before): 0.4786
recall (After): 0.4786
XGBoost best params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100}

########## LightGBM ##########
recall (Before): 0.5175
recall (After): 0.5175
LightGBM best params: {'learning_rate': 0.01, 'n_estimators': 300}


Hyperparameter Optimization for accuracy:

Hyperparameter Optimization....
########## KNN ##########
accuracy (Before): 0.8561
accuracy (After): 0.8561
KNN best params: {'n_neighbors': 5}

########## CART ##########
accuracy (Before): 0.8781
accuracy (After): 0.8781
CART best params: {'max_depth': 1, 'min_samples_split': 2}

########## RF ##########
accuracy (Before): 0.8781
accuracy (After): 0.8707
RF best params: {'max_depth': 15, 'max_features': 7, 'min_samples_split': 15, 'n_estimators': 300}

########## XGBoost ##########
accuracy (Before): 0.8781
accuracy (After): 0.8781
XGBoost best params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100}

########## LightGBM ##########
accuracy (Before): 0.8599
accuracy (After): 0.8599
LightGBM best params: {'learning_rate': 0.01, 'n_estimators': 300}
"""

# Adım 9 Devamı: Stacking & Ensemble Learning

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)
    #    voting : {'hard', 'soft'}, default='hard' If 'hard', uses predicted class labels for majority rule voting.
    #    Else if 'soft', predicts the class label based on the argmax of the sums of the predicted probabilities, which is recommended for
    #    an ensemble of well-calibrated classifiers.
    
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)

"""
Accuracy: 0.8561253561253562
F1Score: 0.5282539682539683
ROC_AUC: 0.8937667590991042
"""
#Adım 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
rf_model = RandomForestClassifier(random_state=17)
best_params = {'max_depth': 8, 'max_features': 5, 'min_samples_split': 15, 'n_estimators': 300}
rf_final = rf_model.set_params(**best_params, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean() 
cv_results['test_f1'].mean() 
cv_results['test_roc_auc'].mean() 


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


plot_importance(rf_final, X, save=True)





