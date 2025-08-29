################################################################
# Customer Segmentation with Unsupervised Learning

#Business Problem:
#FLO wants to segment its customers and determine marketing strategies according 
# to these segments.To achieve this, customer behaviors will be identified, and 
# groups will be formed based on clusters within these behaviors.

# The dataset consists of information derived from the past shopping behaviors of 
# FLO customers who made their most recent purchases in 2020–2021 as OmniChannel 
# customers (those who shop both online and offline).
################################################################

################################################################
#IMPORT LIBARIES
################################################################
import pandas as pd 
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage, dendrogram


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

################################################################
#Görev 1: EDA
################################################################
df = pd.read_csv("flo_data_20K.csv")
df.head()

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

check_df(df)

date_columns = [col for col in df.columns if "date" in col.lower()]
for col in date_columns:
    df[col] = pd.to_datetime(df[col])


def grab_col_names(dataframe, cat_th=2, car_th=20):
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
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ['int', 'float', 'int64', 'float64']]
    num_cols = [col for col in num_cols if col not in cat_cols]
    date_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'datetime64[ns]']

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    print(f'date_cols: {len(date_cols)}')

    return cat_cols, num_cols, cat_but_car, date_cols

cat_cols, num_cols, cat_but_car, date_columns = grab_col_names(df)

# Data Visualization
def cat_summary(dataframe, col_name, plot=False):
    summary_df = pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
    })
    print(summary_df)
    print("##########################################")
    if plot:
        vc = dataframe[col_name].value_counts().reset_index()
        vc.columns = [col_name, 'count']
        fig = px.bar(vc, x=col_name, y='count', title=f'Bar Chart of {col_name}')
        fig.show()
    cols = [
    "order_num_total_ever_online",
    "order_num_total_ever_offline",
    "customer_value_total_ever_offline",
    "customer_value_total_ever_online"
    ]
    for col in cols:
        fig = px.box(df, x= col_name, y=col, title=f"{col} by {col_name}")
        fig.show()
        
for col in cat_cols:
    cat_summary(df, col, plot=True)
        
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    
    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram
        plt.subplot(2, 2, 1)
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col + ' Distribution')
        
        # Boxplot
        plt.subplot(2, 2, 2)
        sns.boxplot(y=numerical_col, data=dataframe)
        plt.title("Boxplot of " + numerical_col)
        plt.xticks(rotation=90)
        
        # Density Plot
        plt.subplot(2, 2, 3)
        sns.kdeplot(dataframe[numerical_col], shade=True)
        plt.xlabel(numerical_col)
        plt.title(numerical_col + ' Density')
        
        # QQ Plot
        plt.subplot(2, 2, 4)
        stats.probplot(dataframe[numerical_col], dist="norm", plot=plt)
        plt.title(numerical_col + ' QQ Plot')
        
        plt.tight_layout()
        plt.show(block=True)
        
for col in num_cols:
    num_summary(df, col, plot=True)
    
df.head()

#Handling Outliers
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.01, q3=0.99):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

#check_df(df)

for col in num_cols:
    print(col, check_outlier(df, col, 0.01, 0.99))

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)
    check_outlier(df, col)

#Correletion Analysis
corr_matrix = df.corr(numeric_only=True).round(2)

fig = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale='RdBu',
    title='Correlation Matrix with Heatmap'
)

fig.update_layout(
    width=700,           # Set figure width
    height=500,          # Set figure height
    font=dict(size=14)   # Set text size
)

fig.show()

#Feature Engineering / Feature Extraction
df['Total_Order'] = df['order_num_total_ever_offline'] + df['order_num_total_ever_online']
df['Total_Value'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']

analysis_date = df['last_order_date'].max() + dt.timedelta(days=1)
df["recency"] = (analysis_date - df["last_order_date"]).dt.days
df["tenure"] = (analysis_date - df["first_order_date"]).dt.days


df.head()

cat_cols, num_cols, cat_but_car, date_cols = grab_col_names(df)


df.shape

#interested_in_categories_12 --> num_interested_categories_12

# Define a function to clean and count categories
def count_cleaned_categories(categories_str):
    if not isinstance(categories_str, str):
        return 0 # Or handle as appropriate if other types exist
    # Split by comma, strip whitespace, and filter out any empty strings
    cleaned_categories = [cat.strip() for cat in categories_str.split(',') if cat.strip()]
    return len(cleaned_categories)

# Apply the function to create the new column
df['num_interested_categories_12'] = df['interested_in_categories_12'].apply(count_cleaned_categories)
df.head()

cat_cols, num_cols, cat_but_car, date_cols = grab_col_names(df)

#One-Hot Encoding for Categorical Variables
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    # Verilen DataFrame ve kategorik sütun listesini One-Hot Encoding'e tabi tutar.
    # 'drop_first' parametresi dışarıdan kontrol edilebilir.
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype='int64')
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()

#Select the required columns for clustering
# Drop the specified columns from the DataFrame
df_clustering = df.drop([
                'master_id',
                'first_order_date',
                'last_order_date',
                'last_order_date_online',
                'last_order_date_offline',
                'interested_in_categories_12'
                ], axis=1)

check_df(df_clustering)
df_clustering.info()
cat_cols, num_cols, cat_but_car, date_columns = grab_col_names(df_clustering)
################################################################
# Task 2: Customer Segmentation with K-Means
################################################################
#Step 1: Feature Scaling
# Apply RobustScaler to the numerical columns

rb = RobustScaler()
df_clustering[num_cols] = rb.fit_transform(df_clustering[num_cols])
df_clustering.head()

#Step 2: Determine the optimal number of clusters using the Elbow Method
kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df_clustering)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("SSE/SSR/SSD Values for different K values")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.show() # The elbow point is around 5-6 clusters

# Apply KElbowVisualizer to find the optimal number of clusters
K = KMeans()
elbow = KElbowVisualizer(K, k=(2,20))
elbow.fit(df_clustering)
elbow.show() 

elbow.elbow_value_ # The optimal number of clusters is 7

#Step 3: Create the KMeans model and fit it to the data
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df_clustering)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

#Step 4: Analyze each segment statistically
labels = kmeans.labels_

# Inverse transform the scaled numerical columns
df_clustering_kmeans = df_clustering.copy()
original_values = rb.inverse_transform(df_clustering_kmeans[num_cols])
df_clustering_kmeans = pd.DataFrame(original_values, columns=num_cols)
df_clustering_kmeans["clusters"] = labels+1 # Adding 1 to start clusters from 1 instead of 0

df_clustering_kmeans.groupby("clusters").agg({"mean","median", "min", "max","count"})

df_clustering_kmeans["clusters"].value_counts()


################################################################
#Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
################################################################
# Step 1: Determine the optimal number of clusters using the hierarchical 
# clustering method

df_clustering_hierarchical = df_clustering.copy()
df_clustering_hierarchical.head()

hc_average = linkage(df_clustering_hierarchical, "average")
#linkage: Hiyerarşik kümeleme algoritmasını uygulayan ana fonksiyon.
#dendrogram: Hiyerarşik kümeleme sonuçlarını ağaç yapısında görselleştiren fonksiyon.

#hc_average variable is created using the linkage function from the scipy library.
# This function performs hierarchical clustering on the provided data using the "average" method.

#Dendogram Visualization
plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
plt.show()

#Dendogram Visualization with Truncation
plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()

# Determine the number of clusters
plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y= 5, color='r', linestyle='--')
plt.axhline(y=4, color='b', linestyle='--')
plt.show()

# Step 2: Create your final model and segment your customers.
cluster = AgglomerativeClustering(n_clusters=7, linkage="average")
clusters = cluster.fit_predict(df_clustering_hierarchical)

# Inverse transform the scaled numerical columns
original_values = rb.inverse_transform(df_clustering_hierarchical[num_cols])
df_clustering_hierarchical = pd.DataFrame(original_values, columns=num_cols)
# Add the cluster labels to the DataFrame
df_clustering_hierarchical["clusters"] = clusters + 1  # Adding 1 to start clusters from 1 instead of 0

df_clustering_hierarchical.head()
# Step 3: Evaluate each model statistically
df_clustering_hierarchical.groupby("clusters").agg({"mean","median", "min", "max","count"})

df_clustering_hierarchical["clusters"].value_counts()
