#############################################
# Association Rule-Based Recommender System

##Business Problem:

#Aşağıda 3 farklı kullanıcının sepet bilgileri verilmiştir. Bu sepet bilgilerine en uygun ürün önerisini
# birliktelik kuralı kullanarak yapınız. Ürün önerileri 1 tane ya da 1'den fazla olabilir. Karar kurallarını
# 2010-2011 Germany müşterileri üzerinden türetiniz.


#Kullanıcı 1’in sepetinde bulunan ürünün id'si: 21987
# Kullanıcı 2’in sepetinde bulunan ürünün id'si : 23235
# Kullanıcı 3’in sepetinde bulunan ürünün id'si : 22747
#############################################

import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from mlxtend.frequent_patterns import apriori, association_rules

#############################################
# Görev 1: Verinin Hazırlanması
#############################################
# Adım 1: Online Retail II veri setinden 2010-2011 sheet’ini okutunuz.
retail_df = pd.read_excel("Bonus Project/online_retail_II.xlsx", sheet_name="Year 2010-2011")
retail_df.head()
retail_df.shape
retail_df.describe().T
retail_df.isnull().sum()

#Adım 2: StockCode’u POST olan gözlem birimlerini drop ediniz. (POST her faturaya eklenen bedel, ürünü ifade etmemektedir.)
#Adım 3: Boş değer içeren gözlem birimlerini drop ediniz.
retail_df.dropna(inplace=True)
retail_df = retail_df[retail_df['StockCode'] != 'POST']
#Adım 4: Invoice içerisinde C bulunan değerleri veri setinden çıkarınız. (C faturanın iptalini ifade etmektedir.)
retail_df = retail_df[~retail_df["Invoice"].str.contains("C", na=False)]
#Adım 5: Price değeri sıfırdan küçük olan gözlem birimlerini filtreleyiniz.
retail_df = retail_df[retail_df['Price'] > 0]

#Adım 6: Price ve Quantity değişkenlerinin aykırı değerlerini inceleyiniz, gerekirse baskılayınız.
def outlier_thresholds(dataframe, variable):
    # Değişkenin 1. yüzdelik dilimini (0.01) al — bu, genellikle 0.25 (1. çeyrek) olarak kullanılır, burada daha uçtan alınmış
    quartile1 = dataframe[variable].quantile(0.01)
    # Değişkenin 99. yüzdelik dilimini (0.99) al — bu, genellikle 0.75 (3. çeyrek) olarak kullanılır, burada daha uçtan alınmış
    quartile3 = dataframe[variable].quantile(0.99)
    # Çeyrekler arası farkı (IQR) hesapla
    interquantile_range = quartile3 - quartile1
    # Üst sınırı hesapla: 3. çeyrek + 1.5 * IQR
    up_limit = quartile3 + 1.5 * interquantile_range
    # Alt sınırı hesapla: 1. çeyrek - 1.5 * IQR
    low_limit = quartile1 - 1.5 * interquantile_range
    # Alt ve üst sınırları döndür
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    #Eşik değerlerini al
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # Alt sınırdan küçük olan değerlere alt sınırı ata
    # Üst sınırdan büyük olan değerlere üst sınırı ata
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    # Miktar (Quantity) sütunundaki aykırı değerleri baskıla
    replace_with_thresholds(dataframe, "Quantity")
    # Fiyat (Price) sütunundaki aykırı değerleri baskıla
    replace_with_thresholds(dataframe, "Price")
    # Temizlenmiş ve aykırı değerleri baskılanmış dataframe'i döndür
    return dataframe

retail_df = retail_data_prep(retail_df)


#############################################
## Görev 2: Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme
#############################################
#Adım 1: Aşağıdaki gibi fatura ürün pivot table’i oluşturacak create_invoice_product_df
# fonksiyonunu tanımlayınız.
def create_invoice_product_df(dataframe,id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            map(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            map(lambda x: 1 if x > 0 else 0)
#Adım 2: Kuralları oluşturacak create_rules fonksiyonunu tanımlayınız ve alman müşteriler
# için kurallarını bulunuz.

"""
retail_df_ger= retail_df[retail_df['Country'] == "Germany"]
retail_df_ger.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)
#Sadece almanyaya ait bir pivot tablo
ger_inv_pro_df = create_invoice_product_df(retail_df_ger)

#apriori() fonksiyonu, verilen veri kümesindeki sık geçen itemset’leri (ürün kombinasyonları) bulur.
#min_support=0.01: En az %1 oranında geçen kombinasyonları dikkate al.
# use_colnames=True: Ürün ID'leri yerine gerçek ürün adlarını göster.
frequent_itemsets = apriori(ger_inv_pro_df,
                            min_support=0.01,
                            use_colnames=True)

#Sık çıkan ürün kombinasyonlarını destek değerine (support) göre sıralar.
#En çok birlikte görülen ürünleri en üstte görebilirsin.
frequent_itemsets.sort_values("support", ascending=False)

# association_rules() fonksiyonu, yukarıda bulunan sık itemset’lerden ilişki kuralları oluşturur.
# metric="support": Kuralların filtrelenmesinde destek (support) metriğini kullan.
# min_threshold=0.01: En az %1 destek değeri olan kurallar filtrelenir.
#Bu fonksiyon, Apriori veya FP-Growth gibi algoritmalarla elde edilen frequent_itemsets’lerden
# şu şekilde kurallar üretir:
# Eğer ürün A alındıysa, ürün B'nin alınma olasılığı yüksektir.

Kullanılan parametreler
| Parametre           | Açıklama                                                                                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `frequent_itemsets` | `apriori()` veya `fpgrowth()` ile elde edilen sonuç.                                                                                                    |
| `metric`            | Kuralları filtrelemek için hangi metriğe göre sıralama yapılacağını belirtir: `"support"`, `"confidence"`, `"lift"`, `"leverage"`, `"conviction"` gibi. |
| `min_threshold`     | Seçilen `metric` için minimum eşik değeri. Örn: `confidence >= 0.5`.                                                                                    |


rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

#Güçlü kuralları filtrele:
# support > 0.05: En az %5 sepet içinde birlikte geçmiş.
# confidence > 0.1: Ön koşul gerçekleştiğinde sonuç ürünün alınma olasılığı > %10.
# lift > 5: Bu kural rastlantıdan çok daha güçlü bir bağımlılık içeriyor.

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]

"""

def create_rules(df, country="Germany",
                 min_support=0.01,
                 metric="support",
                 min_threshold=0.01,
                 support_filter=0.05,
                 confidence_filter=0.1,
                 lift_filter=5):
    """
    Belirli bir ülke için association rule mining işlemi yapar.

    Parametreler:
    - df: Veri çerçevesi
    - country: Hangi ülkeye ait veriler kullanılacak (varsayılan: "Germany")
    - min_support: Apriori için minimum destek eşiği
    - metric: association_rules için kullanılacak metrik
    - min_threshold: association_rules için minimum eşik değeri
    - support_filter: Filtreleme için minimum support
    - confidence_filter: Filtreleme için minimum confidence
    - lift_filter: Filtreleme için minimum lift

    Dönüş: Filtrelenmiş kurallar DataFrame'i
    """

    # Sadece ilgili ülkeye ait verileri al
    df_country = df[df['Country'] == country]

    # Invoice-Product pivot tablosu oluştur
    invoice_product_df = create_invoice_product_df(df_country,id=True)

    # Apriori ile sık ürün setlerini bul
    frequent_itemsets = apriori(invoice_product_df,
                                min_support=min_support,
                                use_colnames=True)

    # Association kuralları oluştur
    rules = association_rules(frequent_itemsets,
                              metric=metric,
                              min_threshold=min_threshold)

    # Güçlü kuralları filtrele
    #rules = rules[(rules["support"] > support_filter) &
                           #(rules["confidence"] > confidence_filter) &
                          # (rules["lift"] > lift_filter)]

    return rules

rules_germany = create_rules(retail_df)

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

#############################################
## Görev 3:  Sepet İçerisindeki Ürün Id’leri Verilen Kullanıcılara Ürün Önerisinde Bulunma
#############################################

#Adım-1: check_id() fonksiyonu, StockCode’a karşılık gelen Description (yani ürün ismi) bilgisini
# ekrana basar.
product_1 = check_id(retail_df, 21987)
product_2 = check_id(retail_df,23235 )
product_3 = check_id(retail_df,22747 )


#Adım 2: arl_recommender fonksiyonunu kullanarak 3 kullanıcı için ürün önerisinde bulununuz.
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

customer_1_rec=arl_recommender(rules_germany, 21987, 1)
customer_2_rec=arl_recommender(rules_germany, 23235, 1)
customer_3_rec=arl_recommender(rules_germany, 22747, 1)

#Adım 3: Önerilecek ürünlerin isimlerine bakınız.
def check_id_list(dataframe, stock_code_list):
    """
    Verilen stok kodlarına karşılık gelen ürün açıklamalarını yazdırır.

    Parametreler:
    - dataframe: Ürün verisi içeren DataFrame
    - stock_code_list: Stok kodlarının listesi (örneğin ["85123A", "84029G"])
    """
    for code in stock_code_list:
        descriptions = dataframe[dataframe["StockCode"] == code]["Description"].unique()
        if len(descriptions) > 0:
            print(f"{code}: {descriptions[0]}")
        else:
            print(f"{code}: Not found")

check_id_list(retail_df,customer_1_rec)
check_id_list(retail_df,customer_2_rec)
check_id_list(retail_df,customer_3_rec)


