
###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

##master_id: Eşsiz müşteri numarası (customer_id)
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
## first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
## last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
## interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.
           # 2. Veri setinde
                     # a. İlk 10 gözlem,
                     # b. Değişken isimleri,
                     # c. Betimsel istatistik,
                     # d. Boş değer,
                     # e. Değişken tipleri, incelemesi yapınız.
           # 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
           # 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
           # 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.
           # 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
           # 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
           # 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.

# GÖREV 2: RFM Metriklerinin Hesaplanması

# GÖREV 3: RF ve RFM Skorlarının Hesaplanması

# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması

# GÖREV 5: Aksiyon zamanı!
           # 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
           # 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.
                   # a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
                   # tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers),
                   # ortalama 250 TL üzeri ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına
                   # yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.
                   # b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
                   # alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
                   # olarak kaydediniz.


# GÖREV 6: Tüm süreci fonksiyonlaştırınız.

###############################################################
# GÖREV 1: Veriyi  Hazırlama ve Anlama (Data Understanding)
###############################################################
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

flo_df_ = pd.read_csv("/Users/sametgirgin/PycharmProjects/Cases/flo_data_20k.csv")
flo_df = flo_df_.copy()
# 2. Veri setinde
        # a. İlk 10 gözlem,
flo_df.head()
        # b. Değişken isimleri,
column_names = flo_df.columns.tolist()
        # c. Boyut,
flo_df.shape
        # d. Betimsel istatistik,
flo_df.describe().T
        # e. Boş değer,
flo_df.isnull().sum()
        # f. Değişken tipleri, incelemesi yapınız.
flo_df.dtypes

# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
flo_df["order_num_total"] = flo_df["order_num_total_ever_online"] + flo_df["order_num_total_ever_offline"]
flo_df["customer_value_total"] = flo_df["customer_value_total_ever_online"] + flo_df["customer_value_total_ever_offline"]

# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
date_columns = [
    "first_order_date",
    "last_order_date",
    "last_order_date_online",
    "last_order_date_offline"
]
flo_df[date_columns] = flo_df[date_columns].apply(pd.to_datetime)
flo_df.dtypes

# 5. Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısı ve toplam harcamaların dağılımına bakınız.
channel_summary = flo_df.groupby("order_channel").agg({
    "master_id": "nunique",
    "order_num_total": "sum",
    "customer_value_total": "sum"
}).rename(columns={
    "master_id": "musteri_sayisi",
    "order_num_total": "toplam_urun_sayisi",
    "customer_value_total": "toplam_harcama"
}).reset_index()

# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
top_earning_customers = flo_df.sort_values("customer_value_total", ascending=False)[["master_id", "customer_value_total"]].head(10)
# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
top_ordering_customers= flo_df.sort_values("order_num_total", ascending=False)[["master_id", "order_num_total"]].head(10)

# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
flo_df.describe().T

def prepare_flo_data(dataframe):
    # Tarih değişkenlerini datetime formatına çevirme
    date_columns = [
        "first_order_date",
        "last_order_date",
        "last_order_date_online",
        "last_order_date_offline"
    ]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # Toplam alışveriş sayısı ve toplam harcama değişkenlerini oluşturma
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]
    return dataframe
prepared_flo_df = prepare_flo_data(flo_df_)
prepared_flo_df.dtypes

###############################################################
# GÖREV 2: RFM Metriklerinin Hesaplanması
###############################################################

# Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi
analysis_date = flo_df["last_order_date"].max() + pd.Timedelta(days=2) # Timestamp('2021-06-01 00:00:00')

# customer_id, recency, frequnecy ve monetary değerlerinin yer aldığı yeni bir rfm dataframe
rfm = flo_df[["master_id", "last_order_date", "order_num_total", "customer_value_total"]].copy()

rfm["recency"] = (analysis_date - rfm["last_order_date"]).dt.days
rfm["frequency"] = rfm["order_num_total"]
rfm["monetary"] = rfm["customer_value_total"]

rfm = rfm[["master_id", "recency", "frequency", "monetary"]]
###############################################################
# GÖREV 3: RF ve RFM Skorlarının Hesaplanması (Calculating RF and RFM Scores)
###############################################################

#  Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çevrilmesi ve
# Bu skorları recency_score, frequency_score ve monetary_score olarak kaydedilmesi

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

# recency_score ve frequency_score’u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi
rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)


###############################################################
# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
###############################################################
# Oluşturulan RFM skorların daha açıklanabilir olması için segment tanımlama ve  tanımlanan seg_map yardımı ile RF_SCORE'u segmentlere çevirme
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

###############################################################
# GÖREV 5: Aksiyon zamanı!
###############################################################

# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean"])

"""
Champions segmentindeki müşteriler çok yakın zamanda alışveriş yapmış (ortalama 
17 gün önce), sık alışveriş yapıyor (8.9) ve yüksek harcama yapıyor (₺1406.6).

Hibernating ve At Risk segmentlerindeki müşteriler uzun süredir alışveriş yapmamış 
ve düşük sıklıkla alışveriş yapıyorlar.

Can’t Lose müşteriler yüksek harcama ve sipariş oranlarına sahip, ancak uzun süredir 
alışveriş yapmamış.
"""

# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulunuz ve müşteri id'lerini csv ye kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Bu müşterilerin sadık  ve
# kadın kategorisinden alışveriş yapan kişiler olması planlandı. Müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs
# olarak kaydediniz.

merged_flo_df = flo_df.merge(rfm[["master_id", "segment"]], on="master_id")

target_customers = merged_flo_df[
    (merged_flo_df["segment"].isin(["champions", "loyal_customers"])) &
    (merged_flo_df["interested_in_categories_12"].apply(lambda x: "KADIN" in x))
]

target_ids = target_customers[["master_id"]]
target_ids.to_csv("hedef_müşteriler_1.csv")

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.
target_segments = ['at_Risk', 'about_to_sleep', 'new_customers']

discount_target_customers = merged_flo_df[
    (merged_flo_df["segment"].isin(target_segments)) &
    (merged_flo_df["interested_in_categories_12"].apply(lambda x: any(cat in x for cat in ["ERKEK", "COCUK"])))
]

# Sadece müşteri ID'leri
discount_target_ids = discount_target_customers[["master_id"]]
discount_target_ids.to_csv("indirimli_müşteriler_1.csv")
