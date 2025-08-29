#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi ve averagebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchase metriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleri ab_testing.xlsx excel’inin ayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBidding uygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç


#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.


import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.
excel_data = pd.ExcelFile("ABTesti/ab_testing.xlsx")

control_df = pd.read_excel(excel_data, sheet_name='Control Group')
test_df = pd.read_excel(excel_data, sheet_name='Test Group')

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.
control_df.head()
control_df.shape
control_df.describe().T

test_df.head()
test_df.shape
test_df.describe().T

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.
control_df['Group'] = 'Control'  # Kontrol grubuna etiket ekliyoruz
test_df['Group'] = 'Test'  # Test grubuna etiket ekliyoruz

merged_df = pd.concat([control_df, test_df], ignore_index=True)

#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.

    #H0: Average Bidding ve Maximum Bidding arasındaki dönüşüm oranlarında (Purchase) anlamlı bir fark yoktur.
    # 𝜇₁ = 𝜇₂ (Average Bidding purchasing ortalaması = Maximum Bidding purchasing ortalaması)

    #H1: Average Bidding, Maximum Bidding'e kıyasla daha fazla dönüşüm sağlar. Yani, test grubunda (Average Bidding)
    # ortalama satın alma (Purchase), kontrol grubundakinden (Maximum Bidding) daha yüksektir.
    # 𝜇₁ > 𝜇₂ (Average Bidding ortalaması > Maximum Bidding ortalaması)

# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz

purchase_means = merged_df.groupby('Group')['Purchase'].mean()

#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz

# Normallik Varsayımı:
control_shapiro_stat, control_shapiro_p_value = shapiro(merged_df[merged_df['Group'] == 'Control']['Purchase'])
test_shapiro_stat, test_shapiro_p_value = shapiro(merged_df[merged_df['Group'] == 'Test']['Purchase'])

print(f"Kontrol Grubu Shapiro-Wilk Testi p-değeri: {control_shapiro_p_value}")
print(f"Test Grubu Shapiro-Wilk Testi p-değeri: {test_shapiro_p_value}")

#SONUÇ: H0 reddedilemez -->Normal dağılım
"""
# p-değerine göre normallik testi sonucu
if control_shapiro_p_value < 0.05:
    print("Kontrol grubu verisi normal dağılımdan sapmaktadır.")
else:
    print("Kontrol grubu verisi normal dağılımdan sapmamaktadır.")

if test_shapiro_p_value < 0.05:
    print("Test grubu verisi normal dağılımdan sapmaktadır.")
else:
    print("Test grubu verisi normal dağılımdan sapmamaktadır.")
"""
#Varyans Homojenliği:
#H0 (Null Hipotezi): Kontrol ve test gruplarının varyansları eşittir.
#H1 (Alternatif Hipotez): Kontrol ve test gruplarının varyansları eşit değildir.

levene_stat, levene_p_value = levene(
    merged_df[merged_df['Group'] == 'Control']['Purchase'],  # Kontrol grubu
    merged_df[merged_df['Group'] == 'Test']['Purchase']     # Test grubu
)

print(f"Levene Testi p-değeri: {levene_p_value}")

#Varyanslar homojendir

# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz
t_stat, p_value = ttest_ind(
    merged_df[merged_df['Group'] == 'Control']['Purchase'],  # Kontrol grubu verisi
    merged_df[merged_df['Group'] == 'Test']['Purchase']     # Test grubu verisi
)

print(f"T-Statistiği: {t_stat}")
print(f"P-Değeri: {p_value}")


# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.


#SONUÇ: H0 reddedilemez (İki grup arasındaki fark istatistiksel olarak anlamlı değildir.)

# T-Statistiği = -0.9416: Bu değer, iki grup arasındaki farkın ne kadar büyük olduğunu gösterir.
# Ancak p-değeri ile birlikte değerlendirildiğinde, T-Statistiği'nin mutlak değeri küçük olduğu
# için gruplar arasındaki farkın anlamlı bir düzeyde olmadığını gösterir. Yani, farkın büyüklüğü ne kadar küçükse, p-değeri de o kadar yüksek olur.



##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

#Varsayım kontrolünde hem normal dağılım hem de varyansın homojenliği görülüdüğü için t testi (parametrik test)
#kullanıldı.


# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

#Sonuç: Test sonuçlarına göre, average bidding (test grubu) yöntemi ile maximum bidding (kontrol grubu)
# arasında anlamlı bir fark yoktur. Bu durumda, average bidding yönteminin maksimum teklifler
# üzerinden bir avantaj sağlamadığını söyleyebiliriz.

# Tavsiye:
# 1- Eğer average bidding yöntemi daha düşük maliyetle uygulanabiliyorsa ve aynı dönüşümü sağlıyorsa,
# daha düşük maliyetle sürdürülebilir performans sağlanabilir.

# 2- Testin süresi uzatılabilir ve başka faktörler (örneğin, zaman dilimi, kullanıcı segmentasyonu) 
# göz önünde bulundurularak farklı analizler yapılabilir. 

# 3- Gelecekteki testlerde, farklı segmentler veya kampanya hedefleri ile her iki teklif verme türünü
# inceleyerek daha spesifik sonuçlar elde edilebilir.
