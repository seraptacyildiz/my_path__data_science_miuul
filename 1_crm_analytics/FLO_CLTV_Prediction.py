##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############################################################
# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
           # 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
           # Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
           # 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
           # aykırı değerleri varsa baskılayanız.
           # 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
           # 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 400)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df_ = pd.read_csv(r"D:\DataScience\Python\Miuul_Data_Science\Hard_Skill\Hard_Skill_Ödevler\Third_Alıstırmalar\FLOMusteriSegmentasyonu\flo_data_20k.csv")
df = df_.copy()
df.head(10)
df.columns
df.describe().T
df.isnull().sum()
df.info()
df["master_id"].nunique()

# 2.Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# alt ve üst değerleri yuvarlar max ve min değerlere göre
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] =  round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)


 # 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
           # aykırı değerleri varsa baskılayanız.
replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")
df.head()

#alternatif
#numeric_cols = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]
#for col in numeric_cols:
 #   replace_with_thresholds(df, col)



# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.


df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head()

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])
df.info()



# GÖREV 2: CLTV Veri Yapısının Oluşturulması
           # 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
           # 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
           # Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

# 1) recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# 2) T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# 3) frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# 4) monetary: satın alma başına ortalama kazanç


df["last_order_date"].max()   # Timestamp('2021-05-30 00:00:00')
today_date = dt.datetime(2021, 6, 1)
cltv_df = df[["master_id", "last_order_date", "first_order_date", "total_order", "total_value"]]

cltv_df.rename(columns={"master_id": "customer_id"}, inplace=True)

# 1) recency:
cltv_df["recency_weekly"] = (cltv_df["last_order_date"] - cltv_df["first_order_date"]).dt.days/7
# 2) T: Müşterinin yaşı
cltv_df["T_weekly"] = (today_date - cltv_df["first_order_date"]).dt.days/7
# 3) frequency:
cltv_df.rename(columns={"total_order": "frequency"}, inplace=True)
cltv_df = cltv_df[cltv_df["frequency"] > 1]
# 4) monetary:
cltv_df["monetary_avg"] = cltv_df["total_value"] / cltv_df["frequency"]


cltv_df = cltv_df[["customer_id", "recency_weekly", "T_weekly", "frequency", "monetary_avg"]]

cltv_df.head(5)
cltv_df.columns

# alt
#df["recency"] = df["last_order_date"]-df["first_order_date"]
# cltv_df = df.groupby("master_id").agg({"recency": [lambda InvoiceDate: (InvoiceDate.max()).days],
                                 #      "first_order_date": [lambda recency: (today_date - recency.max()).days],
                              # "TotalPurchase": "sum",
                              #  "TotalPrice": "sum"})




# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
           # 1. BG/NBD modelini fit ediniz. # satın alma sayısını #Beta Geometric / Negative Binomial Distribution
                # a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
                # b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

# 1. BG/NBD modelini fit ediniz
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency_weekly"],
        cltv_df["T_weekly"])

# a = 3 Ay

cltv_df["sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency_weekly"],
                                                        cltv_df["T_weekly"])
# b = 6 Ay

cltv_df["sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency_weekly"],
                                                        cltv_df["T_weekly"])
cltv_df.head()


            # sipariş başına ortalama satın alma
           # 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.

# 2. Gamma-Gamma modelini fit
ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_avg"])
cltv_df.head()
cltv_df.sort_values("exp_average_value", ascending=False).head()

cltv_df.head()




# Cltv Tahmini
           # 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv_df["clv"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency_weekly"],
                                   cltv_df["T_weekly"],
                                   cltv_df["monetary_avg"],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin zaman bilgisi(haftalık)
                                   discount_rate=0.01)
cltv_df.head(10)

                # b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv_df.sort_values("clv", ascending=False).head(20)




# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
           # 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["clv"], 4, labels=["D", "C", "B", "A"])
cltv_df.sort_values("clv", ascending=False).head(20)
cltv_df["cltv_segment"].value_counts()
cltv_df.groupby("cltv_segment").agg({"clv": ["mean", "min", "max"]})
cltv_df.head()
           # 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz



###############################################################
# BONUS: Tüm süreci fonksiyonlaştırınız.
###############################################################

def create_cltv_df(dataframe):

    def outlier_thresholds(dataframe, variable):
        quartile1 = dataframe[variable].quantile(0.01)
        quartile3 = dataframe[variable].quantile(0.99)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit

    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
        dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

    numeric_cols = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]
    for variable in numeric_cols:
        replace_with_thresholds(dataframe, variable)

    replace_with_thresholds(df, "order_num_total_ever_online")
    replace_with_thresholds(df, "order_num_total_ever_offline")
    replace_with_thresholds(df, "customer_value_total_ever_offline")
    replace_with_thresholds(df, "customer_value_total_ever_online")

    df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col])

    df["last_order_date"].max()
    today_date = dt.datetime(2021, 6, 1)
    cltv_df = df[["master_id", "last_order_date", "first_order_date", "total_order", "total_value"]]
    cltv_df.rename(columns={"master_id": "customer_id"}, inplace=True)
    cltv_df.rename(columns={"total_order": "frequency"}, inplace=True)
    cltv_df["recency_weekly"] = (cltv_df["last_order_date"] - cltv_df["first_order_date"]).dt.days / 7
    cltv_df["T_weekly"] = (today_date - cltv_df["first_order_date"]).dt.days / 7
    cltv_df["monetary_avg"] = cltv_df["total_value"] / cltv_df["frequency"]
    cltv_df = cltv_df[cltv_df["frequency"] > 1]
    cltv_df = cltv_df[["customer_id", "recency_weekly", "T_weekly", "frequency", "monetary_avg"]]


    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df["frequency"],
            cltv_df["recency_weekly"],
            cltv_df["T_weekly"])



    cltv_df["sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 3,
                                                                                       cltv_df["frequency"],
                                                                                       cltv_df["recency_weekly"],
                                                                                       cltv_df["T_weekly"])


    cltv_df["sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 6,
                                                                                       cltv_df["frequency"],
                                                                                       cltv_df["recency_weekly"],
                                                                                      cltv_df["T_weekly"])
    print("sales_3-6_month")
    print(cltv_df.head())


    ggf = GammaGammaFitter(penalizer_coef=0.01)

    ggf.fit(cltv_df["frequency"], cltv_df["monetary_avg"])

    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                           cltv_df["monetary_avg"])
    cltv_df.sort_values("exp_average_value", ascending=False).head()
    cltv_df.reset_index()
    print(cltv_df.head())


    cltv_df["clv"] = ggf.customer_lifetime_value(bgf,
                                                 cltv_df["frequency"],
                                                 cltv_df["recency_weekly"],
                                                 cltv_df["T_weekly"],
                                                 cltv_df["monetary_avg"],
                                                 time=6,  # 6 aylık
                                                 freq="W",  # T'nin zaman bilgisi(haftalık)
                                                 discount_rate=0.01)


    print(cltv_df.sort_values("clv", ascending=False).head(20))


    cltv_df["cltv_segment"] = pd.qcut(cltv_df["clv"], 4, labels=["D", "C", "B", "A"])
    cltv_df.sort_values("clv", ascending=False).head(20)
    cltv_df["cltv_segment"].value_counts()
    print(cltv_df.groupby("cltv_segment").agg({"clv": ["mean", "min", "max"]}))




    return

df = df_.copy()
create_cltv_df(df)




