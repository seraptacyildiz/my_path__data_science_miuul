
##################################################
# Pandas Alıştırmalar
##################################################
import numpy as np
import seaborn as sns
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################

df = sns.load_dataset("titanic")
df.head()

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################
df["sex"].value_counts()

#alt2:
df.groupby("sex").agg({"sex": ["count"]})


#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################
for col in df.columns:
    print(f"{col} : {df[col].nunique()}")

#alt2:
df.nunique()

#alt3:
[f"{col} : {df[col].nunique()}" for col in df.columns ]


#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################

df["pclass"].unique()

#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################

df[["pclass", "parch"]].nunique()

#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################

df.info()

df["embarked"] = df["embarked"].astype("category")

#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

df[df["embarked"] == "C"].head()


#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################
df[df["embarked"] != "S"].head()


#alt2:
df[(df["embarked"] == "C") | (df["embarked"] == "Q")].head()

#alt3:
df[~(df["embarked"] == "S")].head()

#alt4:
df.loc[df["embarked"] != "S"]

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################
df[(df["age"] < 30) & (df["sex"] == "female")].head()



#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################
df[(df["fare"] > 500) | (df["age"] > 70)].head()

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################
df.info()
df.isnull().sum() #her bir değişkeni ayrı ayrı verir.
df.isnull().sum().sum() #tümünün toplamını verir.

#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################
df = df.drop("who", axis=1)
df.head()

#alt2:
df.drop("who", axis=1, inplace=True)

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################
df["deck"].mode()[0]
deck_mode = df["deck"].mode()[0]

df["deck"] = df[["deck"]].apply(lambda col: col.fillna(deck_mode))

#alt2:
df["deck"] = df[["deck"]].apply(lambda col: col.fillna(df["deck"].mode()[0]))

#alt3:
df['deck'] = df['deck'].fillna("C")

#alt4:
df["deck"].fillna(df["deck"].mode()[0], inplace = True)

#alt5:
df["deck"] = df["deck"].fillna(df["deck"].mode()[0])

#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################

df["age"].isnull().sum()
age_median = df["age"].median()
df["age"] = df[["age"]].apply(lambda col : col.fillna(age_median))

#alt2:
df["age"].median()
df["age"].fillna(df["age"].median(), inplace=True)
df["age"]

#alt3:
df["age"] = df["age"].fillna(df["age"].median())

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################

agg_list = ["sum", "count", "mean"]
df.groupby(["pclass", "sex"]).agg({"survived" : agg_list})

#alt2:
df.groupby(["sex", "pclass"]).agg({"survived": ["mean", "sum", "count"]})

#alt3:
df.pivot_table("survived", ["pclass", "sex"], aggfunc=["sum", "count", "mean"])

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################
def flag_age(age):
    if age < 30:
        return 1
    else:
        return 0

df["age_flag"] = df["age"].apply(lambda x: flag_age(x))
df.head()

#alt2:
def age_cat(df):
    df['age_flag'] = df['age'].apply(lambda x: 1 if x < 30 else 0)
age_cat(df)


#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################
import pandas as pd
import numpy as np
import seaborn as sns

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

df = sns.load_dataset("tips")
df.head()

#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################
agg_list = ["sum", "min", "max", "mean"]
df.groupby(["time"]).agg({"total_bill": agg_list})

#alt2:
df.groupby(["time"]).agg({"total_bill": ["sum", "min", "max", "mean"]})

#alt3:
df.pivot_table("total_bill", "time", aggfunc=["sum", "min", "max", "mean"])

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################
agg_list = ["sum", "min", "max", "mean"]
df.groupby(["day", "time"]).agg({"total_bill": agg_list})

#alt2:
df.groupby(["day", "time"]).agg({"total_bill": ["sum", "min", "max", "mean"]})

#alt3:
df.pivot_table("total_bill", ["day", "time"], aggfunc=["sum","min", "max", "mean"])
df.pivot_table("total_bill", "day", "time", aggfunc=["sum","min", "max", "mean"])

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################
agg_list = ["sum", "min", "max", "mean"]
df2 = df[(df["sex"] == "Female") & (df["time"] == "Lunch")]
df2.groupby(["day"]).agg({"total_bill": agg_list, "tip" : agg_list})

#alt2:
lunch_female = df[(df.time == 'Lunch') & (df.sex == 'Female')]
lunch_female.groupby(["day"]).agg({"total_bill": ["sum","min", "max", "mean"], "tip": ["sum","min", "max", "mean"]})

#alt3:
df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum", "min", "max", "mean"],
                                                                          "tip": ["sum", "min", "max", "mean"]})
#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################

df.loc[(df["size"] < 3) & (df["total_bill"] > 10)].mean()

#alt2:
df[(df["size"] < 3) & (df["total_bill"] > 10)].mean()


#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()

df["total_bill_tip_sum"].sum()

#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
new_df = df.sort_values("total_bill_tip_sum", ascending=False)[:30]
new_df = new_df.reset_index()
new_df = new_df.drop("index", axis=1)
new_df.head()


#alt2:
df4 = df["total_bill_tip_sum"].sort_values(ascending = False).iloc[0:30]
df4 = df4.reset_index()
df4 = df4.drop("index", axis = "columns")

#alt3:
tips_new = df.sort_values(by="total_bill_tip_sum", ascending=False).head(30)
df4 = df4.reset_index()
df4 = df4.drop("index", axis = "columns")