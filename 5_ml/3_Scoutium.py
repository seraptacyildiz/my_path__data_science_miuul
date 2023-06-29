################################################################
# YAPAY ÖĞRENME İLE YETENEK AVCILIĞI SINIFLANDIRMA
################################################################

################################################################
# İş Problemi:
################################################################

#Scoutlar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf (average, highlighted) oyuncu olduğunu tahminleme.

################################################################
# Veriseti Hikayesi:
################################################################

# Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.

# attributes: Oyuncuları değerlendiren kullanıcıların bir maçta izleyip değerlendirdikleri her oyuncunun özelliklerine verdikleri puanları içeriyor. (bağımsız değişkenler)

#potential_labels: Oyuncuları değerlendiren kullanıcıların her bir maçta oyuncularla ilgili nihai görüşlerini içeren potansiyel etiketlerini içeriyor. (hedef değişken)

# 9 Değişken, 10730 Gözlem, 0.65 mb


################################################################
# Değişkenler:
################################################################

# task_response_id: Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi.

# match_id: İlgili maçın id'si.

# evaluator_id: Değerlendiricinin(scout'un) id'si.

# player_id: İlgili oyuncunun id'si.

# position_id: İlgili oyuncunun o maçta oynadığı pozisyonun id'si.
# 1- Kaleci
# 2- Stoper
# 3- Sağ bek
# 4- Sol bek
# 5- Defansif orta saha
# 6- Merkez orta saha
# 7- Sağ kanat
# 8- Sol kanat
# 9- Ofansif orta saha
# 10- Forvet

# analysis_id: Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme.

# attribute_id: Oyuncuların değerlendirildiği her bir özelliğin id'si.

# attribute_value: Bir scoutun bir oyuncunun bir özelliğine verilen değer(puan).

# potential_label: Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)


################################################################
# GÖREVLER
################################################################

# Görev 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.

# Görev 2: Okutmuş olduğumuz CSV dosyalarını merge fonksiyonunu kullanarak birleştirelim.  ("task_response_id", 'match_id', 'evaluator_id' "player_id"  4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)

# Görev 3: position_id içerisindeki Kaleci (1) sınıfını verisetinden kaldırınız.

# Görev 4: potential_label içerisindeki below_average sınıfını verisetinden kaldırınız.( below_average sınıfı  tüm verisetinin %1'ini oluşturur)

#Görev 5: Oluşturduğunuz verisetinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satır bir oyuncu olacak şekilde manipülasyon yapınız.

    #Adım 1: Her sütunda oyuncunun “position_id”, “potential_label” ve her oyuncunun sırayla bütün “attribute_idleri” içerecek şekilde işlem yapınız.

    #Adım 2: “reset_index” fonksiyonunu kullanarak index hatasından kurtulunuz ve “attribute_id” sütunlarının isimlerini stringe çeviriniz. (df.columns.map(str))

# Görev 6:  Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.

# Görev 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye kaydediniz.

# Görev 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için standardScaler uygulayınız.

# Görev 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli geliştiriniz.

# Görev 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.

import itertools
pip install xgboost
pip install catboost
import xgboost
import catboost
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 200)
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import warnings
warnings.filterwarnings('ignore')
################################################################
# Görev 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.
################################################################

df = pd.read_csv("C:/Users/Tugce.Dogan/Downloads/scoutium_attributes.csv", sep=";")
df.head()
df.shape
df2 = pd.read_csv("C:/Users/Tugce.Dogan/Downloads/scoutium_potential_labels.csv", sep=";")
df2.head()
df2.shape

################################################################
# Görev 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştirelim.  ("task_response_id", 'match_id', 'evaluator_id' "player_id"  4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)
################################################################

dff = pd.merge(df, df2, how='left', on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])

################################################################
# Görev 3: position_id içerisindeki Kaleci (1) sınıfını verisetinden kaldırınız.
################################################################

dff = dff[dff["position_id"] != 1]

################################################################
# Görev 4: potential_label içerisindeki below_average sınıfını verisetinden kaldırınız.( below_average sınıfı  tüm verisetinin %1'ini oluşturur)
################################################################

dff["potential_label"].value_counts()/dff["potential_label"].count()

dff = dff[dff["potential_label"] != "below_average"]

################################################################
#Görev 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.
################################################################

# Adım 1: İndekste “player_id”,“position_id” ve “potential_label”,  sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan “attribute_value” olacak şekilde pivot table’ı oluşturunuz.

pt = pd.pivot_table(dff, values="attribute_value", columns="attribute_id", index=["player_id","position_id","potential_label"])
pt.head()

#Adım 2: “reset_index” fonksiyonunu kullanarak index hatasından kurtulunuz ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.

pt = pt.reset_index(drop=False)
pt.head()
pt.columns = pt.columns.map(str)

################################################################
# Görev 3: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
################################################################

num_cols = pt.columns[3:]

##################################
# GÖREV 4: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# Adım 1: GENEL RESİM
##################################

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
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(pt)
"""
# eksik veri yapısının incelenmesi
pip install missingno
import missingno as msno

msno.matrix(pt)
plt.show()

msno.heatmap(pt)
plt.show()
"""
##################################
# Adım 2:  Numerik ve kategorik değişkenleri inceleyiniz.
##################################

##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in ["position_id","potential_label"]:
    cat_summary(pt, col)

##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(pt, col, plot=True)


##################################
# Adım 3:  Numerik değişkenler ile hedef değişken incelemesini yapınız.
##################################

##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(pt, "potential_label", col)


##################################
# Adım 4: Korelasyona bakınız.
##################################

pt[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(pt[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


##################################
# GÖREV 5: Feature Extraction uygulayın.
##################################

pt["min"] = pt[num_cols].min(axis=1)
pt["max"] = pt[num_cols].max(axis=1)
pt["sum"] = pt[num_cols].sum(axis=1)
pt["mean"] = pt[num_cols].mean(axis=1)
pt["median"] = pt[num_cols].median(axis=1)

pt.head()
pt["mentality"] = pt["position_id"].apply(lambda x: "defender" if (x == 2) | (x == 5) | (x == 3) | (x == 4) else "attacker")


for i in pt.columns[3:-6]:
    threshold = pt[i].mean() + pt[i].std()

    lst = pt[i].apply(lambda x: 0 if x < threshold else 1)
    pt[str(i) + "_FLAG"] = lst


flagCols = [col for col in pt.columns if "_FLAG" in col]

pt["counts"] = pt[flagCols].sum(axis=1)

pt["countRatio"] = pt["counts"] / len(flagCols)

pt.head()



"""avgDf = pt.groupby(["position_id", "potential_label"]).agg(["mean"]).reset_index()
avgDf = avgDf[avgDf["potential_label"] == "average"]
avgDf.columns[3:-41][0][0]"""

"""# HATA VAR
for i in range(2, 11):
    sozluk = {}
    for j in avgDf.columns[3:-41]:
        df = pt[pt["position_id"] == i]
        sozluk[j[0]] = df[j[0]].mean()
    print(i, sozluk)
    #pt[i[0]]

pt["4322"][3]"""

pt[pt["counts"] == 0]["potential_label"].value_counts()

pt[pt["counts"] != 0]["potential_label"].value_counts()

#########################################################

pt.head()

################################################################
# Görev 6:  Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.
################################################################

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


labelEncoderCols = ["potential_label","mentality"]

for col in labelEncoderCols:
    pt = label_encoder(pt, col)


################################################################
# Görev 7: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için standardScaler uygulayınız.
################################################################

pt.head()
lst = ["counts", "countRatio","min","max","sum","mean","median"]
num_cols = list(num_cols)

for i in lst:
    num_cols.append(i)

scaler = StandardScaler()
pt[num_cols] = scaler.fit_transform(pt[num_cols])

pt.head()


################################################################
# Görev 8: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli geliştiriniz.
################################################################


y = pt["potential_label"]
X = pt.drop(["potential_label", "player_id"], axis=1)


models = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   #("SVC", SVC()),
                   #("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   #('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('CatBoost', CatBoostClassifier(verbose=False)),
              ("LightGBM", LGBMClassifier())]


for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score+" score:"+str(cvs))

# Bonus
def sorting_score():
    list_cvs = []
    for name, model in models:
        for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
            cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
            list_cvs+=[[name,score,cvs]]
    cvs_df=pd.DataFrame(list_cvs,columns=["name","score","cvs"]).sort_values(by="cvs",ascending=False)
    return cvs_df

sorting_score()

"""
@Samet
def base_models(X, y, scoring):
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
                   ('CatBoost', CatBoostClassifier(verbose=False))]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


scores = ["roc_auc", "f1", "precision", "recall", "accuracy"]

for i in scores:
    base_models(X, y, i)

"""

################################################################
# Görev 9: Hiperparametre Optimizasyonu yapınız.
################################################################

lgbm_model = LGBMClassifier(random_state=46)

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]
             }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=10,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

# normal y cv süresi: 16.2s
# scale edilmiş y ile: 13.8s

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
    cvs = cross_val_score(final_model, X, y, scoring=score, cv=10).mean()
    print(score + " score:" + str(cvs))


################################################################
# Görev 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
################################################################

# feature importance
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = LGBMClassifier()
model.fit(X, y)

plot_importance(model, X)


for name, model in models[3:]:
    model=model.fit(X,y)
    plot_importance(model, X)