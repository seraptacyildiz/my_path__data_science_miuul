import pickle
from helpers.data_prep import *
from helpers.eda import *
from helpers.helpers import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


############## DATASET HISTORY ##############################################
# Pregnancies: Number of times pregnant
# Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test (70-105 referans aralığı)
# BloodPressure: Diastolic blood pressure (mm Hg)
# SkinThickness: Triceps skin fold thickness (mm)
# Insulin: 2-Hour serum insulin (mu U/ml) hormon- Glikozu hücre içine alır, yetersizse glikozu hücre içine alamaz kandaki şeker artar(2.6-24.9 referans aralığı)
# BMI: Body mass index (weight in kg/(height in m)2) glikoz hücreye giremediği için hiperglisemi obeziteye yol açıyor
# DiabetesPedigreeFunction: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)
# Age: Age (years)
# Outcome: Class variable (0 if non-diabetic, 1 if diabetic) DEPENDENT VARIABLE
##########################################################################


# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
           # Adım 1: Genel resmi inceleyiniz.
           # Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
           # Adım 3:  Numerik ve kategorik değişkenlerin analizini yapınız.
           # Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
           # Adım 5: Aykırı gözlem analizi yapınız.
           # Adım 6: Eksik gözlem analizi yapınız.
           # Adım 7: Korelasyon analizi yapınız.

# GÖREV 2: FEATURE ENGINEERING
           # Adım 1:  Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
           # değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri
           # 0 olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere
           # işlemleri uygulayabilirsiniz.
           # Adım 2: Yeni değişkenler oluşturunuz.
           # Adım 3:  Encoding işlemlerini gerçekleştiriniz.
           # Adım 4: Numerik değişkenler için standartlaştırma yapınız.
           # Adım 5: Model oluşturunuz.

##################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# GENEL RESİM
##################################

def load():
    data = pd.read_csv("C:/Users/Tugce.Dogan/Downloads/diabetes.csv")
    return data


df = load()
df_ = df.copy()
check_df(df)

df["Outcome"].value_counts()


##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
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

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car


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

cat_summary(df, "Outcome",plot=True)


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
    num_summary(df, col, plot=True)

##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


##################################
# KORELASYON
##################################

df.corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

df.corrwith(df["Outcome"]).sort_values(ascending=False)


diabetic = df[df.Outcome == 1]
healthy = df[df.Outcome == 0]

plt.scatter(healthy.Age, healthy.Insulin, color="green", label="Healthy", alpha = 0.4)
plt.scatter(diabetic.Age, diabetic.Insulin, color="red", label="Diabetic", alpha = 0.4)
plt.xlabel("Age")
plt.ylabel("Insulin")
plt.legend()
plt.show()


##################################
# BASE MODEL KURULUMU
##################################

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Accuracy: 0.77
# Recall: 0.706 # pozitif sınıfın ne kadar başarılı tahmin edildiği
# Precision: 0.59 # Pozitif sınıf olarak tahmin edilen değerlerin başarısı
# F1: 0.64
# Auc: 0.75


##################################
# GÖREV 2: FEATURE ENGINEERING
##################################

##################################
# EKSİK DEĞER ANALİZİ
##################################

# Bir insanda Pregnancies ve Outcome dışındaki değişken değerleri 0 olamayacağı bilinmektedir.
# Bundan dolayı bu değerlerle ilgili aksiyon kararı alınmalıdır. 0 olan değerlere NaN atanabilir .

zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

zero_columns

# 2.yol
for i in df.columns:
    print('{} zero values: {}'.format(i, (df[i] == 0).sum()))

# Gözlem birimlerinde 0 olan degiskenlerin her birisine gidip 0 iceren gozlem degerlerini NaN ile değiştirdik.
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

# Eksik Gözlem Analizi
df.isnull().sum()

# eksik veri yapısının incelenmesi
msno.matrix(df)
plt.show()

msno.bar(df)
plt.show()

msno.heatmap(df)
plt.show()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 3)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns, missing_df

na_columns, missing_df = missing_values_table(df, na_name=True)


# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_columns)

#
# Eksik Değerlerin Doldurulması
for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()


df.isnull().sum()

# 2. alternatif: MEDYAN ile doldurma
def replace_na_to_median(dataframe, na_col):
    for j in na_col:
        if (dataframe[j] == 0).any() == True:
            dataframe[j] = dataframe[j].replace(to_replace=0, value=dataframe[j].median())
    print(dataframe.head())


replace_na_to_median(df, ["Insulin", "SkinThickness", "Glucose", "BloodPressure", "BMI"])

# 3. alternatif: diğer değişkenleri bağımsız değ gibi düşünüp missing value'su olan (0 değeri)
# örn Insulin'e bir reg modeli oluşturup, ona göre tahminleme yapılabilir.

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

y = df["Insulin"]
X = df.drop("Insulin", axis=1)
reg_model = LinearRegression().fit(X, y)
y_pred = reg_model.predict(X)

mean_absolute_error(y, y_pred)

y_pred = pd.DataFrame(y_pred).astype(int)
df["y_pred"] = y_pred
df.loc[(df["Insulin"] == 0), "Insulin"]= df["y_pred"]
df["Insulin"]

def replace_na_to_reg(dataframe, na_col):
    for j in na_col:
        y_pred = 0
        if (dataframe[j] == 0).any() == True:
            y = dataframe[j]
            X = dataframe.drop(j, axis=1)
            reg_model = LinearRegression().fit(X, y)
            y_pred = reg_model.predict(X)
            dataframe["y_pred"] = pd.DataFrame(y_pred)
            dataframe.loc[(dataframe[j] == 0), j] = abs(dataframe["y_pred"])
            print(dataframe.head())


replace_na_to_reg(df, ["Insulin", "SkinThickness", "Glucose", "BloodPressure", "BMI"])


##################################
# AYKIRI DEĞER ANALİZİ
##################################

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

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Değer Analizi ve Baskılama İşlemi
for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))

"""
### LOF

clf=LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df_)
df_scores=clf.negative_outlier_factor_
df_scores[0:5]
np.sort(df_scores)[0:8]

scores=pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True,xlim=[0,20],style=".-")
plt.show()

esik_deger=np.sort(df_scores)[5]

df[df_scores<esik_deger].shape

df[df_scores<esik_deger].index

df[df_scores<esik_deger].drop(axis=0,labels=df[df_scores<esik_deger].index)

num_cols=[col for col in df.columns if df[col].dtypes!="O"]
outlier_thresholds(df,num_cols)
check_outlier(df,num_cols)

for i in df.index:
    for esik in df[df_scores < esik_deger].index:
        if i==esik:
            for col in num_cols:
                print(i,col,replace_with_thresholds(df,col))


"""
df["Insulin"] = df["Insulin"].fillna(df.groupby("NEW_GLUCOSE_CAT")["Insulin"].transform("median"))

df.groupby("NEW_GLUCOSE_CAT")["Insulin"].median()

df["Insulin"].isna().sum()

df["NEW_STHICKNESS_BMI"] = df["SkinThickness"] / df["BMI"]
df["NEW_AGE_DPEDIGREE"] = df["Age"] / df["DiabetesPedigreeFunction"]
df["NEW_GLUCOSE_BPRESSURE"] = (df["BloodPressure"] * df["Glucose"])/100

df.loc[(df['BMI'] < 18.5), 'NEW_BMI_CAT'] = "underweight"
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] <= 24.9), 'NEW_BMI_CAT'] = 'normal'
df.loc[(df['BMI'] >= 25) & (df['BMI'] < 30), 'NEW_BMI_CAT'] = 'overweight'
df.loc[(df['BMI'] >= 30), 'NEW_BMI_CAT'] = 'obese'

df.loc[(df['Age'] < 21), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 21) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['Pregnancies'] == 0), 'NEW_PREGNANCY_CAT'] = 'no_pregnancy'
df.loc[(df['Pregnancies'] == 1), 'NEW_PREGNANCY_CAT'] = 'one_pregnancy'
df.loc[(df['Pregnancies'] > 1), 'NEW_PREGNANCY_CAT'] = 'multi_pregnancy'

df.loc[(df['Glucose'] >= 170), 'NEW_GLUCOSE_CAT'] = 'dangerous'
df.loc[(df['Glucose'] >= 105) & (df['Glucose'] < 170), 'NEW_GLUCOSE_CAT'] = 'risky'
df.loc[(df['Glucose'] < 105) & (df['Glucose'] > 70), 'NEW_GLUCOSE_CAT'] = 'normal'
df.loc[(df['Glucose'] <= 70), 'NEW_GLUCOSE_CAT'] = 'low'

df.loc[(df['BloodPressure'] >= 110), 'NEW_BLOODPRESSURE_CAT'] = 'hypersensitive crisis'
df.loc[(df['BloodPressure'] >= 90) & (
        df['BloodPressure'] < 110), 'NEW_BLOODPRESSURE_CAT'] = 'hypertension'
df.loc[(df['BloodPressure'] < 90) & (df['BloodPressure'] > 70), 'NEW_BLOODPRESSURE_CAT'] = 'normal'
df.loc[(df['BloodPressure'] <= 70), 'NEW_BLOODPRESSURE_CAT'] = 'low'

df.loc[(df['Insulin'] >= 160), 'NEW_INSULIN_CAT'] = 'high'
df.loc[(df['Insulin'] < 160) & (df['Insulin'] >= 16), 'NEW_INSULIN_CAT'] = 'normal'
df.loc[(df['Insulin'] < 16), 'NEW_INSULIN_CAT'] = 'low'

# Kolonların büyültülmesi
df.columns = [col.upper() for col in df.columns]

df.head()

##################################
# ENCODING
##################################

# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.dtypes

# LABEL ENCODING
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


#label_encoder.inverse_transform([0,1])

for col in binary_cols:
    df = label_encoder(df, col)

# One-Hot Encoding İşlemi
# cat_cols listesinin güncelleme işlemi
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
##################################
# STANDARTLAŞTIRMA
##################################

num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

#### Feature Engineering fonksiyonlaştırılmış hali ####
def diabetes_data_prep(dataframe):
    # FEATURE ENGINEERING
    dataframe["NEW_STHICKNESS_BMI"] = dataframe["SkinThickness"] / dataframe["BMI"]
    dataframe["NEW_AGE_DPEDIGREE"] = dataframe["Age"] / dataframe["DiabetesPedigreeFunction"]
    # dataframe["NEW_GLUCOSE_BPRESSURE"] = (dataframe["BloodPressure"] * dataframe["Glucose"])/100

    dataframe.loc[(dataframe['BMI'] < 18.5), 'NEW_BMI_CAT'] = "underweight"
    dataframe.loc[(dataframe['BMI'] >= 18.5) & (dataframe['BMI'] <= 24.9), 'NEW_BMI_CAT'] = 'normal'
    dataframe.loc[(dataframe['BMI'] >= 25) & (dataframe['BMI'] < 30), 'NEW_BMI_CAT'] = 'overweight'
    dataframe.loc[(dataframe['BMI'] >= 30), 'NEW_BMI_CAT'] = 'obese'

    dataframe.loc[(dataframe['Age'] < 21), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['Age'] >= 21) & (dataframe['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

    dataframe.loc[(dataframe['Pregnancies'] == 0), 'NEW_PREGNANCY_CAT'] = 'no_pregnancy'
    dataframe.loc[(dataframe['Pregnancies'] == 1), 'NEW_PREGNANCY_CAT'] = 'one_pregnancy'
    dataframe.loc[(dataframe['Pregnancies'] > 1), 'NEW_PREGNANCY_CAT'] = 'multi_pregnancy'

    dataframe.loc[(dataframe['Glucose'] >= 170), 'NEW_GLUCOSE_CAT'] = 'dangerous'
    dataframe.loc[(dataframe['Glucose'] >= 105) & (dataframe['Glucose'] < 170), 'NEW_GLUCOSE_CAT'] = 'risky'
    dataframe.loc[(dataframe['Glucose'] < 105) & (dataframe['Glucose'] > 70), 'NEW_GLUCOSE_CAT'] = 'normal'
    dataframe.loc[(dataframe['Glucose'] <= 70), 'NEW_GLUCOSE_CAT'] = 'low'

    dataframe.loc[(dataframe['BloodPressure'] >= 110), 'NEW_BLOODPRESSURE_CAT'] = 'hypersensitive crisis'
    dataframe.loc[(dataframe['BloodPressure'] >= 90) & (
                dataframe['BloodPressure'] < 110), 'NEW_BLOODPRESSURE_CAT'] = 'hypertension'
    dataframe.loc[
        (dataframe['BloodPressure'] < 90) & (dataframe['BloodPressure'] > 70), 'NEW_BLOODPRESSURE_CAT'] = 'normal'
    dataframe.loc[(dataframe['BloodPressure'] <= 70), 'NEW_BLOODPRESSURE_CAT'] = 'low'

    dataframe.loc[(dataframe['Insulin'] >= 160), 'NEW_INSULIN_CAT'] = 'high'
    dataframe.loc[(dataframe['Insulin'] < 160) & (dataframe['Insulin'] >= 16), 'NEW_INSULIN_CAT'] = 'normal'
    dataframe.loc[(dataframe['Insulin'] < 16), 'NEW_INSULIN_CAT'] = 'low'

    ### INSULIN için alternatif fonksiyon: assign ve apply fonk yardımıyla!!
    # def set_insulin(df, col = 'Insulin'):
    #     if df[col] >= 16 and df[col] <= 160:
    #        return "normal"
    #     else:
    #        return "abnormal"

    # df["NEW_INSULIN"] = df.apply(set_insulin, axis=1)              # yeni değişkeni bu şekilde de atayabiliriz
    # df = df.assign(NEW_INSULIN2 = df.apply(set_insulin, axis=1)    # yeni değişkeni assign ile de atayabiliriz
    # df[["NEW_INSULIN", "NEW_INSULIN2"]]

    dataframe.columns = [col.upper() for col in dataframe.columns]

    # AYKIRI GOZLEM_OUTLIERS
    num_cols = [col for col in dataframe.columns if len(dataframe[col].unique()) > 20
                and dataframe[col].dtypes != 'O']

    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    # for col in num_cols:
    #    print(col, check_outlier(df, col))
    # print(check_df(df))

    # LABEL ENCODING

    binary_cols = [col for col in dataframe.columns if
                   len(dataframe[col].unique()) == 2 and dataframe[col].dtypes == 'O']

    # label_encoding ile binary olan değişkenleri 1-0 olarak değiştirir
    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)


    dataframe = rare_encoder(dataframe, 0.10)

    # one hot encoding de dummy değişken tuzağına düşmemek için drop_first diyerek ilek oluşan değişkeni drop eder
    ohe_cols = [col for col in dataframe.columns if 10 >= len(dataframe[col].unique()) > 2 and dataframe[col].dtypes == 'O']
    dataframe = one_hot_encoder(dataframe, ohe_cols, drop_first=True)

    #standartlaştırma
    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    return dataframe

df_prep = diabetes_data_prep(df_)

#################

df_prep.to_pickle("prepared_diabetes_df.pkl")

df = pd.read_pickle("prepared_diabetes_df.pkl")

df_prep.to_csv("prepared_diabetes_df.csv")

### MODEL ###

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Accuracy: 0.77
# Recall: 0.727
# Precision: 0.64
# F1: 0.68
# Auc: 0.76

def plot_importance(model, X, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': X.columns})
    plt.figure(figsize=(10, 15))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[1:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    # plt.savefig('importances-01.png')
    plt.show()


plot_importance(rf_model, X)



