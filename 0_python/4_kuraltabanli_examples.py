import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import seaborn as sns
import plotly.express as px


df = pd.read_csv("persona.csv")
# Let's see the data' first 5 row to understand the data variables
df.head()

# Describe method shows the numerical data's summary. Count, mean, standart deviation,
# maximum, minimum and outliers can be seen below

df.describe().T

# Let's try to understand the data.

df.info()
df.columns
df.nunique()

# As we can see, dataset includes 5 different variables.
# Three of them is category and the other two of them is look like numerical variables.
# However, PRICE column has 6 different values, because of that it can be call as "numerical but categorical" variable.



# Let's define the function to seperate the numerical, categorical and cardinal columns in the dataset.

def grab_col_names(df, cat_th=10, car_th=20):

    cat_col = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and str(df[col].dtypes) in ["float64", "int64"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_col = cat_col + num_but_cat

    cat_col = [col for col in cat_col if col not in cat_but_car]

    num_col = [col for col in df.columns if col not in cat_col]

    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f"cat_cols: {len(cat_col)}")
    print(f"num_cols: {len(num_col)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat {len(num_but_cat)}")

    return cat_col, num_col, cat_but_car

cat_col, num_col, cat_but_car = grab_col_names(df)

# In this part, function is defined to describe and understand the categorical variables

def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#######################")
    if plot:
        sns.countplot(x=dataframe[col_name], data = dataframe)
        plt.show(block=True)

for col in cat_col:
    cat_summary(df, col, plot = True)



# In this part, function is defined to describe and understand the numerical variables

def num_summary(dataframe, num_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    print(dataframe[num_col].describe(quantiles).T)
    if plot == True:
        dataframe[num_col].hist()
        plt.xlabel(num_col)
        plt.title(num_col)
        plt.show(block=True)

for col in num_col:
    num_summary(df, col, plot=True)

df.groupby("SOURCE").agg({"PRICE": "sum"})
df.groupby("SOURCE").agg({"PRICE": "mean"})

# Total price for each country
df.groupby("COUNTRY").agg({"PRICE": "sum"})
df.groupby("COUNTRY").agg({"PRICE": "mean"})

# Total price for each sources
df.groupby(["SOURCE", "COUNTRY"]).agg({"PRICE": "mean"})

# Here is the dataframe is arranged according to mean of the Price and sorted also according to Price values

agg_df = df.groupby(["COUNTRY","SOURCE","SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending= False)
agg_df = agg_df.reset_index()
agg_df


# To make categorical age description in the dataframe, AGE values seperated by the cut() function
# and defined as 5 different ranges as categorical.

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], [0, 18, 23, 30, 40, 70], labels=['0_18', '19_23', '24_30','31_40', '41_70'])

agg_df["AGE_CAT"] = agg_df["AGE_CAT"].astype("O")

# To summarize the dataset, COUNTRY, SOURCE, SEX and AGE_CAT columns concatenate to define de user's definitions
# in only a column to readable of the dataset.


customers_level_based = pd.DataFrame(
    [agg_df["COUNTRY"][x] + "_" + agg_df["SOURCE"][x] + "_" + agg_df["SEX"][x] + "_" + agg_df["AGE_CAT"][x]
     for x in range(len(agg_df))])
customers_level_based = pd.DataFrame(customers_level_based)
customers_level_based.columns = ["customers_level_based"]
customers_level_based["customers_level_based"] = customers_level_based["customers_level_based"].str.upper()
customers_level_based["PRICE"] = agg_df["PRICE"]  # Mean Price

customers_level_based.groupby(["customers_level_based"]).agg({"PRICE": "mean"})

agg_df = customers_level_based.groupby(["customers_level_based"]).agg({"PRICE": "mean"})
agg_df2 = agg_df.reset_index()


# According to latest definition above, SEGMENT column will be added to new dataframe as categorical cutting of the mean Price values
segment = pd.qcut(agg_df2["PRICE"], 4, labels = ["D","C","B","A"])

agg_df2["SEGMENT"] = segment

# Mean, max and sum of the Segment distrubition can be seen below
agg_df3 = agg_df2.groupby(["SEGMENT"]).agg({"PRICE": ["mean", "max", "sum"]})

agg_df3


# And the main purpose of the study, we can see the new users Segments, and total Price values that they spend for their
# phone' according to their sex, country, source and age'.

new_user_1 = "TUR_ANDROID_FEMALE_31_40"
agg_df2[agg_df2["customers_level_based"] == new_user_1]
print(agg_df2[agg_df2["customers_level_based"] == new_user_1])

new_user_2 = "FRA_IOS_FEMALE_31_40"
agg_df2[agg_df2["customers_level_based"] == new_user_2]
print(agg_df2[agg_df2["customers_level_based"] == new_user_2])
