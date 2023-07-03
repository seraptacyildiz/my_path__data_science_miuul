##############################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma
##############################################################

###############################################################
# Business Problem
###############################################################
# The company FLO wants to set a roadmap for sales and marketing activities.
# In order for the company to make a medium-long-term plan, it is necessary to estimate the potential value that existing customers will provide to the company in the future.


###############################################################
# Story of Dataset
###############################################################

# The dataset consists of information obtained from the past shopping behaviors of customers who made their last purchases 
# as OmniChannel (both online and offline shopper) in 2020 - 2021.

# master_id: Unique customer number
# order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : The channel where the most recent purchase was made
# first_order_date :Date of first purchase made by the customer
# last_order_date : Date of last purchase made by the customer
# last_order_date_online : The date of the last purchase made by the customer on the online platform
# last_order_date_offline : The date of the last purchase made by the customer on the offline platform
# order_num_total_ever_online : The total number of purchases made by the customer on the online platform
# order_num_total_ever_offline : The total number of purchases made by the customer on the offline platform
# customer_value_total_ever_offline : Total fee paid by the customer for offline purchases
# customer_value_total_ever_online : The total fee paid by the customer for their online shopping
# interested_in_categories_12 : List of categories the customer has shopped in the last 12 months


###############################################################
# GOALS
###############################################################
# Goal 1: Prepare data
            # 1. Read the data flo_data_20K.csv. Make a copy of the dataframe.
            # 2. Define the outlier_thresholds and replace_with_thresholds functions needed to suppress outliers.
            # Note: When calculating cltv, frequency values must be integers. Therefore, round the lower and upper limits with round().
            # 3. Suppress if the variables "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" 
            # have outliers.
            # 4. Omnichannel means that customers shop from both online and offline platforms. 
            # Create new variables for each customer's total purchases and spending.
            # 5. Examine the variable types. Change the type of variables that express date to "date".

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

df_ = pd.read_csv("D:\FLOMusteriSegmentasyonu\flo_data_20k.csv")
df = df_.copy()
df.head(10)
df.columns
df.describe().T
df.isnull().sum()
df.info()
df["master_id"].nunique()

   # 2. Define the outlier_thresholds and replace_with_thresholds functions needed to suppress outliers.

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


 # 3. Suppress if the variables "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" 
 # have outliers.
replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")
df.head()

#alternatif
#numeric_cols = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]
#for col in numeric_cols:
 #   replace_with_thresholds(df, col)



 # 4. Omnichannel means that customers shop from both online and offline platforms. 
 # Create new variables for each customer's total purchases and spending.


df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head()

# 5. Examine the variable types. Change the type of variables that express date to "date".

for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])
df.info()



# Goal 2: Creating the Data Structure CLTV
            # 1. Take 2 days after the date of the last purchase in the data set as the date of analysis.
            # 2. Create a new cltv dataframe with customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg values.
            # Monetary value will be expressed as average value per purchase, recency and tenure values will be expressed in weekly terms.


# 1) recency: Time since last purchase. Weekly. (user specific)
# 2) T: Customer's age. Weekly. (how long before the analysis date the first purchase was made)
# 3) frequency: total number of repeat purchases (frequency>1)
# 4) monetary: average earnings per purchase


df["last_order_date"].max()   # Timestamp('2021-05-30 00:00:00')
today_date = dt.datetime(2021, 6, 1)
cltv_df = df[["master_id", "last_order_date", "first_order_date", "total_order", "total_value"]]

cltv_df.rename(columns={"master_id": "customer_id"}, inplace=True)

# 1) recency:
cltv_df["recency_weekly"] = (cltv_df["last_order_date"] - cltv_df["first_order_date"]).dt.days/7
# 2) T: Customer's age
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




# GÖREV 3: BG/NBD, Gamma-Gamma models, calculation of CLTV
           # 1. Please fit BG/NBD model. #of purchase #Beta Geometric / Negative Binomial Distribution
                # a. Estimate expected purchases from customers in 3 months and add exp_sales_3_month to cltv dataframe.
                # b. Estimate expected purchases from customers in 6 months and add exp_sales_6_month to cltv dataframe.

# 1. Please fit BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency_weekly"],
        cltv_df["T_weekly"])

# a = 3 Months

cltv_df["sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency_weekly"],
                                                        cltv_df["T_weekly"])
# b = 6 Months

cltv_df["sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency_weekly"],
                                                        cltv_df["T_weekly"])
cltv_df.head()


            # average purchase per order
            # 2. Fit the Gamma-Gamma model. 
            # Estimate the average value of the customers and add it to the cltv dataframe as exp_average_value.

# 2. Fit the Gamma-Gamma model. 
ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_avg"])
cltv_df.head()
cltv_df.sort_values("exp_average_value", ascending=False).head()

cltv_df.head()




# Cltv expectation
           # 3. Calculate 6 months CLTV and add it to the dataframe with the name cltv.
cltv_df["clv"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency_weekly"],
                                   cltv_df["T_weekly"],
                                   cltv_df["monetary_avg"],
                                   time=6,  # 6 months
                                   freq="W",  # T's time information (weekly)
                                   discount_rate=0.01)
cltv_df.head(10)

                # b. Check the 20 people with the highest Cltv value.

cltv_df.sort_values("clv", ascending=False).head(20)




# Goal 4: Creating Segments by CLTV
           # 1. Divide all 6-month-old customers into 4 groups (segments) and add the group names to the data set. Add it to the dataframe with the name cltv_segment.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["clv"], 4, labels=["D", "C", "B", "A"])
cltv_df.sort_values("clv", ascending=False).head(20)
cltv_df["cltv_segment"].value_counts()
cltv_df.groupby("cltv_segment").agg({"clv": ["mean", "min", "max"]})
cltv_df.head()
           # 2. Make short 6-month action suggestions to the management for 2 groups that you will choose from among 4 groups.



###############################################################
# BONUS: Functionalize the whole process.
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




