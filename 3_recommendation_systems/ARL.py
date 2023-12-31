#!pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)

pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")

df_ = pd.read_csv("armut_data.csv")
df = df_.copy()

def check_df(df, head=5):
    print("##################### Shape #####################")
    print(df.shape)
    print("##################### Types #####################")
    print(df.dtypes)
    print("##################### Head #####################")
    print(df.head(head))
    print("##################### Tail #####################")
    print(df.tail(head))
    print("##################### NA #####################")
    print(df.isnull().sum())
    print("##################### Quantiles #####################")
    print(df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("##################### Info #####################")
    print(df.info)


check_df(df)

#Creating New Varriables

df["Hizmet"] = df['ServiceId'].astype(str) +"_"+ df["CategoryId"].astype(str)
df.head()

df["SepetID"] = df['UserId'].astype(str) +"_"+ df["CreateDate"].astype(str).str[:7]
df.head()

#Creating The Invoince Product Table

df.pivot_table(columns=["Hizmet"], index=["SepetID"], values=["ServiceId"], aggfunc="count").head()

#Creating Rules

invoice_product_df = df.pivot_table(columns=["Hizmet"],
                                    index=["SepetID"],
                                    values=["ServiceId"],
                                    aggfunc="count").fillna(0).applymap(lambda x: 1 if x > 0 else 0)

invoice_product_df.head()

#Drop the title, ServisID

invoice_product_df.columns = invoice_product_df.columns.droplevel(0)

frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
frequent_itemsets.shape

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()

sorted_rules = rules.sort_values("lift", ascending=False)
sorted_rules.head()

def arl_recommender(rules_df, product_id, rec_count=1):
    recommendation_list = []
    sorted_rules = rules.sort_values("lift", ascending=False)
    for idx, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
              recommendation_list.append(list(sorted_rules.iloc[idx]["consequents"])[0])
    return recommendation_list[:rec_count]

arl_recommender(rules, "5_2", 7)
