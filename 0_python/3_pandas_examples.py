
##################################################
# Pandas examples
##################################################
import numpy as np
import seaborn as sns
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Goal 1: Define the "titanic" dataset from the Seaborn library.
#########################################

df = sns.load_dataset("titanic")
df.head()

#########################################
# Goal 2: Find the number of male and female passengers in the Titanic dataset described above.
#########################################
df["sex"].value_counts()

#alt2:
df.groupby("sex").agg({"sex": ["count"]})


#########################################
# Goal 3: Find the number of unique values for each column.
#########################################
for col in df.columns:
    print(f"{col} : {df[col].nunique()}")

#alt2:
df.nunique()

#alt3:
[f"{col} : {df[col].nunique()}" for col in df.columns ]


#########################################
# Goal 4: Find the unique values of the "pclass" variable.
#########################################

df["pclass"].unique()

#########################################
# Goal 5:  Find the number of unique values of "pclass" and "patch" variables.
#########################################

df[["pclass", "parch"]].nunique()

#########################################
# Goal 6: Check the type of the "embarked" variable. Change its type to category. Check the repetition type.
#########################################

df.info()

df["embarked"] = df["embarked"].astype("category")

#########################################
# Goal 7: Show all information for those with an "embarked" value of "C".
#########################################

df[df["embarked"] == "C"].head()


#########################################
# Goal 8: Show all information for those whose "embarked" value is not "S".
#########################################
df[df["embarked"] != "S"].head()


#alt2:
df[(df["embarked"] == "C") | (df["embarked"] == "Q")].head()

#alt3:
df[~(df["embarked"] == "S")].head()

#alt4:
df.loc[df["embarked"] != "S"]

#########################################
# Goal 9: Show all information for passengers younger than 30 years old and female.
#########################################
df[(df["age"] < 30) & (df["sex"] == "female")].head()



#########################################
# Goal 10: Show information for passengers whose Fare is over 500 or 70 years of age.
#########################################
df[(df["fare"] > 500) | (df["age"] > 70)].head()

#########################################
# Goal 11: Find the sum of the null values in each variable.
#########################################
df.info()
df.isnull().sum() #returns each variable separately.
df.isnull().sum().sum() #gives the sum of all.

#########################################
# Görev 12: Drop the "who" variable from the dataframe.
#########################################
df = df.drop("who", axis=1)
df.head()

#alt2:
df.drop("who", axis=1, inplace=True)

#########################################
# Goal 13: Fill in the empty values in the deck variable with the most repeated value (mode) of the deck variable.
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
# Goal 14: Fill the empty values in the age variable with the median of the age variable.
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
# Goal 15: Find the sum, count, mean values of the survived variable in the breakdown of the Pclass and Sex variables.
#########################################

agg_list = ["sum", "count", "mean"]
df.groupby(["pclass", "sex"]).agg({"survived" : agg_list})

#alt2:
df.groupby(["sex", "pclass"]).agg({"survived": ["mean", "sum", "count"]})

#alt3:
df.pivot_table("survived", ["pclass", "sex"], aggfunc=["sum", "count", "mean"])

#########################################
# Goal 16:  Write a function that returns 1 for those under 30 and 0 for those above or equal to 30.
# Using the function you wrote, create a variable named age_flag in the titanic data set. (use apply and lambda constructs)
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
# Goal 17: Define the "Tips" dataset from the Seaborn library.
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
# Goal 18: Find the sum, min, max and average of the total_bill values according to the categories (Dinner, Lunch) of the time variable.
#########################################
agg_list = ["sum", "min", "max", "mean"]
df.groupby(["time"]).agg({"total_bill": agg_list})

#alt2:
df.groupby(["time"]).agg({"total_bill": ["sum", "min", "max", "mean"]})

#alt3:
df.pivot_table("total_bill", "time", aggfunc=["sum", "min", "max", "mean"])

#########################################
# Goal 19: Find the sum, min, max and average of total_bill values by days and time.
#########################################
agg_list = ["sum", "min", "max", "mean"]
df.groupby(["day", "time"]).agg({"total_bill": agg_list})

#alt2:
df.groupby(["day", "time"]).agg({"total_bill": ["sum", "min", "max", "mean"]})

#alt3:
df.pivot_table("total_bill", ["day", "time"], aggfunc=["sum","min", "max", "mean"])
df.pivot_table("total_bill", "day", "time", aggfunc=["sum","min", "max", "mean"])

#########################################
# Goal 20: Find the sum, min, max and average of the total_bill and type values of the lunch time and female customers according to the day.
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
# Goal 21: What is the average of orders with size less than 3 and total_bill greater than 10?
#########################################

df.loc[(df["size"] < 3) & (df["total_bill"] > 10)].mean()

#alt2:
df[(df["size"] < 3) & (df["total_bill"] > 10)].mean()


#########################################
# Goal 22: Create a new variable called total_bill tip_sum. Let him give the sum of the total bill and tip paid by each customer.
#########################################

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()

df["total_bill_tip_sum"].sum()

#########################################
# Goal 23: Sort from largest to smallest according to the total_bill_tip_sum variable and assign the first 30 people to a new dataframe.
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
