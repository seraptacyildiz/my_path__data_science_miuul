##################################################
# List Comprehensions
##################################################

# ###############################################
# # Goal 1: Using the List Comprehension structure, capitalize the names of the numeric variables in the car_crashes data and add NUM to the beginning.
# ###############################################
#
# # Expected Output
#
# # ['NUM_TOTAL',
# #  'NUM_SPEEDING',
# #  'NUM_ALCOHOL',
# #  'NUM_NOT_DISTRACTED',
# #  'NUM_NO_PREVIOUS',
# #  'NUM_INS_PREMIUM',
# #  'NUM_INS_LOSSES',
# #  'ABBREV']
#
# # Notes:
# # Non-numeric names should also grow.
# # It should be done with a single list comp structure.


import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("car_crashes")
df.columns
df.info()

dene=["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]
["NUM_" + col.upper() if df[col].dtype != "object" else col.upper() for col in df.columns]

df["speeding"].dtype

# ###############################################
# # Goal 2: Using the List Comprehension structure, write "FLAG" after the names of the variables that do not contain "no" in their names in the car_crashes data.
# ###############################################
#
# # Notes:
# # All variable names must be uppercase.
# # It should be done with a single List Comprehension.
#
# # Expected Output
#
# # ['TOTAL_FLAG',
# #  'SPEEDING_FLAG',
# #  'ALCOHOL_FLAG',
# #  'NOT_DISTRACTED',
# #  'NO_PREVIOUS',
# #  'INS_PREMIUM_FLAG',
# #  'INS_LOSSES_FLAG',
# #  'ABBREV_FLAG']


[col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]

[col.upper() if "no" in col else col.upper() + "_FLAG" for col in df.columns]

# ###############################################
# # Goal 3: Using the List Comprehension structure, select the names of the variables that are DIFFERENT from the variable names given below and create a new dataframe.
# ###############################################
#
og_list = ["abbrev", "no_previous"]

#
# # Notes:
# # First, create a new list named new_cols using list comprehension according to the list above.
# # Then create a new df by selecting these variables with df[new_cols] and name it new_df.
#
# # Expected Output
#
# #    total  speeding  alcohol  not_distracted  ins_premium  ins_losses
# # 0 18.800     7.332    5.640          18.048      784.550     145.080
# # 1 18.100     7.421    4.525          16.290     1053.480     133.930
# # 2 18.600     6.510    5.208          15.624      899.470     110.350
# # 3 22.400     4.032    5.824          21.056      827.340     142.390
# # 4 12.000     4.200    3.360          10.920      878.410     165.630
#

og_list = ["abbrev", "no_previous"]
new_cols = [col for col in df.columns if col not in og_list]
new_df = df[new_cols]
new_df1 = df[['total', 'speeding', 'alcohol', 'not_distracted', 'ins_premium', 'ins_losses']]
new_df1.head()

new_cols = [col for col in df.columns if df[col].any() not in og_list]
df = df[new_cols]
df.head()

dict1={"ali":1,"veli":2,"c":3,"den":4}
{k:v**2 for (k,v) in dict1.items()}

{k.upper():v**2 for (k,v) in dict1.items()}



