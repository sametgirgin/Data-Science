
#########################
# Business Problem
#########################
# Turkey’s largest online service platform, Armut, connects service providers with customers
# who want to receive services. With just a few taps on a computer or smartphone, users can
# easily access services such as cleaning, repair, moving, and more.
# Using a dataset that includes users who have received services and the categories of those services,
# an Association Rule Learning–based product recommendation system is intended to be created.

#########################
# Dataset
#########################
#The dataset consists of the services received by customers and the categories of those services.
# Each received service includes date and time information.

# UserId: Customer ID
# ServiceId: Anonymized services belonging to each category. (Example: A sofa cleaning service under the cleaning category)
# A single ServiceId may appear under different categories and represent different services depending on the category.
# (Example: The service with CategoryId = 7 and ServiceId = 4 refers to radiator cleaning, while the same ServiceId = 4 under CategoryId = 2 refers to furniture assembly)
# CategoryId: Anonymized categories. (Example: Cleaning, moving, renovation)
# CreateDate: The date the service was purchased

#########################
# TASK 1: Data Preparation
#########################

# Step 1: armut_data.csv dosyasınız okutunuz.
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

#warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 500)

armut_df = pd.read_csv("ArmutARL/armut_data.csv")
armut_df.head()


def check_dataframe(df, head=5):
    print("Shape of the DataFrame:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

    print("Data Types:")
    print(df.dtypes, "\n")

    print("Missing Values:")
    print(df.isnull().sum(), "\n")

    print("Descriptive Statistics (Numerical Features):")
    print(df.describe().T, "\n")

    print("Unique Value Counts (for each column):")
    for col in df.columns:
        unique_vals = df[col].nunique()
        print(f"{col}: {unique_vals} unique values")

check_dataframe(armut_df)
#Step 2: The ServiceID represents a different service for each CategoryID.
# Create a new variable representing the services by combining the ServiceID and CategoryID with an underscore ("_").
armut_df["Service"] = armut_df["ServiceId"].astype(str) + "_" + armut_df["CategoryId"].astype(str)


#Step 3: The dataset consists of the date and time when the services were received, and does not include any basket definition (like an invoice, etc.).
#To apply Association Rule Learning, a basket definition (such as an invoice) must be created.
#Here, the basket is defined as the monthly services received by each customer. For example:
    #The services 9_4 and 46_4 received by customer with ID 7256 in August 2017 form one basket.
    #The services 9_4 and 38_4 received by the same customer in October 2017 form another basket.
#Each basket needs to be identified with a unique ID.
#To do this, first create a new date variable that contains only the year and month. Then, combine the UserID and the newly created date variable using an underscore ("_"), and assign this to a new variable named ID.

armut_df["NewDate"] = pd.to_datetime(armut_df["CreateDate"]).dt.to_period("M")
armut_df["BasketID"] = armut_df["UserId"].astype(str) + "_" + armut_df["NewDate"].astype(str)

#armut_df.sort_values(by='BasketID')
#########################
# TASK 2: Create Association Rules
#########################

# Step 1: Create a pivot table basket vs service like below.

# Service         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# BasketID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

basket_service_df = armut_df.groupby(['BasketID', 'Service'])['Service'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

#Step-2: Create association rules.
#The apriori() function finds frequent itemsets (product combinations) in the given dataset.
#min_support=0.01: Only consider combinations that occur in at least 1% of the data.
frequent_itemsets = apriori(basket_service_df, min_support=0.01, use_colnames=True)

#Sort the frequent product combinations by their support value.
#You can see the most commonly co-occurring products at the top.
frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

#Step 3:Use the arl_recommender function to recommend services to a user who most recently received
# the service 2_0.
def arl_recommender(rules_df, service, rec_count=1):
    # Sort the rules based on lift in descending order
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    # Filter the rules where the antecedent contains the given service
    filtered_rules = sorted_rules[sorted_rules["antecedents"].apply(lambda x: service in x)]
    # Extract the recommended services from the filtered rules
    recommendation_list = [', '.join(rule) for rule in filtered_rules["consequents"].apply(list).tolist()[:rec_count]]
    print("Other recommended services for the service you entered: ", recommendation_list)

arl_recommender(rules, '2_0', 2)




