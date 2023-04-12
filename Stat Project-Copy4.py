#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing libraries
import sqlite3
import pandas as pd
import seaborn as sns
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math as math
from scipy.stats import ttest_ind


# In[3]:


#Setting float format
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# In[4]:


data = pd.read_csv('bigmart_product.csv')
data.head()


# In[5]:


data.info()


# In[6]:


data['Item_Weight'].isna().sum()/data.index.max()


# In[7]:


data['Outlet_Size'].isna().sum()/data.index.max()


# In[8]:


@dataclass
class Cleaning:
    df : pd.DataFrame
        
    #Using upper and lower boundaries to define outliers and change their value using median
    def impute_outliers(self):
        columns = list(self.df.columns.values)
        for column in columns:
            try:
                q3 = self.df[column].quantile(q=0.75)
                q1 = self.df[column].quantile(q=0.25)
                iqr = q3 - q1
                upper_limit = q3 + iqr*1.5
                lower_limit = q1 - iqr*1.5
                median = self.df[column].median()
                self.df.loc[self.df[column]>upper_limit, column] = median
                self.df.loc[self.df[column]<lower_limit, column] = median
            except:
                pass
            
    #Change non-float null into 'unknown' and float null into median
    def impute_missing(self):
        columns = list(self.df.columns.values)
        for column in columns:
            try:
                if self.df[column].dtypes.name in ('float64', 'int34', 'int64'):
                    pass
                else:
                    self.df[column] = self.df[column].fillna('unknown')
            except:
                pass
    
    #Drop duplicates
    def del_duplicate(self):
        self.df = self.df.drop_duplicates(keep='first')


# In[9]:


clean_data = Cleaning(data)
clean_data.impute_missing()
data.info()


# In[10]:


data.head()


# In[11]:


data = data.dropna()


# In[12]:


data.info()


# In[13]:


clean_data.del_duplicate()


# In[14]:


data.head()


# In[15]:


data.describe()


# In[16]:


data.info()


# In[17]:


data.head()


# In[18]:


#Found some inconsistencies in Item_Fat_Conteng
set(list(data['Item_Fat_Content']))


# In[19]:


#Replacing values
data['Item_Fat_Content'] = data['Item_Fat_Content'].str.replace('LF', 'Low Fat', regex=True)
data['Item_Fat_Content'] = data['Item_Fat_Content'].str.replace('low fat', 'Low Fat', regex=True)
data['Item_Fat_Content'] = data['Item_Fat_Content'].str.replace('reg', 'Regular', regex=True)
set(list(data['Item_Fat_Content']))


# In[20]:


set(list(data['Item_Type']))


# In[21]:


set(list(data['Outlet_Size']))


# In[22]:


set(list(data['Outlet_Location_Type']))


# In[23]:


set(list(data['Outlet_Type']))


# In[24]:


data.head()


# In[25]:


set(list(data['Outlet_Establishment_Year']))


# In[26]:


data.info()


# In[27]:


encoded_data = data
encoded_data.head()


# In[28]:


set(list(encoded_data['Outlet_Size'].values))


# In[29]:


set(list(encoded_data['Outlet_Location_Type'].values))


# In[30]:


#Encoding categorical columns into numerical ones to allow more statistical works
mapper = {'Item_Fat_Content':{'Low Fat':1, 'Regular':2},
          'Outlet_Size':{'Small':1, 'Medium':2, 'High':3},
         'Outlet_Location_Type':{'Tier 1':1, 'Tier 2':2, 'Tier 3':3},
         'Outlet_Type':{'Grocery Store':1, 'Supermarket Type1':2, 'Supermarket Type2':3, 'Supermarket Type3':4}}


# In[31]:


encoded_data = encoded_data.replace(mapper)
encoded_data.head()


# In[32]:


#Adding a Sales_Amount using Item_Outlet_Sales (sales frequency) * Item_MRP
encoded_data['Sales_Amount'] = encoded_data['Item_Outlet_Sales']*encoded_data['Item_MRP']
encoded_data.head()


# In[33]:


sales_amount_sum = encoded_data['Sales_Amount'].sum()
sales_amount_sum


# In[34]:


sales_amount_mean = sales_amount_sum/encoded_data['Item_Outlet_Sales'].sum()
sales_amount_mean


# In[35]:


sales_amount_var = ((encoded_data['Sales_Amount'] - sales_amount_mean)**2).sum()/(encoded_data['Item_Outlet_Sales'].sum())
sales_amount_var


# In[36]:


sales_amount_std = np.sqrt(sales_amount_var)
sales_amount_std


# In[37]:


#Creating a probability distribution using Outlet_Type and Outlet_Location_Type as two indicators
item_dist_by_type_and_loc = encoded_data.pivot_table(values='Item_Outlet_Sales', index='Outlet_Location_Type', columns='Outlet_Type', 
                 aggfunc= lambda x: round(sum(x)/sum(encoded_data['Item_Outlet_Sales']) * 100, 2), 
                 fill_value=0, margins=True)
item_dist_by_type_and_loc


# In[38]:


#Converting table into percentage values
outlet_type = item_dist_by_type_and_loc.loc[:,item_dist_by_type_and_loc.columns!='All'].drop('All').index
outlet_loc = item_dist_by_type_and_loc.loc[:,item_dist_by_type_and_loc.columns!='All'].drop('All').columns
outlet_weight = item_dist_by_type_and_loc.loc[:,item_dist_by_type_and_loc.columns!='All'].drop('All').values/100


# In[39]:


outlet_type


# In[40]:


outlet_loc


# In[41]:


outlet_weight


# In[42]:


mult_table = [[i*j for j in outlet_type] for i in outlet_loc]
mult_table


# In[43]:


#building an array of X (Outlet_Type) and Y (Outlet_Location_Type) to allow for a multiplication with outlet_weight
np.asarray(mult_table)


# In[44]:


#Expected value gained from outlet_weight * array of X and Y
expected_value = (outlet_weight * np.asarray(mult_table)).sum()
expected_value


# In[45]:


frequency_by_type = encoded_data[['Item_Type', 'Item_Outlet_Sales']].groupby('Item_Type').sum().sort_values('Item_Outlet_Sales', ascending=False)
frequency_by_type


# In[46]:


list(frequency_by_type.head(3).index)


# In[47]:


#Distribution of Snack Foods
item_type = 'Snack Foods'
item_dist_by_type_and_loc_per_item = encoded_data[encoded_data['Item_Type']==item_type].pivot_table(values='Item_Outlet_Sales', index='Outlet_Location_Type', columns='Outlet_Type', 
                 aggfunc= lambda x: round(sum(x)/sum(encoded_data[encoded_data['Item_Type']==item_type]['Item_Outlet_Sales']) * 100, 2), 
                 fill_value=0, margins=True)
item_dist_by_type_and_loc_per_item


# In[48]:


#Distribution of Fruits and Vegetables
item_type = 'Fruits and Vegetables'
item_dist_by_type_and_loc_per_item = encoded_data[encoded_data['Item_Type']==item_type].pivot_table(values='Item_Outlet_Sales', index='Outlet_Location_Type', columns='Outlet_Type', 
                 aggfunc= lambda x: round(sum(x)/sum(encoded_data[encoded_data['Item_Type']==item_type]['Item_Outlet_Sales']) * 100, 2), 
                 fill_value=0, margins=True)
item_dist_by_type_and_loc_per_item


# In[49]:


#Distribution of Household
item_type = 'Household'
item_dist_by_type_and_loc_per_item = encoded_data[encoded_data['Item_Type']==item_type].pivot_table(values='Item_Outlet_Sales', index='Outlet_Location_Type', columns='Outlet_Type', 
                 aggfunc= lambda x: round(sum(x)/sum(encoded_data[encoded_data['Item_Type']==item_type]['Item_Outlet_Sales']) * 100, 2), 
                 fill_value=0, margins=True)
item_dist_by_type_and_loc_per_item


# In[50]:


outlet_type_per_item = item_dist_by_type_and_loc_per_item.loc[:,item_dist_by_type_and_loc_per_item.columns!='All'].drop('All').index
outlet_loc_per_item = item_dist_by_type_and_loc_per_item.loc[:,item_dist_by_type_and_loc_per_item.columns!='All'].drop('All').columns
outlet_weight_per_item = item_dist_by_type_and_loc_per_item.loc[:,item_dist_by_type_and_loc_per_item.columns!='All'].drop('All').values/100
mult_table_per_item = [[i*j for j in outlet_type_per_item] for i in outlet_loc_per_item]
expected_value_per_item = (outlet_weight_per_item * np.asarray(mult_table_per_item)).sum()
expected_value_per_item


# In[51]:


item_type_list = list(encoded_data['Item_Type'].values)
item_type_list = set(item_type_list)
item_type_list = list(item_type_list)
item_type_list


# In[52]:


#Creating a list of expected frequency list using Outlet_Location_Type and Outlet_Type as indicators
expected_frequency_list = []
for item in item_type_list:
    item_dist_by_type_and_loc_per_item = encoded_data[encoded_data['Item_Type']==item].pivot_table(values='Item_Outlet_Sales', index='Outlet_Location_Type', columns='Outlet_Type', 
                 aggfunc= lambda x: round(sum(x)/sum(encoded_data[encoded_data['Item_Type']==item]['Item_Outlet_Sales']) * 100, 2), 
                 fill_value=0)
    
    outlet_type_per_item = item_dist_by_type_and_loc_per_item.index
    outlet_loc_per_item = item_dist_by_type_and_loc_per_item.columns
    outlet_weight_per_item = item_dist_by_type_and_loc_per_item.values/100
    mult_table_per_item = [[i*j for j in outlet_type_per_item] for i in outlet_loc_per_item]
    expected_value_per_item = (outlet_weight_per_item * np.asarray(mult_table_per_item)).sum()
    expected_frequency_list.append(expected_value_per_item)

expected_frequency_list = {'Item_Type':item_type_list, 'Expected_Frequency':expected_frequency_list}
expected_frequency_list = pd.DataFrame(expected_frequency_list)
expected_frequency_list.sort_values('Expected_Frequency', ascending=False).reset_index().drop('index', axis=1)


# In[53]:


sns.barplot(data=expected_frequency_list, x='Expected_Frequency', y='Item_Type', order=expected_frequency_list.sort_values('Expected_Frequency', ascending=False)['Item_Type'])


# In[54]:


encoded_data.head()


# In[55]:


encoded_data.describe()


# In[56]:


sales_amount_mean


# In[57]:


sales_amount_var


# In[58]:


sales_amount_std


# In[59]:


#Create a standard deviation graph
def normal_pdf(x, mu, var):
    return (1 / np.sqrt(2 * np.pi * var)) *        np.exp(-(1 / (2 * var)) * (x - mu) ** 2)

x_axis_1 = np.arange(0, 350, 0.01)
y_axis_1 = normal_pdf(x_axis_1, sales_amount_mean, sales_amount_var)
plt.plot(x_axis_1, y_axis_1)
#plt.ylim(0,0.6)
plt.xlabel("Sales Amount")
plt.title('PDF of Normal Distribution with mean = '+str(np.round(sales_amount_mean,2))+' and variance = '+str(np.round(sales_amount_var, 2))+'\n standard deviation = ' + str(np.round(sales_amount_std,2)))
plt.show()


# In[60]:


from scipy.stats import norm

#estimating sales > 30.000
estimated_sales = 30000
p_est_sales = 1-norm.cdf(estimated_sales, sales_amount_mean, sales_amount_std)
p_est_sales


# In[61]:


#estimating sales <= 30.000
estimated_sales = 30000
p_est_sales = norm.cdf(estimated_sales, sales_amount_mean, sales_amount_std)
p_est_sales


# In[62]:


#estimating sales > 10.000
estimated_sales = 10000
p_est_sales = norm.cdf(estimated_sales, sales_amount_mean, sales_amount_std)
p_est_sales


# In[63]:


#estimating probability of 0.995
p_est_sales = 0.995
estimated_sales = norm.ppf(p_est_sales, sales_amount_mean, sales_amount_std)
estimated_sales


# In[64]:


#A jointplot of Price and Sales to show their joint density
sns.jointplot(x='Item_MRP', y='Sales_Amount', data=encoded_data, kind='kde')


# In[65]:


#Creating a conditional item to see the difference compared to all items
cond_item = 'Breads'


# In[66]:


cond_item_data = encoded_data[encoded_data['Item_Type'] == cond_item]
cond_item_data.head()


# In[67]:


cond_item_sales_amount_sum = cond_item_data['Sales_Amount'].sum()
cond_item_sales_amount_mean = cond_item_sales_amount_sum/cond_item_data['Item_Outlet_Sales'].sum()
cond_item_sales_amount_var = ((cond_item_data['Sales_Amount'] - cond_item_sales_amount_sum)**2).sum()/(cond_item_data['Item_Outlet_Sales'].sum())
cond_item_sales_amount_std = np.sqrt(cond_item_sales_amount_var)


# In[68]:


cond_item_sales_amount_sum


# In[69]:


cond_item_sales_amount_mean


# In[70]:


cond_item_sales_amount_var


# In[71]:


cond_item_sales_amount_std


# In[72]:


x_axis_2 = np.arange(0, 350, 0.01)
y_axis_2 = normal_pdf(x_axis_2, cond_item_sales_amount_mean, cond_item_sales_amount_var)
plt.plot(x_axis_2, y_axis_2)
#plt.ylim(0,0.6)
plt.xlabel("Sales Amount")
plt.title('PDF of Normal Distribution with mean = '+str(np.round(cond_item_sales_amount_mean,2))+' and variance = '+str(np.round(cond_item_sales_amount_var, 2))+'\n standard deviation = ' + str(np.round(cond_item_sales_amount_std,2)))
plt.show()


# In[73]:


cond_estimated_sales = 3000000
cond_p_est_sales = 1-norm.cdf(cond_estimated_sales, cond_item_sales_amount_mean, cond_item_sales_amount_std)
cond_p_est_sales


# In[74]:


cond_p_est_sales = 0.9
cond_estimated_sales = norm.ppf(cond_p_est_sales, cond_item_sales_amount_mean, cond_item_sales_amount_std)
cond_estimated_sales


# In[75]:


sns.jointplot(x='Item_MRP', y='Sales_Amount', data=cond_item_data, kind='kde')


# In[76]:


#Creating another conditional item to see the difference compared to all items
cond_item_2 = 'Dairy'
cond_item_data_2 = encoded_data[encoded_data['Item_Type'] == cond_item_2]
cond_item_data_2.head()


# In[77]:


cond_item_sales_amount_sum_2 = cond_item_data_2['Sales_Amount'].sum()
cond_item_sales_amount_mean_2 = cond_item_sales_amount_sum_2/cond_item_data_2['Item_Outlet_Sales'].sum()
cond_item_sales_amount_var_2 = ((cond_item_data_2['Sales_Amount'] - cond_item_sales_amount_sum_2)**2).sum()/(cond_item_data_2['Item_Outlet_Sales'].sum())
cond_item_sales_amount_std_2 = np.sqrt(cond_item_sales_amount_var_2)


# In[78]:


x_axis_3 = np.arange(0, 350, 0.01)
y_axis_3 = normal_pdf(x_axis_2, cond_item_sales_amount_mean_2, cond_item_sales_amount_var_2)
plt.plot(x_axis_3, y_axis_3)
#plt.ylim(0,0.6)
plt.xlabel("Sales Amount")
plt.title('PDF of Normal Distribution with mean = '+str(np.round(cond_item_sales_amount_mean,2))+' and variance = '+str(np.round(cond_item_sales_amount_var, 2))+'\n standard deviation = ' + str(np.round(cond_item_sales_amount_std,2)))
plt.show()


# In[79]:


sns.jointplot(x='Item_MRP', y='Sales_Amount', data=cond_item_data_2, kind='kde')


# In[80]:


sns.jointplot(x='Item_MRP', y='Sales_Amount', data=encoded_data, kind='kde')
sns.jointplot(x='Item_MRP', y='Sales_Amount', data=cond_item_data, kind='kde')
sns.jointplot(x='Item_MRP', y='Sales_Amount', data=cond_item_data_2, kind='kde')


# In[81]:


#Scatterplot of Price and Sales for all items data
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,8))

sns.scatterplot(data=encoded_data, x='Item_MRP', y='Sales_Amount', hue='Outlet_Type', ax=ax[0], palette='Set2')
sns.scatterplot(data=encoded_data, x='Item_MRP', y='Sales_Amount', hue='Outlet_Location_Type', ax=ax[1], palette='Set2')

plt.show()


# In[82]:


#Scatterplot of Price and Sales for Breads
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,8))

sns.scatterplot(data=encoded_data, x='Item_MRP', y='Sales_Amount', hue='Outlet_Type', ax=ax[0], palette='Set2')
sns.scatterplot(data=cond_item_data, x='Item_MRP', y='Sales_Amount', hue='Outlet_Location_Type', ax=ax[1], palette='Set2')

plt.show()


# In[83]:


#Calculating correlation of Price and Sales
price_sales_corr = encoded_data[['Item_MRP', 'Sales_Amount']].corr()
price_sales_corr


# In[84]:


sns.heatmap(price_sales_corr)


# In[85]:


#Calculating correlation of Price and item characteristics
item_char_corr = encoded_data[['Item_MRP', 'Item_Fat_Content', 'Item_Visibility', 'Item_Weight']].corr()
item_char_corr


# In[86]:


sns.heatmap(item_char_corr)


# In[87]:


#Devising a distplot to show sales density for all items
sns.distplot(encoded_data['Sales_Amount'], axlabel='All Types Sales', kde=True)


# In[88]:


#Devising a distplot to show sales density for Dairy
conditional_data = encoded_data[encoded_data['Item_Type']=='Dairy']
sns.distplot(conditional_data['Sales_Amount'], axlabel='Dairy Sales', kde=True)


# In[89]:


#Devising a distplot to show sales density for all items vs Dairy
fig = sns.distplot(encoded_data['Sales_Amount'], axlabel='Sales', kde=True)
fig = sns.distplot(conditional_data['Sales_Amount'], axlabel='Sales', kde=True)
plt.show()


# In[90]:


item_type_list


# In[91]:


#Building a function to generate estimated probability with given sales
def given_sales_prob(given_amount, data_to_use): #amount means sales larger than given amount (given=10000, prob of sales > 10000)
    given_sales_prob = []
    for item in item_type_list:
        item_data = data_to_use[data_to_use['Item_Type'] == item]
        item_sales_amount_sum = item_data['Sales_Amount'].sum()
        item_sales_amount_mean = item_sales_amount_sum/item_data['Item_Outlet_Sales'].sum()
        item_sales_amount_var = ((item_data['Sales_Amount'] - item_sales_amount_sum)**2).sum()/(item_data['Item_Outlet_Sales'].sum())
        item_sales_amount_std = np.sqrt(item_sales_amount_var)
        item_p_given_sales = 1-norm.cdf(given_amount, item_sales_amount_mean, item_sales_amount_std)
        given_sales_prob.append(item_p_given_sales)
    
    given_sales_prob = {'Item_Type':item_type_list, 'Estimated_Probability':given_sales_prob}
    given_sales_prob = pd.DataFrame(given_sales_prob)
    given_sales_prob = given_sales_prob.sort_values('Estimated_Probability', ascending=False).reset_index().drop('index', axis=1)
    return given_sales_prob


# In[92]:


#Estimated probability with original data
est_prob_1 = given_sales_prob(5000000, encoded_data)
est_prob_1


# In[93]:


sns.barplot(data=est_prob_1, x='Estimated_Probability', y='Item_Type')


# In[94]:


#Building a function to generate estimated sales with given probability
def given_prob_sales(given_prob, data_to_use): #estimating sales less or equal to with given probability
    given_prob_sales = []
    for item in item_type_list:
        item_data = data_to_use[data_to_use['Item_Type'] == item]
        item_sales_amount_sum = item_data['Sales_Amount'].sum()
        item_sales_amount_mean = item_sales_amount_sum/item_data['Item_Outlet_Sales'].sum()
        item_sales_amount_var = ((item_data['Sales_Amount'] - item_sales_amount_sum)**2).sum()/(item_data['Item_Outlet_Sales'].sum())
        item_sales_amount_std = np.sqrt(item_sales_amount_var)
        sales_given_prob = norm.ppf(given_prob, item_sales_amount_mean, item_sales_amount_std)
        given_prob_sales.append(sales_given_prob)
    
    given_prob_sales = {'Item_Type':item_type_list, 'Estimated_Sales':given_prob_sales}
    given_prob_sales = pd.DataFrame(given_prob_sales)
    given_prob_sales = given_prob_sales.sort_values('Estimated_Sales', ascending=False).reset_index().drop('index', axis=1)
    return given_prob_sales


# In[95]:


#Estimated sales with original data
est_sales_1 = given_prob_sales(0.8, encoded_data)
est_sales_1


# In[96]:


sns.barplot(data=est_sales_1, x='Estimated_Sales', y='Item_Type')


# In[97]:


cond_data_1 = encoded_data[encoded_data['Outlet_Location_Type']==1]


# In[98]:


#Estimated  probability with conditional data
est_prob_2 = given_sales_prob(5000000, cond_data_1)
est_prob_2


# In[99]:


sns.barplot(data=est_prob_2, x='Estimated_Probability', y='Item_Type')


# In[100]:


#Estimated  sales with conditional data
est_sales_2 = given_prob_sales(0.8, cond_data_1)
est_sales_2


# In[101]:


sns.barplot(data=est_sales_2, x='Estimated_Sales', y='Item_Type')


# In[102]:


cond_data_2 = encoded_data[encoded_data['Outlet_Location_Type']==2]


# In[103]:


#Estimated probability with conditional data
est_prob_3 = given_sales_prob(5000000, cond_data_2)
est_prob_3


# In[104]:


sns.barplot(data=est_prob_3, x='Estimated_Probability', y='Item_Type')


# In[105]:


#Estimated sales with conditional data
est_sales_3 = given_prob_sales(0.8, cond_data_2)
est_sales_3


# In[106]:


sns.barplot(data=est_sales_3, x='Estimated_Sales', y='Item_Type')


# In[107]:


cond_data_2 = encoded_data[encoded_data['Outlet_Location_Type']==2]


# In[108]:


#Estimated probability with conditional data
est_prob_2 = given_sales_prob(5000000, cond_data_1)
est_prob_2


# In[109]:


list(set(encoded_data['Outlet_Location_Type'].values))


# In[110]:


encoded_data[encoded_data['Outlet_Location_Type']==1].head()


# In[111]:


#Create a list of probability with given sales of 5.000.000
est_prob_joined = pd.DataFrame({'Item_Type':item_type_list})
for loc_type in list(set(encoded_data['Outlet_Location_Type'].values)):
    est_prob = given_sales_prob(5000000, encoded_data[encoded_data['Outlet_Location_Type']==loc_type])
    est_prob_joined = est_prob_joined.merge(est_prob, on='Item_Type')

est_prob_joined.rename(columns={'Estimated_Probability_x':'Loc_Type__1',
                               'Estimated_Probability_y':'Loc_Type_2',
                               'Estimated_Probability':'Loc_Type_3'}, inplace=True)
est_prob_joined


# In[112]:


est_prob_joined.plot(x='Item_Type', kind='bar', figsize=(10,4), grid=False)
plt.xlabel("Item Type")
plt.ylabel("Probability")
plt.title("Estimated Probability of Sales")
plt.ylim(0,0.2)
plt.legend()
plt.show()


# In[113]:


#Create a list of sales with given probability of 0.8
est_sales_joined = pd.DataFrame({'Item_Type':item_type_list})
for loc_type in list(set(encoded_data['Outlet_Location_Type'].values)):
    est_sales = given_prob_sales(0.8, encoded_data[encoded_data['Outlet_Location_Type']==loc_type])
    est_sales_joined = est_sales_joined.merge(est_sales, on='Item_Type')

est_sales_joined.rename(columns={'Estimated_Sales_x':'Loc_Type__1',
                               'Estimated_Sales_y':'Loc_Type_2',
                               'Estimated_Sales':'Loc_Type_3'}, inplace=True)
est_sales_joined


# In[114]:


est_sales_joined.plot(x='Item_Type', kind='bar', figsize=(10,4), grid=False)
plt.xlabel("Item Type")
plt.ylabel("Sales")
plt.title("Estimated Sales")
plt.ylim(0,5000000)
plt.legend()
plt.show()


# In[115]:


outlet_loc_1_sales = list(encoded_data[encoded_data['Outlet_Location_Type']==1]['Sales_Amount'].values)
outlet_loc_1_sales


# In[116]:


outlet_loc_2_sales = list(encoded_data[encoded_data['Outlet_Location_Type']==2]['Sales_Amount'].values)
outlet_loc_2_sales


# In[117]:


outlet_loc_3_sales = list(encoded_data[encoded_data['Outlet_Location_Type']==3]['Sales_Amount'].values)
outlet_loc_3_sales


# In[118]:


#Hypothesis Testing with H0: Loc 3 Sales >= Loc 1 Sales, H1: Loc 3 Sales < Loc 1 Sales
stat, p = ttest_ind(outlet_loc_3_sales, outlet_loc_1_sales, equal_var=False, alternative='less') # eaual_var= False due to different population

# Interpretation
print('Statistics = %.4f, p = %.4f' % (stat, p)) 


# In[119]:


#Hypothesis Testing with H0: Loc 3 Sales >= Loc 2 Sales, H1: Loc 3 Sales < Loc 2 Sales
stat, p = ttest_ind(outlet_loc_3_sales, outlet_loc_2_sales, equal_var=False, alternative='less') # eaual_var= False due to different population

# Interpretation
print('Statistics = %.4f, p = %.4f' % (stat, p))


# In[120]:


#Hypothesis Testing with H0: Loc 2 Sales >= Loc 1 Sales, H1: Loc 2 Sales < Loc 1 Sales
stat, p = ttest_ind(outlet_loc_2_sales, outlet_loc_1_sales, equal_var=False, alternative='less') # eaual_var= False due to different population

# Interpretation
print('Statistics = %.4f, p = %.4f' % (stat, p))


# In[121]:


outlet_type_1_sales = list(encoded_data[encoded_data['Outlet_Type']==1]['Sales_Amount'].values)
outlet_type_1_sales


# In[122]:


outlet_type_2_sales = list(encoded_data[encoded_data['Outlet_Type']==2]['Sales_Amount'].values)
outlet_type_2_sales


# In[123]:


outlet_type_3_sales = list(encoded_data[encoded_data['Outlet_Type']==3]['Sales_Amount'].values)
outlet_type_3_sales


# In[124]:


outlet_type_4_sales = list(encoded_data[encoded_data['Outlet_Type']==4]['Sales_Amount'].values)
outlet_type_4_sales


# In[125]:


#Hypothesis Testing with H0: Type 3 Sales >= Type 1 Sales, H1: Type 3 Sales < Type 1 Sales
stat, p = ttest_ind(outlet_type_3_sales, outlet_type_1_sales, equal_var=False, alternative='less') # eaual_var= False due to different population

# Interpretation
print('Statistics = %.4f, p = %.4f' % (stat, p)) 


# In[126]:


#Hypothesis Testing with H0: Type 3 Sales >= Type 2 Sales, H1: Type 3 Sales < Type 2 Sales
stat, p = ttest_ind(outlet_type_3_sales, outlet_type_2_sales, equal_var=False, alternative='less') # eaual_var= False due to different population

# Interpretation
print('Statistics = %.4f, p = %.4f' % (stat, p)) 


# In[127]:


#Hypothesis Testing with H0: Type 2 Sales >= Type 1 Sales, H1: Type 2 Sales < Type 1 Sales
stat, p = ttest_ind(outlet_type_2_sales, outlet_type_1_sales, equal_var=False, alternative='less') # eaual_var= False due to different population

# Interpretation
print('Statistics = %.4f, p = %.4f' % (stat, p)) 


# In[ ]:





# In[ ]:




