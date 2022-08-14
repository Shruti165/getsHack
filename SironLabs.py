#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#.pyplot as plt
import pip
#pip.main(['install','seaborn'])
import seaborn as sns

import warnings


# In[33]:


supplier_data = pd.read_csv("supplier1.csv")


# In[34]:


supplier_data.isnull().sum()
supplier_data.isnull().values.any()


# In[35]:


supplier_data


# In[36]:


supplier_data.head()
supplier_data.dtypes
supplier_data.shape
supplier_data.describe()


# In[37]:


supplier_data['Good_Supplier'] = np.where(supplier_data['Rating']>80, 'Yes', 'No')


# In[38]:


supplier_data


# In[39]:


supplier_data_num = supplier_data[['Average Delivery Time',
             'Number of Escalations',
             'Year',
             'Resources']].copy()

plt.figure(figsize= (10,10), dpi=100)

# Show the number of employees that left and stayed by age
import matplotlib.pyplot as plt
fig_dims = (12, 4)
fig, ax = plt.subplots(figsize=fig_dims)
# ax = axis
sns.countplot(x='Rating', hue='Good_Supplier', data=supplier_data, palette="colorblind", ax=ax,
              edgecolor=sns.color_palette("dark", n_colors=1));


# In[40]:


supplier_data['Good_Supplier'].value_counts()
attrition_count = pd.DataFrame(supplier_data['Good_Supplier'].value_counts())
attrition_count
plt.pie(attrition_count['Good_Supplier'] , labels = ['No' , 'Yes'] , explode = (0.2,0))


# In[41]:


sns.countplot(supplier_data['Good_Supplier'])


# In[42]:


sns.heatmap(supplier_data_num.corr())


# In[43]:


# Visualize the correlation
plt.figure(figsize=(14, 14))  # 14in by 14in
sns.heatmap(supplier_data.corr(), annot=True, fmt='.0%')


# In[44]:


supplier_data


# In[45]:


#To remove the strongly correlated variables

supplier_data_uc = supplier_data_num[['Average Delivery Time',
             'Number of Escalations',
             'Year',
             'Resources'
             ]].copy()


supplier_data_uc.head()


# In[46]:


#Copy categorical data
supplier_data_cat = supplier_data[['Good_Supplier','Function', 'Country','Region',
                       'Service']].copy()
supplier_data_cat.head()
Num_val = {'Yes':1, 'No':0}
supplier_data_cat['Good_Supplier'] = supplier_data_cat["Good_Supplier"].apply(lambda x: Num_val[x])
supplier_data_cat.head()
supplier_data_cat = pd.get_dummies(supplier_data_cat)
supplier_data_cat.head()
supplier_data_final = pd.concat([supplier_data_num, supplier_data_cat], axis=1)
supplier_data_final.head()


# In[47]:


from sklearn.cross_validation import train_test_split
target = supplier_data_final['Good_Supplier']
features = supplier_data_final.drop('Good_Supplier', axis = 1)
# Split the dataset into 75% Training set and 25% Testing set
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=10)


# In[26]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)
from sklearn.metrics import accuracy_score
test_pred = model.predict(X_test)
print (accuracy_score(y_test, test_pred))


# In[27]:


# Return the feature importances (the higher, the more important the feature).
feat_importances = pd.Series(model.feature_importances_, index=features.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances


# In[28]:


feat_importances.plot(kind='barh')
warnings.filterwarnings("ignore")


# In[ ]:




