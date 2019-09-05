
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search


# In[13]:


import pandas as pd
df=pd.read_csv('MODEL_DATA_SCALA_TRAIN.csv')
val=pd.read_csv('MODEL_DATA_SCALA_VALID_SHARED.csv')


# In[3]:



val['Target']=0
df=pd.concat([df_train,val],axis=0)


# In[4]:


df['Order_date']=pd.to_datetime(df['Order_date'])
dates=df['Order_date'] 


# In[8]:


del df['Courier_Name']

del df['Weekend_OFD']
del df['Weekday_OFD']
del df['Days_bw_Order_OFD']


# In[43]:


ones=df_train[df_train['Target']==1]


# In[42]:


df_train['Target'].index=range(len(df_train))


# In[44]:


ones


# In[218]:


df=df.sample(25000)


# In[14]:


#preprocessing-Encoding & Dummification

numeric=df.dtypes[(df.dtypes=='int64')  | (df.dtypes=='float64')]

non_numeric=df.dtypes[df.dtypes=='object']


num_df=df[numeric.index]

non_num_df=df[non_numeric.index]

num_df=num_df.fillna(num_df.mean())

non_num_fil=[]
non_num_lar=[]
for i in non_num_df.columns:
    if len(non_num_df[i].unique()) <30:
        non_num_fil= non_num_fil+[i]
    else:
        non_num_lar=non_num_lar+[i]


lar=df[non_num_lar]

from sklearn.preprocessing import LabelEncoder

lar_enc=lar.apply(LabelEncoder().fit_transform)


non_num_df=df[non_num_fil]

non_num_ddf=pd.get_dummies(non_num_df)

trainxy=pd.concat([non_num_ddf,num_df,lar_enc],axis=1)


# In[15]:


#feature Engineering
df['size_1']=0
df.loc[df['Cart_Size']==1,'size_1']=1

df['Cart_Value']=df['Cart_Value'].astype('str')
df['Order_date']=df['Order_date'].astype('str')
df['fullvisitorid']=df['fullvisitorid'].astype('str')
df['Target']=df['Target'].astype('str')
df['txn_hour']=df['txn_hour'].astype('str')
df['Item_Discount']=df['Item_Discount'].astype('str')
df['Order_Month']=df['Order_Month'].astype('str')
df['Men_Footwear']=df['Men_Footwear'].astype('str')
df['Women_Footwear']=df['Women_Footwear'].astype('str')
df['Men_Apparel']=df['Men_Apparel'].astype('str')
df['Women_Apparel']=df['Women_Apparel'].astype('str')
df['Cart_Size']=df['Cart_Size'].astype('str')
df['cust_address']=df['cust_address'].astype('str')
df['campaign']=df['campaign'].astype('str')
df['size_1']=df['size_1'].astype('str')
df['Zip_Code']=df['Zip_Code'].astype('str')
df['Men_Acc']=df['Men_Acc'].astype('str')
df['Women_Acc']=df['Women_Acc'].astype('str')

df['identity_id']=df['fullvisitorid']+df['Cart_Value']+df['Item_Discount']+df['Order_date']+df['txn_hour']+df['Men_Footwear']+df['Men_Apparel']+df['Women_Apparel']+df['Women_Footwear']+df['Men_Acc']+ df['Women_Acc']
df['identity_id2']=df['cust_address']+df['campaign']
df['identity_id3']=df['cust_address']+df['Cart_Value']+df['Item_Discount']+df['Order_date']+df['txn_hour']+df['Men_Footwear']+df['Men_Apparel']+df['Women_Apparel']+df['Women_Footwear']+df['Men_Acc']+ df['Women_Acc']    
df['identity_id4']=df['cust_address']+df['Cart_Value']+df['Item_Discount']
df['identity_id5']=df['cust_address']+df['Cart_Value']+df['Item_Discount']+df['Order_date']+df['txn_hour']+df['Men_Footwear']+df['Men_Apparel']+df['Women_Apparel']+df['Women_Footwear']+df['Men_Acc']+ df['Women_Acc']+df['size_1']



df['identity_id6']=df['cust_address']+df['Cart_Value']+df['Item_Discount']

dups=df.duplicated(['identity_id'],keep=False)
dups2=df.duplicated(['identity_id2'],keep=False)
dups3=df.duplicated(['identity_id3'],keep=False)
dups4=df.duplicated(['identity_id4'],keep=False)
dups5=df.duplicated(['identity_id5'],keep=False)
dups6=df.duplicated(['identity_id6'],keep=False)

trainxy['id1']=dups
trainxy['id2']=dups2
trainxy['id3']=dups3
trainxy['id4']=dups4
trainxy['id5']=dups5
trainxy['id6']=dups6



# # Event Rate for flags:-

# In[22]:


id1=df[dups]['Target']

id1.value_counts()/len(id1)


# In[23]:


id1=df[dups2]['Target']

id1.value_counts()/len(id1)


# In[24]:


id1=df[dups3]['Target']

id1.value_counts()/len(id1)


# In[25]:


id1=df[dups4]['Target']

id1.value_counts()/len(id1)


# In[26]:


id1=df[dups5]['Target']

id1.value_counts()/len(id1)


# In[8]:


df_cols=pd.read_csv('fil_cols_360.csv',header=None)
fil_cols=list(df_cols[1])


# In[ ]:





# In[172]:


trainxy_copy=trainxy.copy()


# In[12]:


#training
trainxy['dt']=dates
trainxy=trainxy[fil_cols]
train_xy=trainxy[trainxy['dt']<='2017-04-30 00:00:00']
validation=trainxy[(trainxy['dt']>='2017-04-01 00:00:00') & (trainxy['dt']<='2017-04-30 00:00:00')]

del train_xy['dt']
del validation['dt']


# In[13]:


#validation
trainxy['dt']=dates
trainxy=trainxy[fil_cols]
train_xy=trainxy[trainxy['dt']<='2017-04-30 00:00:00']
validation=trainxy[(trainxy['dt']>='2017-05-01 00:00:00') & (trainxy['dt']<='2017-05-31 00:00:00')]

del train_xy['dt']
del validation['dt']


# In[23]:


#train_y=train_xy['Target']
val_y=validation['Target']

#del train_xy['Target']
del validation['Target']


# # Algorithm Hyperparameters

# In[16]:


alg = XGBClassifier( scale_pos_weight=2.5,subsample=0.9,n_estimators=500)
xgb_param = alg.get_xgb_params()

#val_y=val_y.astype(int)
#train_y=train_y.astype(int)
mods=alg.fit(train_xy,train_y) 

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score




# In[20]:


validation=trainxy[(trainxy['dt']>='2017-04-01 00:00:00') & (trainxy['dt']<='2017-04-30 00:00:00')]


# In[25]:


del validation['dt']


# In[26]:


preds=mods.predict(validation)

print 'recall',recall_score(val_y,preds)

print 'precision',precision_score(val_y,preds)


pred_proba=mods.predict_proba(validation)

prob_df=pd.DataFrame(pred_proba)

val_y.index=range(len(val_y))

prob_df['actual']=val_y
prob_df['pred']=preds

prob_dfs=prob_df.sort_values([1],ascending=False)

prob_dfs.index=range(len(prob_dfs))

top_20=prob_dfs[0:int(len(prob_dfs)*.20)]

print 'top 20 recall',sum(top_20['actual'])/float(sum(prob_dfs['actual']))


# In[150]:


validation['prob']=proba[1]

validation['Order_No']=val['Order_No']

sorteds=validation.sort('prob',ascending=False)

sub1=sorteds[['Order_No','pred','prob']]


# In[70]:


#all april
recall 0.596284730563
precision 0.46529209622
top 20 recall 0.446552966611

#id6
recall 0.569381055329
precision 0.485923192565
top 20 recall 0.448875010009

#REMOVDED ID5,ID6
recall 0.566418448234
precision 0.485618178074
top 20 recall 0.447754023541

recall 0.563776122988
precision 0.486324077911
top 20 recall 0.448474657699

recall 0.56265513652
precision 0.487207931776
top 20 recall 0.448554728161

#without order_date diif
recall 0.55865161342
precision 0.487970345503
top 20 recall 0.448474657699

#all
recall 0.614861077748
precision 0.458749029213
top 20 recall 0.445912402915

#non-zero #365 var
recall 0.559292177116
precision 0.485777870506
top 20 recall 0.448154375851

#340 vars
recall 0.557530626952
precision 0.486786912752
top 20 recall 0.446953318921

#all
recall 0.614861077748
precision 0.458749029213
top 20 recall 0.445912402915

#non-zero 365
recall 0.559292177116
precision 0.485777870506
top 20 recall 0.448154375851

#with id6-2 vars
#non-zero 365
recall 0.559292177116
precision 0.485777870506
top 20 recall 0.4477

recall 0.492163009404
precision 0.524498886414
top 20 recall 0.435736677116

#10k
recall 0.491859468723
precision 0.480737018425
top 20 recall 0.420736932305

recall 0.491859468723
precision 0.480737018425
top 20 recall 0.420736932305

top 20 recall 0.448154375851


recall 0.510778080533
precision 0.4968049707


# In[27]:





# In[134]:


#feature selection
ids=['fullvisitorid', 'visitid', 'visitorid','txnid','Order_No']
cols=[]
for i in zipped:
    if i[1]!=0:
        cols=cols+[i[0]]

fil_cols=set(cols).difference(set(ids))
fil_cols=list(fil_cols)

fil_cols.append('dt')
fil_cols.append('Target')

#train_xy[fil_cols].to_csv('train_xy_fil.csv')


# In[5]:


from sklearn.externals import joblib
clf=joblib.load( 'model_full.pkl')


# In[9]:


zipped=zip(fil_cols,clf.feature_importances_)
zipped.sort(key = lambda t: t[1],reverse=True)


# # Key Features

# In[10]:


zipped


# In[ ]:




