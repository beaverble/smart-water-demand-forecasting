#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Dropout
import os
import random
from tensorflow.python.keras.models import load_model
import sys


# In[2]:


seed = 2500
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)


# # 일반용

# In[3]:


realtime_data = pd.read_csv("api_final_normal_2022(long).csv")
realtime_data.drop(['Unnamed: 0'],axis=1, inplace=True)
realtime_data.drop(['WSPDBTCD'],axis=1, inplace=True)
realtime_data.drop(['DAY'],axis=1, inplace=True)


# In[4]:




# In[5]:


realtime_data.set_index(['CUSNUM',"DATE"],drop=False, inplace=True)
realtime_data.drop(['DATE'],axis=1, inplace=True)
realtime_data = realtime_data.sort_index(ascending=True)


# In[6]:


#dummy encoding
realtime_data = pd.get_dummies(data=realtime_data, columns=['DAY_OF_WEEK'])
realtime_data = pd.get_dummies(data=realtime_data, columns=['MONTH'])
realtime_data = pd.get_dummies(data=realtime_data, columns=['HOUR'])
realtime_data = pd.get_dummies(data=realtime_data, columns=['CUSNUM'])
realtime_data = pd.get_dummies(data=realtime_data, columns=['7D_AM_LONG_WEATHER'])
realtime_data = pd.get_dummies(data=realtime_data, columns=['7D_PM_LONG_WEATHER'])


# In[7]:


count=0
k = 0
scaled_usage_array = np.empty((0,1), dtype=float)
scaler_usage_list=[]

for i in range(len(realtime_data.index)):
    if i < (len(realtime_data.index) - 1):
        if not realtime_data.index[i][0] == realtime_data.index[i+1][0]:
            count = count + 1
            scaled_usage = realtime_data["USAGE"][k:i]            
            globals()['scaler_usage_{}'.format(count)] = MinMaxScaler()
            scaled_usage_array = np.append(scaled_usage_array, 
                                         globals()['scaler_usage_{}'.format(count)].fit_transform((scaled_usage.values).reshape(-1,1)),
                                        axis=0)           
            k = i
            scaler_usage_list.append(globals()['scaler_usage_{}'.format(count)])
    else:
        count = count + 1
        scaled_usage = realtime_data["USAGE"][k:i+1]        
        globals()['scaler_usage_{}'.format(count)] = MinMaxScaler()
        scaled_usage_array = np.append(scaled_usage_array,
                                      globals()['scaler_usage_{}'.format(count)].fit_transform((scaled_usage.values).reshape(-1,1)),
                                     axis=0)        
        k = i
        scaler_usage_list.append(globals()['scaler_usage_{}'.format(count)])

scaled_usage_array = scaled_usage_array.reshape(-1,)


# In[8]:


len(scaler_usage_list)


# In[9]:


realtime_data["USAGE"] = scaled_usage_array


# #### 스케일

# In[10]:


scaler_feature = MinMaxScaler()


# In[11]:


realtime_data.loc[:,"TEMPERATURE":"7D_AM_LONG_TEMP_HIGH.2"] = scaler_feature.fit_transform(realtime_data.loc[:,"TEMPERATURE":"7D_AM_LONG_TEMP_HIGH.2"])


# In[12]:


idx = pd.IndexSlice
train_data = realtime_data.loc[idx[:,'2021-12-26 00:00:00':'2022-01-21 23:00:00'],:]
test_data = realtime_data.loc[idx[:,'2022-01-22 00:00:00':'2022-01-30 23:00:00'],:]


# In[13]:


realtime_train_dataset = train_data.values
realtime_test_dataset = test_data.values


# ### Window 생성 

# In[14]:


def day_cal(dataset):
    day_list = []

    for i in range(len(dataset)-1):
        if dataset.index[i][0] != dataset.index[i+1][0]:
            day_list.append(i)

    day_list.append(len(dataset))
    return day_list


# In[15]:


train_day = day_cal(train_data)
test_day = day_cal(test_data)


# In[16]:


len(test_day)


# In[17]:


def make_dataset(dataset, target, history_size, 
                      cusnum_len, day, target_size):
    
    data = []
    labels = [] 
    check_point = 0
    check_list = []
    k = 0

    for i in range(cusnum_len):        
        start_index = check_point + history_size 
        end_index = day[i]

        for j in range(start_index, end_index, 24):
            if j+target_size <= end_index:
                indices = range(j-history_size, j)         
                data.append(dataset[indices])
                labels.append(target[j:j+target_size])
                check_point = end_index
                k = k + 1
            else :
                break
            
            print("last x_data : ",indices)
            print('last y_data_start : ', j)
            print('last y_data_end : ', j+target_size)
        
        check_list.append(k)

    return np.array(data), np.array(labels),check_list


# In[18]:


#윈도우 설정
shift_days = 14
shift_steps = shift_days * 1
future_target = 7


# # Train, Test 데이터셋 생성

# In[19]:


len(realtime_train_dataset)


# In[20]:


final_x_train, final_y_train,length = make_dataset(realtime_train_dataset, realtime_train_dataset[:,0], shift_steps, 
                                     len(scaler_usage_list),train_day, future_target)


# In[21]:


len(realtime_test_dataset)


# In[22]:


final_x_test, final_y_test,length = make_dataset(realtime_test_dataset, realtime_test_dataset[:,0], shift_steps,
                                         len(scaler_usage_list), test_day, future_target)


# In[23]:


final_y_test.shape


# In[24]:


final_x_train.shape[-2:]


# # 모델 생성

# In[25]:


adam = keras.optimizers.Adam()


# In[26]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=64,return_sequences=True, input_shape=final_x_train.shape[-2:]))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units=64))
model.add(tf.keras.layers.Dropout(0.2))


# In[27]:


model.add(tf.keras.layers.Dense(7, activation='sigmoid'))
model.compile(loss="mse", optimizer= adam)


# In[28]:


model.summary()


# In[29]:


early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

path_checkpoint = '2022_long_normal.h5'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,monitor='val_loss',
                                      verbose=1,save_weights_only=False,save_best_only=True)


# In[30]:


model.fit(final_x_train, final_y_train, validation_data=(final_x_test,final_y_test), batch_size=64,
          epochs=1000, verbose=2, callbacks=[early_stopping, callback_checkpoint])


# In[31]:


y_true = final_y_test
y_pred = model.predict(final_x_test)


# In[32]:


"""j = 0
for i in range(len(length)):
    y_true[j:length[i]] = scaler_usage_list[i].inverse_transform(y_true[j:length[i]])
    y_pred[j:length[i]] = scaler_usage_list[i].inverse_transform(y_pred[j:length[i]])
    j = length[i]"""


# In[33]:


wrong_num = []
for i in range(len(y_true)):
    test_num = np.where(y_true[i]==0)
    num = len(test_num[0])
    
    if num >= 3:
        wrong_num.append(i)

print("잘못된 레이블 : ", len(wrong_num))


# In[34]:


final_y_true = np.delete(y_true,wrong_num,axis=0)
final_y_pred = np.delete(y_pred,wrong_num,axis=0)


# In[35]:


y_true_sum = final_y_true.sum(axis=1)
y_pred_sum = final_y_pred.sum(axis=1)


# # MAPE

# In[36]:


def MAPE(y_true,y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[37]:


MAPE(y_true_sum, y_pred_sum)


# In[ ]:




