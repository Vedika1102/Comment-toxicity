#!/usr/bin/env python
# coding: utf-8

# # Install dependencies 

# In[2]:


get_ipython().system('pip install tensorflow tensorflow-gpu pandas matplotlib sklearn')


# In[3]:


import os
import pandas as pd
import numpy as np
import tensorflow as tf


# In[4]:


df = pd.read_csv('train.csv')


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.iloc[0]['comment_text']


# # Preprocessing 

# In[8]:


from tensorflow.keras.layers import TextVectorization


# In[9]:


#sperating comments and labels
X = df['comment_text']
y = df[df.columns[2:]].values


# In[10]:


MAX_FEATURES = 200000 #number of words in the vocab


# In[11]:


vectorizer = TextVectorization(max_tokens = MAX_FEATURES,
                              output_sequence_length = 1800,
                              output_mode = 'int') #taking words and converting them to interger values, meaning each word will be given a specific int value


# In[12]:


vectorizer.adapt(X.values) # teaching vectorizr out vocabulary


# In[13]:


vectorized_text = vectorizer(X.values)


# In[14]:


vectorized_text # a numpy array containing integer value for each of our tokens in comment_text


# In[15]:


#tensorflow data pipeline
# steps 
#MCSHBAP = map, shuffle, batch, prefetch from_tensor_slices, list_file
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text,y)) #passing input features and target labels
dataset = dataset.cache()
dataset = dataset.shuffle(1600000) #shuffles the data
dataset = dataset.batch(16) #makes a batch of 16
dataset = dataset.prefetch(8) #helps prevent bottleneck


# In[16]:


#batch = comments + labels
batch_X,batch_y = dataset.as_numpy_iterator().next()


# In[17]:


len(dataset) #this len is the number of batches


# In[18]:


train = dataset.take(int(len(dataset)*.7)) #70% of data as training dataset
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2)) #validation, fist skip 70% data and take 20%
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))


# In[19]:


train_generator = train.as_numpy_iterator()


# In[20]:


train_generator.next()


# In[21]:


#what exactly is happening
#Our deep learning model it is running a particular batch, which goes through forward pass, backward pass, updation of weights 
#and then from the command of .next() it grabs next batch and same process applies


# # Creating sequential model

# In[22]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding

#bidirectional layer : modifier. going to pass features or the values from lstm output across the board as we're passing through our sequences
# dropout is a method of regularization 
# dense layer is a fully connected layer
#embedding layer: works as a personality test for each word in the comments


# In[23]:


model = Sequential()
#1st layer - embedding layer
model.add(Embedding(MAX_FEATURES+1, 32)) # MAX_FEATURES+1 embedding layer and 32 values long 

#Bidirectional LSTM layer
model.add(Bidirectional(LSTM(32, activation = 'tanh')))

#Feature extractor fully connected layers
model.add(Dense(128, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
#final layer
model.add(Dense(6, activation = 'sigmoid'))



# In[24]:


#compiling our model
model.compile(loss= 'BinaryCrossentropy', optimizer = 'Adam')


# In[25]:


model.summary()


# In[26]:


#training our model
history = model.fit(train,epochs=1,validation_data=val)


# In[27]:


history.history


# In[28]:


from matplotlib import pyplot as plt


# In[29]:


plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()


# # Make predictions

# In[40]:


input_text = vectorizer('You freaking suck! I am going to kill you.')


# In[41]:


input_text


# In[44]:


df.columns[2:]


# In[50]:


res = model.predict(np.expand_dims(input_text,0))


# In[51]:


(res > 0.5).astype(int)


# In[52]:


batch_X, batch_y = test.as_numpy_iterator().next()


# In[54]:


batch_y


# In[55]:


(model.predict(batch_X) > 0.5).astype(int)


# In[56]:


res.shape


# # Evaluation

# In[57]:


from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy


# In[58]:


pre = Precision()
re = Recall()
acc = CategoricalAccuracy()


# In[59]:


for batch in test.as_numpy_iterator(): 
    # Unpack the batch 
    X_true, y_true = batch
    # Make a prediction 
    yhat = model.predict(X_true)
    
    # Flatten the predictions
    y_true = y_true.flatten()
    yhat = yhat.flatten()
    
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)


# In[60]:


print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')


# # Test and Gradio

# In[61]:


get_ipython().system('pip install gradio jinja2')


# In[62]:


import tensorflow as tf
import gradio as gr


# In[63]:


model.save('toxicity.h5')


# In[64]:


model = tf.keras.models.load_model('toxicity.h5')


# In[65]:


input_str = vectorizer('hey i freaken hate you!')


# In[66]:


res = model.predict(np.expand_dims(input_str,0))


# In[67]:


res


# In[68]:


def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    
    return text


# In[69]:


interface = gr.Interface(fn=score_comment, 
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),
                        outputs='text')


# In[70]:


interface.launch(share=True)


# In[ ]:




