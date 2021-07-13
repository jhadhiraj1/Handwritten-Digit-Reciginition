#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
from sklearn.datasets import load_digits
from sklearn import svm
import matplotlib.pyplot as plt
digits=load_digits()

plt.gray()
plt.matshow(digits.images[20])
plt.show()


# In[12]:


print(digits.images[20])


# In[13]:


model=svm.SVC()
model.fit(digits.data[:-1],digits.target[:-1])
prediction=model.predict(digits.data[20:21])
print(prediction)

