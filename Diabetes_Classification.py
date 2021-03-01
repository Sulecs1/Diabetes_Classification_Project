########################################################
#          Diabetes Classification Project             #
########################################################
#<<<Şule AKÇAY>>>
#Pregnancies: Hamilelik sayısı
#Glucose: Oral glikoz tolerans testintinde 2 saatlik plazma glikoz
#konsantrasyonu
#BloodPressure: Kan basıncı (Küçük tansiyon) (mm Hg)
#SkinT hickness: Cilt kalınlığı
#Insulin: 2 saatlik serum insülini (mu U/ ml)
#BMIBody: Vücut kitle indeksi (Weight in kg/ (height in m)^2)
#DiabetesPedigreeFunc tion: Aile geçmişine göre diyabet olasılığını
#puanlayan bir fonksyion.
#Age: Yaş (yıl)
#Outcome: Hastalığa sahip (1) ya da değil (0)
#########################################################

#Gerekli Olan Kütüphaneler eklendi
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz, export_text
from sklearn.preprocessing import MinMaxScaler
import missingno as msno
import pickle
import pydotplus
from skompiler import skompile
import joblib
import warnings
from sklearn.metrics import *
from sklearn.model_selection import *


#Eklentiler eklendi
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)
warnings.filterwarnings('ignore')

