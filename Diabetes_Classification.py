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

import pickle
from helpers.data_prep import *
from helpers.eda import *
from helpers.helpers import *

#Eklentiler eklendi
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)
warnings.filterwarnings('ignore')

#Veri seti için fonskiyon oluşturuldu
def load():
    data = pd.read_csv(r"C:\Users\Suleakcay\PycharmProjects\pythonProject6\data\diabetes.csv")
    return data

df = load()

#Aykırı değer varsa görebilmek için
msno.bar(df)
plt.show()
#veri seti gözlemler hakkında inceleme yapıldı
grab_col_names(df)

def data_understand(df):
    print("DF SHAPE:", df.shape)
    print("------------------------------------------------------------------------")
    print("OUTCOME 1 DF RATIO:", len(df[df["Outcome"] == 1]) / len(df))
    print("OUTCOME 0 DF RATIO:", len(df[df["Outcome"] == 0]) / len(df))
    print("------------------------------------TYPES------------------------------------")
    print(df.dtypes)
    print("------------------------------------HEAD------------------------------------")
    print(df.head())
    print("-------------------------------------TAİL-----------------------------------")
    print(df.tail())
    print("------------------------------------DESCRİBE------------------------------------")
    print(df.describe().T)
    print("-------------------------------------QUANTİLE-----------------------------------")
    print(df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("-----------------------------------CORR-------------------------------------")
    # Isı haritasında, daha parlak renkler daha fazla korelasyonu gösterir.
    # Tablodan ve ısı haritasından da görebileceğimiz gibi, glikoz seviyeleri, yaş, vücut kitle indeksi ve gebelik sayısı, sonuç değişkeni ile önemli bir korelasyona sahiptir. Ayrıca yaş ve gebelikler veya insülin ve cilt kalınlığı gibi özellik çiftleri arasındaki korelasyona dikkat ediniz.

    corr = df.corr()
    print(corr)
    print("--------------------------------------HEATMAP----------------------------------")
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns)
    plt.show()
    print("------------------------------------------------------------------------")
    print(sns.pairplot(df));
    plt.show()
    print("------------------------------------------------------------------------")
    df.hist(bins=20, color="#1c0f45", edgecolor='orange', figsize=(15, 15));
    plt.show()
    print("------------------------------------------------------------------------")

data_understand(df)

#Veri setindeki eksik değerleri sorgulamak için
def df_questioning_null(df):
    """
    Veri seti içerisindeki eksik değere sahip değişken bilgisini sorgular. Bununla birlikte eksik gözlem olması durumunda sadece eksik gözleme sahip sütun isimlerini getirir.
    """
    print(f"Veri kümesinde hiç boş değer var mı?: {df.isnull().values.any()}")
    if df.isnull().values.any():
        null_values = df.isnull().sum()
        print(f"Hangi sütunlarda eksik değerler var?:\n{null_values[null_values > 0]}")

df_questioning_null(df)





