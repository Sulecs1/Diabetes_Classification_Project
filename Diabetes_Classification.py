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
df.head()

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

#Kategorik ve numerik değişkenleri sorgulamak için fonksiyon oluşturuldu
def cat_num_col(df):
    cat_cols = col_names = [col for col in df.columns if df[col].dtypes != "O"]
    if len(cat_cols) == 0:
        print("Kategorik değişken bulunmamaktadır!")
    else:
        print(f'Kategorik değişkenler : {len(cat_cols)}')
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    if len(num_cols) == 0:
        print("Numerik değişken bulunmamaktadır")
    else:
        print(f'Numerik değişkenler : {len(num_cols)} ')

cat_num_col(df)

def missing_values_table(df):
    na_variable = [col for col in df.columns if df[col].isnull().sum() > 0]

    n_miss = df[na_variable].isnull().sum().sort_values(ascending=False)

    ratio = (df[na_variable].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    if len(missing_df) > 0:
        print("\nEksik değerleri olan {} sütun var\n".format(len(missing_df)))
    else:
        print("Eksik değerler yoktur!")

missing_values_table(df)

def col_nan_assigment(df):
    for col in df.columns:
        for row in range(len(df)):
            if col != "Outcome":
                if df.loc[row, col] == 0:
                    if df.loc[row, "Outcome"] == 1:
                        df.loc[row, col] = df.loc[df["Outcome"] == 1, col].median()
                    else:
                        df.loc[row, col] = df.loc[df["Outcome"] == 0, col].median()

col_nan_assigment(df)
df.head()

#numerik ve kategorik değişken isimleri
def num_and_cat_name(df):
    cat_cols = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes != "O"]
    num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in ["Outcome"]]
    return cat_cols, num_cols

num_and_cat_name(df)

list_num = []
for col in df.columns:
    if df[col].dtypes != 'O' and col not in ["Outcome"]:
        list_num.append(col)


#Aykırı değerler boxplot grafiği  gözlendi
def plot_outliers(df):
    for col in df.columns:
       if col in list_num:
                sns.boxplot(x=df[col])
                plt.title("BoxPlot Grafik Gösterimi")
                plt.show()
plot_outliers(df)


#numerik değişkenlere göre target analizi yaptım
num_cols = [col for col in df.columns if df[col].nunique() > 10
            and df[col].dtypes != 'O'
            and col not in ["Outcome"]]

def target_summary_with_num(dataframe, target, numerical_col):#yukarıdaki işlemin genelleştirlmiş hali

     print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

#sayısal değişkenleri birbirleri ile karşılaştırma işlemi yapıldı
def check_plot(dataframe):
    for colx in df.columns:
        for coly in list_num:
            if colx != coly:
              sns.lmplot(x=colx, y=coly, data=dataframe)
              plt.show()

check_plot(df)







