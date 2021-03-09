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
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor
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
    df.hist(bins=20, color="#1c0f45", edgecolor='orange', figsize=(15, 15));
    plt.show()
    print("------------------------------------------------------------------------")

data_understand(df)

#Veri setindeki eksik değerleri sorgulamak için
def df_questioning_null(df):

    print(f"Veri kümesinde hiç boş değer var mı?: {df.isnull().values.any()}")
    if df.isnull().values.any():
        null_values = df.isnull().sum()
        print(f"Hangi sütunlarda eksik değerler var?:\n{null_values[null_values > 0]}")

df_questioning_null(df)
df.shape


def col_nan_assigment(df):
    #Nehir Günde Daşçı
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

#sayısal değişkenleri birbirleri ile karşılaştırma işlemi grafik incelenerek  yapıldı
#def check_plot(dataframe):
#    for colx in df.columns:
#        for coly in list_num:
#            if colx != coly:
#             sns.lmplot(x=colx, y=coly, data=dataframe)
#             plt.show()

#check_plot(df)

#eşik değerini bulmak için kırılım grafiği incelendi
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()

esik_deger = np.sort(df_scores)[5] #eşik değerimiz
df[df_scores < esik_deger] #eşik değerine göre seçim aykırıları seçtik yaptık
df[df_scores < esik_deger].shape #sayısı
df.describe().T
#indeksleri tutma amacımız indekse göre kolay işlem yapmak
df[df_scores < esik_deger].index
#hepsini silmek istersek
df.drop(axis=0, labels=df[df_scores < esik_deger].index)
df = df.drop(axis=0, labels=df[df_scores < esik_deger].index)
df.head()

#kişinin akrabalarının diabet olma olasılığını 0-1 arasına çektik
transformer = MinMaxScaler()
df["DiabetesPedigreeFunction"] = transformer.fit_transform(df[["DiabetesPedigreeFunction"]])
df.head()

# 140 altı normal
# 140-199 gizli şeker
# 200 ve üzeri diyabet
df["Insulin_Category"] = pd.cut(x=df["Insulin"],
                           bins=[0, 140, 200, df["Insulin"].max()],
                           labels=["Normal", "Gizli_Şeker", "Diyabet"])

df["Insulin_Category"] = df["Insulin_Category"].fillna("Normal")
df = one_hot_encoder(df, ["Insulin_Category"], drop_first=True)
df = df.drop("Insulin", axis=1)
df.head()


#MODEL OLUŞTURMA
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train) #train veri setine göre fit ettik

# train hatası
y_pred = cart_model.predict(X_train) #train setinin bağımsız değişkenlerini yerine koyarak bağımlı değişkenlerini tahmin ettim
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob) #1.0 overfit

# test hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob) #modelin perforomansı
#0.7533928239449712


#karar ağacını görselleştirme işlemi
def tree_graph_to_png(tree, feature_names, png_file_to_save):
    tree_str = export_graphviz(tree, feature_names=feature_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(png_file_to_save)

tree_graph_to_png(tree=cart_model, feature_names=X_train.columns, png_file_to_save='cart.png')

#karar kuralları çıkarma işlemi
tree_rules = export_text(cart_model, feature_names=list(X_train.columns))
print(tree_rules)


#değişken önem düzeylerini incelemek
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(cart_model, X_train)

###############################
#HİPERPARAMETRE OPTİMİZSAYONU
###############################

#boş model nesnesi oluşturduk
cart_model = DecisionTreeClassifier(random_state=17)
# arama yapılacak hiperparametre setleri
cart_params = {'max_depth': range(1, 11),
               "min_samples_split": [2, 3, 4]}

#cross validation ile hiperparemetre araması yapacağız
#Yani hiperparametre araması yapılırken train seti üzerinde yapılır tüm veri üzerinde yapılmaz!
#model doğrulama yapılınca asla bütün veri kullanılmaz !!Yanlılık oluşturur!!!
cart_cv = GridSearchCV(cart_model, cart_params, cv=5, n_jobs=-1, verbose=True)
cart_cv.fit(X_train, y_train) #train setini çapraz doğrulamaya sokuyor



cart_tuned = DecisionTreeClassifier(**cart_cv.best_params_).fit(X_train, y_train)
#Model incelemesi
#** tüm parametreler
# train hatası
y_pred = cart_tuned.predict(X_train)
y_prob = cart_tuned.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)
#train setindeki hata :0.9123278443113773

#test  hatası
y_pred = cart_tuned.predict(X_test)
y_prob = cart_tuned.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)
#test hatamız :0.8587562744004462


################################
# FİNAL MODELİN YENİDEN TÜM VERİYE FİT EDİLMESİ
################################

cart_tuned_final = DecisionTreeClassifier(**cart_cv.best_params_).fit(X, y)

################################
# MODELİN DAHA SONRA KULLANILMAK ÜZERE KAYDEDİLMESİ
################################

import joblib
joblib.dump(cart_tuned_final, "cart_tuned_final.pkl")

cart_model_from_disk = joblib.load("cart_tuned_final.pkl")

cart_model_from_disk.predict(X_test)










