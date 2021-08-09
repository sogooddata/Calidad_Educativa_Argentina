#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('bmh')


#Datasets#
ds16= pd.read_csv('Downloads/raw/Estudiante_5-6 año Secundaria 2016.csv', sep= ',')
dict_2016 = pd.read_excel('Downloads/raw/Diccionarios/Dic_2016_Sec.xlsx')

#Bases Anexas#

#pob_escuelas = pd.read_csv('Downloads/gobiernoslocales_2020/poblacion.csv', sep=',')
#q_escuelas = pd.read_csv('Downloads/gobiernoslocales_2020/escuelas.csv', sep=',')
#dpto_arg = pd.read_csv('Downloads/gobiernoslocales_2020/departamentos.csv', sep=',')
#mun = pd.read_csv('Downloads/gobiernoslocales_2020/ign_municipio.csv', sep=';')
#dptos = pd.read_csv('Downloads/gobiernoslocales_2020/ign_departamento.csv', sep=';')
#dtos_match = pd.read_csv('Downloads/gobiernoslocales_2020/Dpto_transformado.csv', sep=';')
dptos_merge = pd.read_csv('Downloads/gobiernoslocales_2020/Dptos BsAs Mod.csv', sep= ';')

print("dataset total",ds16.shape)
print("dataset total", ds16.size)
print("diccionario",dict_2016.shape)
print("diccionario", dict_2016.size)

#Muestra BsAs y CABA#
val_filt = [2,6]
df = ds16[ds16.cod_provincia.isin(val_filt)]
df = df.infer_objects()
#df = ds16
print("muestra", df.shape)
print("muestra", df.size)

df2 = df.merge(dptos_merge, on =['Municipio'], how ='left',indicator = True) 
df2.size


# In[2]:


def plot_multiple_vars(df,x,y,title='Titulo de referencia',label= False):
    suma = sum(df[y].value_counts())
    suma2 = round((df[y].value_counts()/suma),2) 
    suma3 = round(suma2  ,2)
    data = suma3.to_frame().reset_index().sort_values(y)

    if list(label):
        label = label[label.Variable==y].set_index('Códigos')
        data['index'] =data['index'].map(label['Codigo_Et']) 
        
    plt.figure(figsize=(20,10))
    
    sns.set(style="whitegrid")
    ax = sns.barplot(data = data , 
                     x =x,y=y,order = data['index']
                    )



    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    if list(label):
        ax.set(xlabel= label['Variabl_Et'].unique().item(), ylabel='Count')
    else:
        ax.set(xlabel= y, ylabel='Count')
    # add proper Dim values as x labels
    for item in ax.get_xticklabels(): item.set_rotation(45)
    for i, v in enumerate(data[y].iteritems()):        
        ax.text(i ,v[1], "{:.0%}".format(v[1]), color='k', va ='bottom', rotation=45, fontsize=(20))
    plt.tight_layout()
    plt.title(title,size =20)
    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small') 
    return plt.show()

variables_explicitas =dict_2016[dict_2016.Variable.isin(ds16.columns)]
variables_explicitas.head()
len(variables_explicitas.Variable.unique())


# In[3]:


#val_filt = ['Privada']
#muestra = df[df.sector.isin(val_filt)]
#muestra= muestra.infer_objects()
plot_multiple_vars(df2, x ='index',y='Ap40e',title='',label = dict_2016)


# In[4]:


missing_ratio = df.isnull().sum().sum() / df.size
print("missing ratio", round(missing_ratio,2))


# In[21]:


#Estadística Descriptiva#
val_filt = [82]
muestra = df[df.cod_provincia.isin(val_filt)]
muestra= muestra.infer_objects()

corr = muestra.corr(method = 'kendall')
export_excel2= corr.to_excel('Downloads/Outputs/correlacKStaFe.xlsx',index= True, header=True)
#Desc_muestra = df.describe(percentiles= [.1,.25,.3,.5,.6,.7,.8,.9],include= 'all' ).transpose()
#export_excel2= Desc_muestra.to_excel('Downloads/Outputs/SummAprenderTotal.xlsx',index= True, header=True)


# In[32]:


# Gráfico de tarta de pasajeros del Titanic
val_filt = [78]
muestra = df[df.cod_provincia.isin(val_filt)]
muestra= muestra.infer_objects()

plot = muestra['ldesemp'].value_counts().plot(kind='pie', autopct='%.2f', 
                                            figsize=(6, 6))
plt.ylabel(ylabel='')
plt.suptitle('Desempeño Matemática en Formosa')


# In[30]:


df2 = df.merge(dptos_merge, on =['Municipio'], how ='left',indicator = True) 
df2.head()


# In[44]:


var = 'Ap7'
ap = 'Subcategory'
df2=df2
variables_explicitas.loc[variables_explicitas.Variable == ap]
secundario_2016_sector_provincia = df2[[ap,var]].copy()

#secundario_2016_sector_provincia[ap] = secundario_2016_sector_provincia[ap]\
 #                               .map(variables_explicitas.loc[variables_explicitas.Variable == ap]\
  #                              .set_index('Códigos')['Codigo_Et'])

secundario_2016_sector_provincia[var] = secundario_2016_sector_provincia[var]                                .map(variables_explicitas.loc[variables_explicitas.Variable == var]                                .set_index('Códigos')['Codigo_Et'])

secundario_2016_sector_provincia['count'] = 1

groupby_secundario_2016_sector_provincia = secundario_2016_sector_provincia                                            .groupby([ap,var]).count().reset_index()
    
groupby_secundario_2016_sector_provincia.groupby(var).sum()['count']

#
data = (groupby_secundario_2016_sector_provincia.set_index([var,
                                                     ap])['count'] / groupby_secundario_2016_sector_provincia.groupby(ap).sum()['count'])\
.to_frame().reset_index()


#
data= groupby_secundario_2016_sector_provincia.pivot(index=var,columns=ap,values='count')
data.plot(kind='barh' , figsize=(20,10), label=dict_2016, fontsize=(20))

params = {'legend.fontsize': 20,
          'legend.handlelength': 3}

plt.rcParams.update(params)

plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


# In[7]:


var = 'Ap45'
ap = 'Subcategory'
df2=df2
variables_explicitas.loc[variables_explicitas.Variable == ap]
secundario_2016_sector_provincia = df2[[ap,var]].copy()

#secundario_2016_sector_provincia[ap] = secundario_2016_sector_provincia[ap]\
 #                               .map(variables_explicitas.loc[variables_explicitas.Variable == ap]\
  #                              .set_index('Códigos')['Codigo_Et'])

secundario_2016_sector_provincia[var] = secundario_2016_sector_provincia[var]                                .map(variables_explicitas.loc[variables_explicitas.Variable == var]                                .set_index('Códigos')['Codigo_Et'])

secundario_2016_sector_provincia['count'] = 1

groupby_secundario_2016_sector_provincia = secundario_2016_sector_provincia                                            .groupby([ap,var]).count().reset_index()
    
groupby_secundario_2016_sector_provincia.groupby(var).sum()['count']

#
data = (groupby_secundario_2016_sector_provincia.set_index([var,
                                                     ap])['count'] / groupby_secundario_2016_sector_provincia.groupby(ap).sum()['count'])\
.to_frame().reset_index()


# In[92]:


#missing_ratio = df.isnull().sum().sum() / df.size


#data= groupby_secundario_2016_sector_provincia.pivot(index=var,columns=ap,values='count')
#data.plot(y= ap , values = 'count' , kind= 'barh', stacked = True)
#data.plot(y = ap, kind='barh' , figsize=(20,10), label=dict_2016, fontsize=(20), stacked = True)


# In[8]:


export_excel2 = data.to_excel('Downloads/Outputs/test3ultpost.xlsx', index = True, header=True)


# In[7]:


pivot = df2[['lpuntaje','mpuntaje','ldesemp','mdesemp']].copy()
export_excel2= pivot.to_excel('Downloads/Outputs/verlim.xlsx',index= True, header=True)


# In[12]:


#Tabla tabulada de proporciones
from tabulate import tabulate
df2=df
preg = 'ldesemp'
filt1 = ['CABA']
filt2 = ['AMBA sin CABA']
filt3 = ['Interior GBA']
caba = df2[df2.Subcategory.isin(filt1)]
gba = df2[df2.Subcategory.isin(filt2)]
interior = df2[df2.Subcategory.isin(filt3)]

privado_caba = caba.sector.value_counts()[2.0]
privado_gba = gba.sector.value_counts()[2.0]
privado_int = interior.sector.value_counts()[2.0]
pub_caba = caba.sector.value_counts()[1.0]
pub_gba = gba.sector.value_counts()[1.0]
pub_int = interior.sector.value_counts()[1.0]
m1 = privado_caba + pub_caba + privado_gba + pub_gba
m3 = privado_int + pub_int
mtot = privado_caba + pub_caba + privado_gba + pub_gba + privado_int + pub_int


data = [("Privado AMBA", (privado_caba + privado_gba) / m1),
       ("Privado Int GBA", (privado_int) / m3),
       ("Publico AMBA", (pub_caba+ pub_gba) / m1),
       ("Publico Int GBA", (pub_int)/m3)] 

dataset= tabulate(data)

print(dataset)


# In[24]:



t = df[['ldesemp','Ap41','lpuntaje']]
export_excel2= t.to_excel('Downloads/Celpropio.xlsx',index= True, header=True)


# In[6]:


#Regresion Nivel Ecnómico y Desempeño#
from scipy import stats
muestra= df[['isocioa','mpuntaje','lpuntaje'].copy()]
muestra.dropna(inplace=True)

# dataframe containing only females
df_females = muestra[muestra['isocioa'] == 1.0]
# pearson correlation coefficient and p-value
pearson_coef, p_value = stats.pearsonr(df_females.lpuntaje, df_females.mpuntaje)
print(pearson_coef)
# dataframe containing only males
df_males = muestra[muestra['isocioa'] == 2.0]
# pearson correlation coefficient and p-value
pearson_coef, p_value = stats.pearsonr(df_males.lpuntaje, df_males.mpuntaje)
print(pearson_coef)
df_males2 = muestra[muestra['isocioa'] == 3.0]
# pearson correlation coefficient and p-value
pearson_coef, p_value = stats.pearsonr(df_males2.lpuntaje, df_males2.mpuntaje)
print(pearson_coef)


df_bajo = muestra[muestra['isocioa'] == 1.0].sample(550)
df_medio = muestra[muestra['isocioa'] == 2.0].sample(550)
df_alto = muestra[muestra['isocioa']== 3.0].sample(550)

df_bajo.dropna(inplace=True)
df_medio.dropna(inplace=True)
df_alto.dropna(inplace=True)

# Scatter plots.
ax1 = df_bajo.plot(kind='scatter', x='lpuntaje', y='mpuntaje', color='blue', alpha=0.5, figsize=(10, 7))
ax1 =df_medio.plot(kind='scatter', x='lpuntaje', y='mpuntaje', color='magenta', alpha=0.5, figsize=(10, 7), ax=ax1)
df_alto.plot(kind='scatter', x='lpuntaje', y='mpuntaje', color='black', alpha=0.5, figsize=(10,7), ax=ax1)


# In[8]:


df_bajo = df[df['cod_provincia'] == 6.0].sample(3550)
df_medio = df[df['cod_provincia'] == 2.0].sample(3550)

df_bajo.dropna(inplace=True)
df_medio.dropna(inplace=True)

# Scatter plots.
ax1 = df_bajo.plot(label = 'Buenos Aires', kind='scatter', x='lpuntaje', y='mpuntaje', color='darkblue', alpha=0.5, figsize=(10, 7))
ax1 =df_medio.plot(label = 'CABA',kind='scatter', x='lpuntaje', y='mpuntaje', color='magenta', alpha=0.5, figsize=(10, 7), ax=ax1)


# Feature Importance

# In[ ]:


from sklearn.model_selection import train_test_split
#Preparo Dataset#
df3 = df.copy()
delete = pd.read_csv('Downloads/elimina2.csv', sep= ';')
col_list = delete['del']
df3.drop(col_list,axis=True,inplace=True)
df3.shape

df3 = df3.replace('nan','NaN')
df3 = df3.fillna(0)
data = df3.dropna() 

features = [feat for feat in list(data) 
            if feat != 'lpuntaje']
datamat = np.array(features)
X, y = data[features], data.lpuntaje
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size= 0.25)


# In[17]:


X_test.info()


# In[20]:


#RandomForest#
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import sklearn


clf = RandomForestRegressor(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Accuracy on test data: {:.2f}".format(clf.score(X_test, y_test)))


# In[21]:


from sklearn.model_selection import cross_val_score

#clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
scores


# In[ ]:


print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[10]:


result = permutation_importance(clf, X_train, y_train, n_repeats=10, random_state=42) 
perm_sorted_idx = result.importances_mean.argsort()

tree_importance_sorted_idx = np.argsort(clf.feature_importances_) 
tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(50, 24))
ax1.barh(tree_indices,
         clf.feature_importances_[tree_importance_sorted_idx], height=0.7)
ax1.set_yticklabels(datamat[tree_importance_sorted_idx])
ax1.set_yticks(tree_indices)
ax1.set_ylim((0, len(clf.feature_importances_)))
ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
            labels=datamat[perm_sorted_idx])
fig.tight_layout()
plt.show()


# In[6]:


#XGBoost#

from numpy import loadtxt
from xgboost import XGBClassifier, plot_importance
from matplotlib import pyplot

# split data into X and y
# fit model no training data
model = XGBClassifier()
model.fit(X, y)
# feature importance
print(model.feature_importances_)
# plot
# plot feature importance
plot_importance(model, max_num_features = 20)
pyplot.show()

plot_importance(model, importance_type='gain', max_num_features = 20)
pyplot.show()


# In[32]:


#Modelo#
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

feature_cols = datamat
#X, y = data[features], data.lpuntaje
X,y = X_transformed2, data.lpuntaje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

clf = DecisionTreeRegressor(max_depth = 6)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

clf.score(X,y)
clf.get_depth()
clf.get_n_leaves()
clf.get_params()


# In[18]:


clf.get_depth()


# In[33]:


clf.score(X_test,y_test)


# In[31]:


# evaluate decision tree performance on train and test sets with different tree depths
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
# create dataset
#X, y = data[features], data.mpuntaje
X,y = X_transformed2, data.lpuntaje
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
# define lists to collect scores
train_scores, test_scores = list(), list()
# define the tree depths to evaluate
values = [i for i in range(1, 15)]
# evaluate a decision tree for each depth
for i in values:
    # configure the model
    model = DecisionTreeRegressor(max_depth = i)
    # fit model on the training dataset
    model.fit(X_train, y_train)
    # evaluate on the train dataset
    #train_yhat = model.predict(X_train)
    train_acc = model.score(X_train, y_train)
    train_scores.append(train_acc)
    # evaluate on the test dataset
    #test_yhat = model.predict(X_test)
    test_acc = model.score(X_test, y_test)
    test_scores.append(test_acc)
    # summarize progress
    print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
# plot of train and test scores vs tree depth
pyplot.plot(values, train_scores, '-o', label='Train')
pyplot.plot(values, test_scores, '-o', label='Test')
pyplot.legend()
pyplot.show()


# In[149]:


clf.score(X_test,y_test)


# In[34]:


#print(dict(zip(datamat, clf.feature_importances_)))
clf.feature_importances_


# In[30]:


#Kolmogorov Smirnov test#
from scipy import stats

t = df[['Ap48k','mpuntaje','mdesemp']]

x = t.Ap48k
y = t.mdesemp

stats.kstest(x, y)


# In[14]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 0.90)
pca.fit(X)
reduced = pca.transform(X)
reduced.shape


# In[59]:



from sklearn.decomposition import FactorAnalysis

transformer = FactorAnalysis(n_components=19,random_state=0)
X_transformed = transformer.fit_transform(X)
X_transformed.shape


# In[51]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 0.95)
pca.fit(X)
reduced = pca.transform(X)

pca = PCA().fit(X)

plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
xi = np.arange(0, 170, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 11, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()


# In[15]:


#mismo PCA factor analyzer pero con otra librería"
from factor_analyzer import FactorAnalyzer
transformer2 = FactorAnalyzer(n_factors=3, rotation = 'varimax')
X_transformed2 = transformer2.fit_transform(X)


# In[16]:


pc2 = pd.DataFrame(transformer2.loadings_, index=datamat)
export_excel2 = pc2.to_excel('Downloads/pcvartotpais2021.xlsx', index=True, header= True)


# In[ ]:


pcaex = pd.DataFrame(transformer.components_ ,columns=datamat)
export_excel2= pcaex.to_excel('Downloads/pcaex.xlsx',index= True, header=True)


# In[57]:


transformer.get_precision()


# In[67]:


transformer.score_samples(X, y)

