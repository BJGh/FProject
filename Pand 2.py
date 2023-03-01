import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.manifold import TSNE, MDS
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import time
from tqdm.notebook import tqdm
import gensim
import gensim.corpora as corpora
from gensim.models import LsiModel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.preprocessing import OneHotEncoder
start=time.time()

test= pd.read_json(r'') #r'K:\mag\наработки\АНАлиз\ML\no_hash_train.json'
print(test)

test2=pd.read_json('ha2_diff.json')
#copy=test2.copy()
#test2=test2.append(copy,ignore_index=True)
print(test2)

def to_pd_categorical(data):
  """change the format of data to categorical"""
  X = data.copy()
  for col in X.columns:
    X[col] = pd.Categorical(X[col])
  return X

def categorical_to_texts(data):
  """Transform categorical data rows into texts as follows:
  “Var1Name_Value1 Var2Name_Value2 …”
  """
  new_data_list = []
  columns = list(data.columns)
  for line in tqdm(data.values):
    new_line = ''
    for pair in zip(columns, line):
      new_line = new_line + f'{pair[0]}_{pair[1]} '
    new_data_list.append(new_line)
  return new_data_list


test2.drop(['TLSH','TLSHORIG','diff','tlsh_hash'],axis=1,inplace=True)



for i in test.index:
    test.loc[i,['id']]=int(i)
for i in test2.index:
    test2.loc[i,['id']]=int(i)
x=test[test['id']==402]
y=test[test['id']==403]
z=test[test['id']==404]
print(x.at[402,'user-agent'])
print(y.at[403,'user-agent'])
print(z.at[404,'user-agent'])




#test.to_json(r'K:\mag\наработки\АНАлиз\ML\no_hash_full_id.json')
#test.to_json(r'K:\mag\наработки\АНАлиз\ML\no_hash_test_6000.json')



categorical_cols = [col for col in test2.columns if col != 'id']

X_train = test[categorical_cols]
y_train = test['id']
X_test = test2[categorical_cols]
y_test = test2['id']

X_train_texts = categorical_to_texts(X_train)
X_test_texts = categorical_to_texts(X_test)
def vect2gensim(vectorizer, dtmatrix):
    """ transform sparse matrix into gensim corpus and dictionary """
    corpus_vect_gensim = gensim.matutils.Sparse2Corpus(dtmatrix, documents_columns=False)
    dictionary = gensim.corpora.dictionary.Dictionary.from_corpus(corpus_vect_gensim,
        id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))

    return (corpus_vect_gensim, dictionary)
def get_topic_vectors(model,corpus):
  """Get new features form the topic LSA model and gensim corpus"""
  num_topics = len(model[corpus[1]])
  doc_vectors = np.zeros((len(corpus), num_topics))
  for i, doc in enumerate(tqdm(corpus)):
    topics = model[doc]
    for pair in topics:
      j = pair[0]
      doc_vectors[i,j] = pair[1]
  return doc_vectors

#-----------------load model test
''''
loaded_model = LsiModel.load('lsi.model')
vectorizer = CountVectorizer(ngram_range=(1, 2)) #min_df=10,max_df=0.2
bow_matrix = vectorizer.fit_transform(X_train_texts)

#(gensim_corpus, gensim_dict) = vect2gensim(vectorizer, bow_matrix)

gensim_dict = corpora.Dictionary.load('dict.dict')
gensim_corpus = corpora.MmCorpus('corp.corp')
bow_matrix_test = vectorizer.fit_transform(X_test_texts)
gensim_corpus_test,_ = vect2gensim(vectorizer, bow_matrix_test)
print("gensim corpus matr")

lsa_features_train = get_topic_vectors(loaded_model, gensim_corpus)
lsa_features_test = get_topic_vectors(loaded_model, gensim_corpus_test)
min=10000000
a=lsa_features_test[64]


i=0

for i in range(len(lsa_features_test)):
    dst=0
    min=1000000
    a = lsa_features_test[i]
    j=0
    for j in range(len(lsa_features_train)):
        if j !=i:
            b = lsa_features_train[j]
            dst=distance.cosine(a,b)
            if dst<min:
                min=dst
                print(f'Раст {min} ,  номер test {i} с train {j}')



    #print(dst,i)




_, emb_sample, _, y_target = train_test_split(lsa_features_train , y_train, test_size=0.7)
_, emb_sample2, _, y_target2 = train_test_split(lsa_features_test , y_test, test_size=0.9)
print(len(emb_sample))
print(lsa_features_train[22])
print(lsa_features_test[22])
scaler = StandardScaler()
scaler2 = StandardScaler()
scaler.fit(lsa_features_test)
#scaler2.fit(lsa_features_train)
scaled_emb = scaler.transform(emb_sample)
scaled2_emb = scaler.transform(emb_sample2)
tsne = TSNE(n_components=2, random_state=33)
T = tsne.fit_transform(scaled_emb)
E= tsne.fit_transform(scaled2_emb)
plt.figure(figsize=(12,8))
plt.scatter(T[:,0], T[:,1], alpha=0.7, c = '#FF8C00')
plt.scatter(E[:,0], E[:,1], alpha=0.7, c = '#00008B', s=100)
labels = y_test
#for label, x, y in zip(labels, T[:,0], T[:,1]):
#    plt.annotate(label, (x,y), xycoords = 'data')
plt.grid()
plt.show()

time.sleep(100)
'''

#------------------train model
vectorizer = CountVectorizer(ngram_range=(1, 2),min_df=10,max_df=0.2)
bow_matrix = vectorizer.fit_transform(X_train_texts)

(gensim_corpus, gensim_dict) = vect2gensim(vectorizer, bow_matrix)

bow_matrix_test = vectorizer.transform(X_test_texts)
gensim_corpus_test,_ = vect2gensim(vectorizer, bow_matrix_test)
gensim_dict.save('G:\Models\dict_2000.dict')
corpora.MmCorpus.serialize('G:\Models\corp_2000.corp', gensim_corpus)
print("gensim corpus matr save \n start lsi")


lsamodel = gensim.models.LsiModel(gensim_corpus, num_topics=2000, id2word = gensim_dict, power_iters=30)
lsamodel.save("G:\Models\lsi_clear_2000.model")
print("lsamodel")
end=time.time()
razn=(end-start)/60
print(razn)
#lsa_features_train = get_topic_vectors(lsamodel, gensim_corpus)
#lsa_features_test = get_topic_vectors(lsamodel, gensim_corpus_test)


print("Start train logistic reg")


''''
model_lr = LogisticRegression().fit(lsa_features_train, y_train)

pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model_lr, file)
print("end train model")
probs = model_lr.predict_proba(lsa_features_test)
score_lr_lsi = roc_auc_score(y_test, probs[:,1])
print(score_lr_lsi)
time.sleep(100)



features = pd.get_dummies(test)
#features2=pd.get_dummies(test2)

feature_list = list(features.columns)

#feature_list2=list(features2.columns)


for i in features.index:
    features.loc[i,['id']]=int(i)
#for i in features2.index:
#    features2.loc[i,['id']]=22



print(features)
#print(features2)

#Z1=features2.loc[:,'color_depth':'WindowText_rgb(0, 0, 0)м0']
#Z2=features2['id']

#определяем исходные данные X и таргеты Y
X=features.loc[:,'color_depth':'WindowText_rgb(53, 64, 74)']
Y=features['id']

#разделяем исходные данные на тренировочную и тестовую выборку
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestRegressor

#инициализируем модель и обучаем ее
clf=RandomForestRegressor(n_estimators = 30, random_state = 42)
clf=clf.fit(X_train,Y_train)

#извлекаем оценку точности модели
print("Accuracy train set: {:.3f}".format(clf.score(X_train,Y_train)))
print("Accuracy test set: {:.3f}".format(clf.score(X_test,Y_test)))

y_pred=clf.predict(X_test)
print(y_pred)
print(Y_train)



importances = list(clf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
#print(feature_importances[0:10])

#features.to_excel(r'K:\mag\наработки\АНАлиз\ML\no_hash_test_dum.xlsx')

print(test)
for i in range(6000):
    try:
        test = test.drop([i])
    except:
        print(i)
print(test)
test.to_excel(r'K:\mag\наработки\АНАлиз\ML\no_hash_test_hot.xlsx')
test.to_json(r'K:\mag\наработки\АНАлиз\ML\no_hash_test_hot.json')

#print(features.iloc[:,7:].head(7))
'''