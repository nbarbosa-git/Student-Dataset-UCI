
# coding: utf-8

# In[1]:


import pandas
import itertools
import numpy as np
from patsy import dmatrices
from prettytable import PrettyTable
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import svm, metrics, tree
from sklearn import naive_bayes as nb
from sklearn import linear_model as lm
from sklearn import cross_validation as cv
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.externals.six import StringIO
import pydotplus 
from functools import reduce


# In[2]:

# Abrindo o arquivo
df = pandas.read_csv('student-por.csv', sep=';')

# Adaptando os dados (Relatorio 1)
del df['reason']
df['school']   =   df['school'].apply(lambda x: 0 if x == 'GP' else 1)
df['sex']      =      df['sex'].apply(lambda x: 0 if x == 'F' else 1)
df['address']  =  df['address'].apply(lambda x: 0 if x == 'R' else 1)
df['famsize']  =  df['famsize'].apply(lambda x: 0 if x == 'LE3' else 1)
df['Pstatus']  =  df['Pstatus'].apply(lambda x: 0 if x == 'T' else 1)
df['Mjob']     =     df['Mjob'].apply(lambda x: 0 if x == 'at_home' else 1)
df['Fjob']     =     df['Fjob'].apply(lambda x: 0 if x == 'at_home' else 1)
df['guardian'] = df['guardian'].apply(lambda x: 1 if x == 'other' else 0)
for i in ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
'internet', 'romantic']:
    df[i] = df[i].apply(lambda x: 1 if x == 'yes' else 0)


# In[3]:

# Pie Chart
def pie(df, v, title):
    t = df[v].value_counts().to_dict()
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'mediumslateblue']
    plt.pie(t.values(), labels=t.keys(), colors=colors, shadow=True, autopct='%1.1f%%')
    plt.axis('equal')
    plt.savefig(title + 'png', transparent=True)
    plt.close()


# In[4]:

# Preparando para Relatorio 2


#pie(df, 'Medu', 'Medu-5classes')
#pie(df, 'Fedu', 'Fedu-5classes')
df['Medu'] = df['Medu'].apply(lambda x: 1 if x == 0 else x)
df['Fedu'] = df['Fedu'].apply(lambda x: 1 if x == 0 else x)

#pie(df, 'Medu', 'Medu-4classes')
#pie(df, 'Fedu', 'Fedu-4classes')
del df['Fedu']
colunas = reduce(lambda x, y: x if y == 'Medu' else x + ' + ' + y, df.columns)
y, X = dmatrices('Medu ~ ' + colunas, df, return_type='dataframe')
y = np.ravel(y)
class_names = range(1,5)


# In[5]:

# Confusion Matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de Confusao',
cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(len(cm)):
        for k in range(len(cm[i])):
            cm[i][k] = round(cm[i][k]*100, 1)
    plt.imshow(cm, interpolation='nearest', cmap=cmap, clim=[0,100])
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.colorbar(ticks=[0,25,50,75,100])
    thresh = 70
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.ylabel('Classe correta')
    plt.xlabel('Classe predita')
tabela = PrettyTable(['Modelo', 'f1', 'Mean Squared Error'])


# In[6]:

# Regressao Logistica
# Modelo
logistic = lm.LogisticRegression().fit(X, y)
predicted = cv.cross_val_predict(logistic, X, y, cv=10)
# Cross Validation
scores = cv.cross_val_score(lm.LogisticRegression(), X, y, cv=10,
scoring='f1_weighted')
print ('Regressao Logistica')
print (scores.mean())
# Avaliacao
cnf_matrix = metrics.confusion_matrix(y, predicted)
cr = metrics.classification_report(y, predicted)
print (cr)
with open('cr.txt', 'w') as text_file:
    text_file.write(cr)
    text_file.write('\n')
mse = metrics.mean_squared_error(y, predicted)
tabela.add_row(['Regressao Logistica', scores.mean(), mse])
# Matriz de Confusao Normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
title='Matriz de Confusao Normalizada')
plt.savefig('cm_rl_n.png', transparent=True)
plt.close()
print ('\n')


# In[7]:

# Classificador Bayesiano (Multinomial)
# Modelo
bayes = nb.MultinomialNB()
bayes = bayes.fit(X, y)
predicted = cv.cross_val_predict(bayes, X, y, cv=10)
# Cross Validation
scores = cv.cross_val_score(nb.MultinomialNB(), X, y, cv=10, scoring='f1_weighted')
print ('bayesiano multinomial')
print (scores.mean())
# Avaliacao
cnf_matrix = metrics.confusion_matrix(y, predicted)
cr = metrics.classification_report(y, predicted)
print (cr)
with open('cr.txt', 'a') as text_file:
    text_file.write(cr)
    text_file.write('\n')
mse = metrics.mean_squared_error(y, predicted)
tabela.add_row(['Bayesiano Multinomial', scores.mean(), mse])
# Matriz de Confusao Normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
title='Matriz de Confusao Normalizada')
plt.savefig('cm_bayes_n.png', transparent=True)
plt.close()


# In[8]:

##MLP 2,2
## Modelo
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,2),
random_state=1)
mlp = mlp.fit(X, y)
predicted = cv.cross_val_predict(mlp, X, y, cv=10)
# Cross Validation
scores = cv.cross_val_score(MLPClassifier(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(2,2), random_state=1), X, y, cv=10, scoring='f1_weighted')
print ('redes neurais 22')
print (scores.mean())
mse = metrics.mean_squared_error(y, predicted)
tabela.add_row(['RN (2,2)', scores.mean(), mse])
# Avaliacao
cnf_matrix = metrics.confusion_matrix(y, predicted)
cr = metrics.classification_report(y, predicted)
print (cr)
with open('cr.txt', 'a') as text_file:
    text_file.write(cr)
    text_file.write('\n')
# Matriz de Confusao Normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
title='Matriz de Confusao Normalizada')
plt.savefig('cm_rn_n22.png', transparent=True)
plt.close()


# In[9]:

##MLP 5,5
## Modelo
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,5),
random_state=1)
mlp = mlp.fit(X, y)
predicted = cv.cross_val_predict(mlp, X, y, cv=10)
# Cross Validation
scores = cv.cross_val_score(MLPClassifier(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(5,5), random_state=1), X, y, cv=10, scoring='f1_weighted')
print ('redes neurais 55')
print (scores.mean())
mse = metrics.mean_squared_error(y, predicted)
tabela.add_row(['RN (5,5)', scores.mean(), mse])
# Avaliacao
cnf_matrix = metrics.confusion_matrix(y, predicted)
cr = metrics.classification_report(y, predicted)
print (cr)
with open('cr.txt', 'a') as text_file:
    text_file.write(cr)
    text_file.write('\n')
    
# Matriz de Confusao Normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
title='Matriz de Confusao Normalizada')
plt.savefig('cm_rn_n55.png', transparent=True)
plt.close()


# In[10]:

##MLP 10,10
## Modelo
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,10),
random_state=1)
mlp = mlp.fit(X, y)
predicted = cv.cross_val_predict(mlp, X, y, cv=10)
# Cross Validation
scores = cv.cross_val_score(MLPClassifier(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(20,20), random_state=1), X, y, cv=10, scoring='f1_weighted')
print ('redes neurais 1010')
print (scores.mean())
mse = metrics.mean_squared_error(y, predicted)
tabela.add_row(['RN (20,20)', scores.mean(), mse])
# Avaliacao
cnf_matrix = metrics.confusion_matrix(y, predicted)
cr = metrics.classification_report(y, predicted)
print (cr)
with open('cr.txt', 'a') as text_file:
    text_file.write(cr)
    text_file.write('\n')
# Matriz de Confusao Normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
title='Matriz de Confusao Normalizada')
plt.savefig('cm_rn_n1010.png', transparent=True)
plt.close()


# In[11]:

##MLP 20,20

## Modelo
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,20),
random_state=1)
mlp = mlp.fit(X, y)
predicted = cv.cross_val_predict(mlp, X, y, cv=10)
# Cross Validation
scores = cv.cross_val_score(MLPClassifier(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(20,20), random_state=1), X, y, cv=10, scoring='f1_weighted')
print ('redes neurais 2020')
print (scores.mean())
mse = metrics.mean_squared_error(y, predicted)
tabela.add_row(['RN (20,20)', scores.mean(), mse])
# Avaliacao
cnf_matrix = metrics.confusion_matrix(y, predicted)
cr = metrics.classification_report(y, predicted)
print (cr)
with open('cr.txt', 'a') as text_file:
    text_file.write(cr)
    text_file.write('\n')
# Matriz de Confusao Normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
title='Matriz de Confusao Normalizada')
plt.savefig('cm_rn_n2020.png', transparent=True)
plt.close()


# In[12]:

# SVM Linear
# Modelo
svc = svm.SVC(kernel='linear')
svc = svc.fit(X, y)
predicted = cv.cross_val_predict(svc, X, y, cv=10)
# Cross Validation
scores = cv.cross_val_score(svm.SVC(kernel='linear'), X, y, cv=10,
scoring='f1_weighted')
print ('svm linear')
print (scores.mean())
mse = metrics.mean_squared_error(y, predicted)
tabela.add_row(['SVM Linear', scores.mean(), mse])
# Avaliacao
cnf_matrix = metrics.confusion_matrix(y, predicted)
cr = metrics.classification_report(y, predicted)
print (cr)
with open('cr.txt', 'a') as text_file:
    text_file.write(cr)
    text_file.write('\n')
# Matriz de Confusao Normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
title='Matriz de Confusao Normalizada')
plt.savefig('cm_svm_l_n.png', transparent=True)
plt.close()


# In[13]:

# SVM Polinomial
# Modelo
poly_svc = svm.SVC(kernel='poly', degree=3)
poly_svc = poly_svc.fit(X, y)
predicted = cv.cross_val_predict(poly_svc, X, y, cv=10)
# Cross Validation
scores = cv.cross_val_score(svm.SVC(kernel='poly', degree=3), X, y, cv=10,
scoring='f1_weighted')
print ('svm polinomial')
print (scores.mean())
# Avaliacao
cnf_matrix = metrics.confusion_matrix(y, predicted)
cr = metrics.classification_report(y, predicted)
print (cr)
with open('cr.txt', 'a') as text_file:
    text_file.write(cr)
    text_file.write('\n')
mse = metrics.mean_squared_error(y, predicted)
tabela.add_row(['SVM Polinomial', scores.mean(), mse])
# Matriz de Confusao Normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
title='Matriz de Confusao Normalizada')
plt.savefig('cm_svm_p_n.png', transparent=True)
plt.close()


# In[14]:

# SVM RBF
# Modelo
rbf_svc = svm.SVC(kernel='rbf')
rbf_svc = rbf_svc.fit(X, y)
predicted = cv.cross_val_predict(rbf_svc, X, y, cv=10)
# Cross Validation
scores = cv.cross_val_score(svm.SVC(kernel='rbf'), X, y, cv=10,
scoring='f1_weighted')
print ('svm rbf')
print (scores.mean())
# Avaliacao
cnf_matrix = metrics.confusion_matrix(y, predicted)
cr = metrics.classification_report(y, predicted)
print (cr)
with open('cr.txt', 'a') as text_file:
    text_file.write(cr)
    text_file.write('\n')
mse = metrics.mean_squared_error(y, predicted)
tabela.add_row(['SVM RBF', scores.mean(), mse])
# Matriz de Confusao Normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
title='Matriz de Confusao Normalizada')
plt.savefig('cm_svm_rbf_n.png', transparent=True)
plt.close()


# In[15]:

# Decision Tree
# Modelo
clf = tree.DecisionTreeClassifier(random_state=1)
clf = clf.fit(X, y)
predicted = cv.cross_val_predict(clf, X, y, cv=10)
# Cross Validation
scores = cv.cross_val_score(clf, X, y, cv=10)
print ('decision tree')
print (scores.mean())
# Avaliacao
cnf_matrix = metrics.confusion_matrix(y, predicted)
cr = metrics.classification_report(y, predicted)
print (cr)
with open('cr.txt', 'a') as text_file:
    text_file.write(cr)
    text_file.write('\n')
mse = metrics.mean_squared_error(y, predicted)
tabela.add_row(['Decision Tree', scores.mean(), mse])
# Matriz de Confusao Normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
title='Matriz de Confusao Normalizada')
plt.savefig('cm_tree_n.png', transparent=True)
plt.close()


# In[16]:

# Decision Tree 3
# Modelo
clf = tree.DecisionTreeClassifier(max_depth=3, random_state=1)
clf = clf.fit(X, y)
predicted = cv.cross_val_predict(clf, X, y, cv=10)
# Cross Validation
scores = cv.cross_val_score(clf, X, y, cv=10)
print ('decision tree')
print (scores.mean())
# Avaliacao
cnf_matrix = metrics.confusion_matrix(y, predicted)
cr = metrics.classification_report(y, predicted)
print (cr)
with open('cr.txt', 'a') as text_file:
    text_file.write(cr)
    text_file.write('\n')
mse = metrics.mean_squared_error(y, predicted)
tabela.add_row(['Decision Tree (D=3)', scores.mean(), mse])


teste = list(df.columns[1::])
#del teste[6]
teste = [0] + teste
dot_data = tree.export_graphviz(clf,out_file = None,
class_names=['1','2','3','4'], rounded=True, feature_names=teste)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('teste3.pdf')





# Matriz de Confusao Normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
title='Matriz de Confusao Normalizada')
plt.savefig('cm_tree3_n.png', transparent=True)
plt.close()


# In[17]:

# Decision Tree 5
# Modelo
clf = tree.DecisionTreeClassifier(max_depth=5, random_state=1)
clf = clf.fit(X, y)
predicted = cv.cross_val_predict(clf, X, y, cv=10)
# Cross Validation
scores = cv.cross_val_score(clf, X, y, cv=10)
print ('decision tree')
print (scores.mean())


# Avaliacao
cnf_matrix = metrics.confusion_matrix(y, predicted)
cr = metrics.classification_report(y, predicted)
print (cr)
with open('cr.txt', 'a') as text_file:
    text_file.write(cr)
    text_file.write('\n')
mse = metrics.mean_squared_error(y, predicted)
tabela.add_row(['Decision Tree (D=5)', scores.mean(), mse])


#teste = list(df.columns[1::])
#del teste[6]
teste = [0] + teste
dot_data = tree.export_graphviz(clf,out_file = None,
class_names=['1','2','3','4'], rounded=True, feature_names=teste)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('teste5.pdf')



# Matriz de Confusao Normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
title='Matriz de Confusao Normalizada')
plt.savefig('cm_tree5_n.png', transparent=True)
plt.close()


# In[ ]:

# Random Forest
# Modelo
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(X, y)
predicted = cv.cross_val_predict(clf, X, y, cv=10)
# Cross Validation
scores = cv.cross_val_score(clf, X, y, cv=10)
print ('Random Forest 100')
print (scores.mean())
# Avaliacao
cnf_matrix = metrics.confusion_matrix(y, predicted)
cr = metrics.classification_report(y, predicted)
print (cr)
with open('cr.txt', 'a') as text_file:
    text_file.write(cr)
    text_file.write('\n')
mse = metrics.mean_squared_error(y, predicted)
tabela.add_row(['Random Forest 100', scores.mean(), mse])
# Matriz de Confusao Normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
title='Matriz de Confusao Normalizada')
plt.savefig('randomforest100.png', transparent=True)
plt.close()


# In[ ]:

# Random Forest 500
# Modelo
clf = RandomForestClassifier(n_estimators=500)
clf = clf.fit(X, y)
predicted = cv.cross_val_predict(clf, X, y, cv=10)
# Cross Validation
scores = cv.cross_val_score(clf, X, y, cv=10)
print ('Random Forest 500')
print (scores.mean())
# Avaliacao
cnf_matrix = metrics.confusion_matrix(y, predicted)
cr = metrics.classification_report(y, predicted)
print (cr)
with open('cr.txt', 'a') as text_file:
    text_file.write(cr)
    text_file.write('\n')
mse = metrics.mean_squared_error(y, predicted)
tabela.add_row(['Random Forest 500', scores.mean(), mse])
# Matriz de Confusao Normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
title='Matriz de Confusao Normalizada')
plt.savefig('randomforest500.png', transparent=True)
plt.close()
tabela.align = "l"
print (tabela)
with open('tabela.txt', 'w') as text_file: text_file.write(tabela.get_string())


# In[ ]:

df


# In[ ]:

teste[1::]
del teste[6]
teste


# In[ ]:

graph


# In[ ]:

teste


# In[ ]:

clf


# In[ ]:

y


# In[ ]:



