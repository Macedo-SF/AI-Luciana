from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import plot_confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys

sys.stdout = open('C:/Users/Saulo/source/repos/AI Luciana/output.txt','w')

#sem seleção
wine = datasets.load_wine()#pequena mudança, antes só 12 dos 13 features eram usados
df = pd.DataFrame(wine.data, columns = wine.feature_names)

#print(df.keys())
#'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
#'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
#'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
X_training, X_testing, y_training, y_testing = train_test_split(df, wine.target, test_size=0.4)

clf = MLPClassifier(alpha=0.01,max_iter=5000, hidden_layer_sizes=20,activation='relu')
clf.fit(X_training, y_training)

yp = clf.predict(X_testing)

#titles_options = [("Matriz de confusão, sem normalizar", None), ("Matriz de
#confusão normalizada", 'true')]
titles_options = [("Matriz de confusão normalizada", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, X_testing, y_testing,cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    plt.savefig('Figures/' + title + '_semSelecao.png', dpi=400)
    plt.clf()
#plt.show()
print("MLP Classifier all-features\n",classification_report(y_testing, yp))

#com seleção
df['classe'] = wine.target
correlacoes = df.corr()
fig, ax = plt.subplots(figsize=(20,15))         
heat = sns.heatmap(correlacoes, annot = True,ax=ax)#cmap="YlGnBu"
figure = heat.get_figure()
figure.savefig('Figures/heatmap.png', dpi=400)
#checando as combinações
"""
names=df.keys()
for x in range(12):
    for y in range(12):
        if(y<x):
            continue
        wine_n = np.array(df[[names[x], names[y+1]]])
        Xn_training, Xn_testing, yn_training, yn_testing = train_test_split(wine_n, wine.target, test_size=0.4)

        clfn = MLPClassifier(alpha=0.01,max_iter=5000, hidden_layer_sizes=20,activation='relu')
        clfn.fit(Xn_training, yn_training)

        ypn = clfn.predict(Xn_testing)

        for title, normalize in titles_options:
            disp = plot_confusion_matrix(clfn, Xn_testing, yn_testing,cmap=plt.cm.Blues,normalize=normalize)
            disp.ax_.set_title(title)
            plt.savefig('Figures/'+title+str(x)+'_'+str(y+1)+'.png', dpi=400)

        print(names[x],names[y+1],'\n',classification_report(yn_testing, ypn))
"""
#Logistica
wine_log = np.array(df[['alcalinity_of_ash', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'hue', 'od280/od315_of_diluted_wines', 'proline']])
X_training_log, X_testing_log, y_training_log, y_testing_log = train_test_split(wine_log, wine.target, test_size=0.4)

clf_log = LogisticRegression(max_iter=5000)
clf_log.fit(X_training_log, y_training_log)

yp_log = clf_log.predict(X_testing_log)

for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf_log, X_testing_log, y_testing_log,cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    plt.savefig('Figures/' + title + '_log.png', dpi=400)
#plt.show()
print("RegLog all-features\n",classification_report(y_testing_log, yp_log))

#Logistica 2
wine_slog = np.array(df[['alcalinity_of_ash', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'hue', 'od280/od315_of_diluted_wines', 'proline']])
X_training_slog, X_testing_slog, y_training_slog, y_testing_slog = train_test_split(wine_slog, wine.target, test_size=0.4)

clf_slog = LogisticRegression(max_iter=5000)
clf_slog.fit(X_training_slog, y_training_slog)

yp_slog = clf_slog.predict(X_testing_slog)

for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf_slog, X_testing_slog, y_testing_slog,cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    plt.savefig('Figures/' + title + '_slog.png', dpi=400)
#plt.show()
print("RegLog\n",classification_report(y_testing_slog, yp_slog))

#MP 1
wine_mp1 = np.array(df[['alcohol','total_phenols', 'flavanoids','proanthocyanins','hue', 'od280/od315_of_diluted_wines']])
X_training_mp1, X_testing_mp1, y_training_mp1, y_testing_mp1 = train_test_split(wine_mp1, wine.target, test_size=0.4)

clf_mp1 = MLPClassifier(alpha=0.01,max_iter=5000, hidden_layer_sizes=20,activation='relu')
clf_mp1.fit(X_training_mp1, y_training_mp1)

yp_mp1 = clf_mp1.predict(X_testing_mp1)

for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf_mp1, X_testing_mp1, y_testing_mp1,cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    plt.savefig('Figures/' + title + '_mp1.png', dpi=400)
#plt.show()
print("MLP Classifier 1\n",classification_report(y_testing_mp1, yp_mp1))

#MP 2
wine_mp2 = np.array(df[['alcohol', 'flavanoids']])
X_training_mp2, X_testing_mp2, y_training_mp2, y_testing_mp2 = train_test_split(wine_mp2, wine.target, test_size=0.4)

clf_mp2 = MLPClassifier(alpha=0.01,max_iter=5000, hidden_layer_sizes=20,activation='relu')
clf_mp2.fit(X_training_mp2, y_training_mp2)

yp_mp2 = clf_mp2.predict(X_testing_mp2)

for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf_mp2, X_testing_mp2, y_testing_mp2,cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    plt.savefig('Figures/' + title + '_mp2.png', dpi=400)
#plt.show()
print("MLP Classifier 2\n",classification_report(y_testing_mp2, yp_mp2))

#Naive Bayes
wine_nb = np.array(df[['alcohol','total_phenols', 'flavanoids','proanthocyanins','hue', 'od280/od315_of_diluted_wines']])
X_training_nb, X_testing_nb, y_training_nb, y_testing_nb = train_test_split(wine_nb, wine.target, test_size=0.4)

clf_nb = GaussianNB()
clf_nb.fit(X_training_nb, y_training_nb)

yp_nb = clf_nb.predict(X_testing_nb)

for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf_nb, X_testing_nb, y_testing_nb,cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    plt.savefig('Figures/' + title + '_nb.png', dpi=400)
#plt.show()
print("Naive Bayes\n",classification_report(y_testing_nb, yp_nb))

#Decision Tree
wine_dt = np.array(df[['alcohol','total_phenols', 'flavanoids','proanthocyanins','hue', 'od280/od315_of_diluted_wines']])
X_training_dt, X_testing_dt, y_training_dt, y_testing_dt = train_test_split(wine_dt, wine.target, test_size=0.4)

clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_training_dt, y_training_dt)

yp_dt = clf_dt.predict(X_testing_dt)

for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf_dt, X_testing_dt, y_testing_dt,cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    plt.savefig('Figures/' + title + '_dt.png', dpi=400)
#plt.show()
print("Decision Tree\n",classification_report(y_testing_dt, yp_dt))

#KNeighboors
wine_kn = np.array(df[['alcohol','total_phenols', 'flavanoids','proanthocyanins','hue', 'od280/od315_of_diluted_wines']])
X_training_kn, X_testing_kn, y_training_kn, y_testing_kn = train_test_split(wine_kn, wine.target, test_size=0.4)

clf_kn = KNeighborsClassifier()
clf_kn.fit(X_training_kn, y_training_kn)

yp_kn = clf_kn.predict(X_testing_kn)

for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf_kn, X_testing_kn, y_testing_kn,cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    plt.savefig('Figures/' + title + '_kn.png', dpi=400)
#plt.show()
print("KNeighboors\n",classification_report(y_testing_kn, yp_kn))

#KMeans
data = datasets.load_wine().data
pca = PCA(2)
DF = pca.fit_transform(data)

kmeans = KMeans(n_clusters= 3)
label = kmeans.fit_predict(DF)
 
centroids = kmeans.cluster_centers_
u_labels = np.unique(label)

plt.clf()
#plotting the results:
for i in u_labels:
    plt.scatter(DF[label == i , 0] , DF[label == i , 1] , label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.title("KMeans")
plt.savefig('Figures/KMeans.png', dpi=400)

#Linear Reg
tp = np.array(df['total_phenols']).reshape(-1, 1)
fl = np.array(df['flavanoids'])
X_training_lr, X_testing_lr, y_training_lr, y_testing_lr = train_test_split(tp,fl, test_size=0.4)
clf_lr = LinearRegression()
clf_lr.fit(X_training_lr, y_training_lr)

yp_lr = clf_lr.predict(X_testing_lr)
print(clf_lr.score(X_testing_lr,y_testing_lr))

sys.stdout.close()