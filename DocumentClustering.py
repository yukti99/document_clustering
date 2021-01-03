from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from sklearn import manifold
import scipy
import nltk
import re
import string
import gensim
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 


# READING DOCUMENTS 

f1 = open("docs/F1.txt","r",encoding="utf8")
f2 = open("docs/F2.txt","r",encoding="utf8")
f3 = open("docs/F3.txt","r",encoding="utf8")
f4 = open("docs/F4.txt","r",encoding="utf8")
f5 = open("docs/F5.txt","r",encoding="utf8")
f6 = open("docs/F6.txt","r",encoding="utf8")
f7 = open("docs/F7.txt","r",encoding="utf8")
f8 = open("docs/F8.txt","r",encoding="utf8")
f9 = open("docs/F9.txt","r",encoding="utf8")
f10 = open("docs/F10.txt","r",encoding="utf8")

s1 = open("docs/S1.txt","r",encoding="utf8")
s2 = open("docs/S2.txt","r",encoding="utf8")
s3 = open("docs/S3.txt","r",encoding="utf8")
s4 = open("docs/S4.txt","r",encoding="utf8")
s5 = open("docs/S5.txt","r",encoding="utf8")
s6 = open("docs/S6.txt","r",encoding="utf8")
s7 = open("docs/S7.txt","r",encoding="utf8")
s8 = open("docs/S8.txt","r",encoding="utf8")
s9 = open("docs/S9.txt","r",encoding="utf8")
s10 = open("docs/S10.txt","r",encoding="utf8")


l1 = open("docs/L1.txt","r",encoding="utf8")
l2 = open("docs/L2.txt","r",encoding="utf8")
l3 = open("docs/L3.txt","r",encoding="utf8")
l4 = open("docs/L4.txt","r",encoding="utf8")
l5 = open("docs/L5.txt","r",encoding="utf8")
l6 = open("docs/L6.txt","r",encoding="utf8")
l7 = open("docs/L7.txt","r",encoding="utf8")
l8 = open("docs/L8.txt","r",encoding="utf8")
l9 = open("docs/L9.txt","r",encoding="utf8")
l10 = open("docs/L10.txt","r",encoding="utf8")


e1 = open("docs/E1.txt","r",encoding="utf8")
e2 = open("docs/E2.txt","r",encoding="utf8")
e3 = open("docs/E3.txt","r",encoding="utf8")
e4 = open("docs/E4.txt","r",encoding="utf8")
e5 = open("docs/E5.txt","r",encoding="utf8")
e6 = open("docs/E6.txt","r",encoding="utf8")
e7 = open("docs/E7.txt","r",encoding="utf8")
e8 = open("docs/E8.txt","r",encoding="utf8")
e9 = open("docs/E9.txt","r",encoding="utf8")
e10 = open("docs/E10.txt","r",encoding="utf8")

a1 = open("docs/A1.txt","r",encoding="utf8")
a2 = open("docs/A2.txt","r",encoding="utf8")
a3 = open("docs/A3.txt","r",encoding="utf8")
a4 = open("docs/A4.txt","r",encoding="utf8")
a5 = open("docs/A5.txt","r",encoding="utf8")
a6 = open("docs/A6.txt","r",encoding="utf8")
a7 = open("docs/A7.txt","r",encoding="utf8")
a8 = open("docs/A8.txt","r",encoding="utf8")
a9 = open("docs/A9.txt","r",encoding="utf8")
a10 = open("docs/A10.txt","r",encoding="utf8")

h1 = open("docs/H1.txt","r",encoding="utf8")
h2 = open("docs/H2.txt","r",encoding="utf8")
h3 = open("docs/H3.txt","r",encoding="utf8")
h4 = open("docs/H4.txt","r",encoding="utf8")
h5 = open("docs/H5.txt","r",encoding="utf8")
h6 = open("docs/H6.txt","r",encoding="utf8")
h7 = open("docs/H7.txt","r",encoding="utf8")
h8 = open("docs/H8.txt","r",encoding="utf8")
h9 = open("docs/H9.txt","r",encoding="utf8")
h10 = open("docs/H10.txt","r",encoding="utf8")

b1 = open("docs/B1.txt","r",encoding="utf8")
b2 = open("docs/B2.txt","r",encoding="utf8")
b3 = open("docs/B3.txt","r",encoding="utf8")
b4 = open("docs/B4.txt","r",encoding="utf8")

p1 = open("docs/P1.txt","r",encoding="utf8")
p2 = open("docs/P2.txt","r",encoding="utf8")
p3 = open("docs/P3.txt","r",encoding="utf8")

c1 = open("docs/C1.txt","r",encoding="utf8")






def preprocessing(s):
    s = re.sub(' +|\\n+', ' ', s)
    s = re.sub('<[^<]+?>', '', s)
    s = "".join([ch for ch in s if not ch.isdigit()])
    s = "".join([ch for ch in s if ch not in string.punctuation])   
    s  = s.lower()   
    return s

""" 
SIX CATEGORIES OF DOCUMENTS ARE :
        SPORTS
        ENVIRONMENT
        FINANCE
        ASTRONOMY & ASTROPHYSICS
        HEALTH
        SHAKESPEARE

"""

documents = [s1.read(),s2.read(),s3.read(),s4.read(),s5.read(),s6.read(),s7.read(),s8.read(),s9.read(),s10.read(),
f1.read(),f2.read(),f3.read(),f4.read(),f5.read(),f6.read(),f7.read(),f8.read(),f9.read(),f10.read(), 
a1.read(),a2.read(),a3.read(),a4.read(),a5.read(),a6.read(),a7.read(),a8.read(),a9.read(),a10.read(),
e1.read(),e2.read(),e3.read(),e4.read(),e5.read(),e6.read(),e7.read(),e8.read(),e9.read(),e10.read(),
h1.read(),h2.read(),h3.read(),h4.read(),h5.read(),h6.read(),h7.read(),h8.read(),h9.read(),h10.read(),
l1.read(),l2.read(),l3.read(),l4.read(),l5.read(),l6.read(),l7.read(),l8.read(),l9.read(),l10.read(),

]

real_labels = [
"S1","S2","S3","S4","S5","S6","S7","S8","S9","S10",
"F1","F2","F3","F4","F5","F6","F7","F8","F9","F10",
"A1","A2","A3","A4","A5","A6","A7","A8","A9","A10",
"E1","E2","E3","E4","E5","E6","E7","E8","E9","E10",
"H1","H2","H3","H4","H5","H6","H7","H8","H9","H10",
"L1","L2","L3","L4","L5","L6","L7","L8","L9","L10",

]

"""
documents = [
s1.read(),s2.read(),s3.read(),s4.read(),s5.read(),
f1.read(),f2.read(),f3.read(),f4.read(),f5.read(), 
a1.read(),a2.read(),a3.read(),a4.read(),a5.read(),
e1.read(),e2.read(),e3.read(),e4.read(),e5.read(),
h1.read(),h2.read(),h3.read(),h4.read(),h5.read(),
l1.read(),l2.read(),l3.read(),l4.read(),l5.read(),
b1.read(),b2.read(),b3.read(),b4.read(),

]

real_labels = [
"S1","S2","S3","S4","S5",
"F1","F2","F3","F4","F5",
"A1","A2","A3","A4","A5",
"E1","E2","E3","E4","E5",
"H1","H2","H3","H4","H5",
"L1","L2","L3","L4","L5",
"B1","B2","B3","B4"]
"""

vectorizer = TfidfVectorizer(stop_words='english',preprocessor=preprocessing)
X = vectorizer.fit_transform(documents)


def My_Kmeans_Algo(X):
    clusters_len = 6
    from cluster2 import KMeans   
    #print(len(vectorizer.get_feature_names()))
    # in this corpus there are 28 docs/sentences - > samples  and 4338 features
    k = KMeans(K=clusters_len, max_iters=250, plot_steps=False)
    y_pred = k.predict(X)
    print(y_pred)

    labels = y_pred
    print("LABELS = ",labels)

    print("\n\n-----------------------DOCUMENT CLUSTERS FORMED ARE------------------------")
    for i in range(clusters_len):
        indices = [index for index, element in enumerate(labels) if element == i]
        print("Documents of Cluster - ",i+1,": ")
        for j in indices:
            print(real_labels[j])
        print('--------------------------------------------------------------------------\n')
            
    labels_color_map = {
        0: 'blue', 1: 'green', 2: 'purple', 3: 'red', 4: 'yellow',
        5: 'orange', 6: 'pink', 7: 'cyan', 8: 'magenta', 9: 'black' 
    }
    pca_num_components = 2
    X2 = X.todense()
    reduced_data = PCA(n_components=pca_num_components).fit_transform(X2)
    # print reduced_data
    fig, ax = plt.subplots()
    xval=[]
    yval=[]
    for index, instance in enumerate(reduced_data):
        # print instance, index, labels[index]
        pca_comp_1, pca_comp_2 = reduced_data[index]
        color = labels_color_map[labels[index]]
        ax.scatter(pca_comp_1, pca_comp_2, c=color)
        xval.append(pca_comp_1)
        yval.append(pca_comp_2)

    for i, txt in enumerate(real_labels):
        ax.annotate(txt, (xval[i], yval[i]))

    plt.title("K-Means Clustering Graph by Yukti Khurana: ")
    plt.ylabel('Y-Position')
    plt.xlabel('X-Position')
    plt.show()

def Real_Kmeans_Algo(X):
    clusters_len = 6
    from sklearn.cluster import KMeans    
    model = KMeans(n_clusters=clusters_len, init='k-means++', max_iter=250, n_init=1)
    model.fit(X)
    labels = model.fit_predict(X)
    #print(labels)

    print("\nTop terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    for i in range(clusters_len):
        print("\nCluster %d:" % i),
        for index in order_centroids[i, :10]:
            print(' %s' % terms[index],end=" ")
        print("\n")

    print("\n\n\n")
    print("Prediction")

    Y = vectorizer.transform(["sports is good for health"])
    prediction = model.predict(Y)
    print(prediction)

    Y = vectorizer.transform(["environment is in danger, we must clean it"])
    prediction = model.predict(Y)
    print(prediction)

    Y = vectorizer.transform(["space is far but astronomy can be studied"])
    prediction = model.predict(Y)
    print(prediction)


    Y = vectorizer.transform(["exercising and heating good makes people healthy"])
    prediction = model.predict(Y)
    print(prediction)


    Y = vectorizer.transform(["i want reliable credit for finance"])
    prediction = model.predict(Y)
    print(prediction)

    labels_color_map = {
        0: 'blue', 1: 'green', 2: 'purple', 3: 'red', 4: 'yellow',
        5: 'orange', 6: 'pink', 7: 'cyan', 8: 'magenta', 9: 'black' 
    }
    pca_num_components = 2
    X2 = X.todense()
    reduced_data = PCA(n_components=pca_num_components).fit_transform(X2)
    figure, ax = plt.subplots(figsize=(16,16))
    xval=[]
    yval=[]
    for index, instance in enumerate(reduced_data):
        pca_comp_1, pca_comp_2 = reduced_data[index]
        color = labels_color_map[labels[index]]
        ax.scatter(pca_comp_1, pca_comp_2, c=color)
        xval.append(pca_comp_1)
        yval.append(pca_comp_2)

    for i, txt in enumerate(real_labels):
        ax.annotate(txt, (xval[i], yval[i]))

    plt.title("K-Means Clustering Graph by Yukti Khurana: ")
    plt.ylabel('Y-Position')
    plt.xlabel('X-Position')
    plt.show()




#Real_Kmeans_Algo(X)
My_Kmeans_Algo(X)



