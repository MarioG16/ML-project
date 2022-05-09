import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


pj = pd.read_csv('../data/langosteria.csv', sep= ';')
pj_2 = pj.dropna(axis = 1 , how = 'any' , thresh = None , subset = None , inplace = False)
print(pj_2)

#Pre_Processing
#Normalization min/max
def minmax_norm(pj_2_input):
    return (pj_2 - pj_2.min()) / ( pj_2.max() - pj_2.min())
pj_2_minmax_norm = minmax_norm(pj_2)
print(pj_2_minmax_norm)

#Correlation between features for clustering
plt.rcParams["figure.figsize"]=[10,6]
sns.heatmap(pj_2_minmax_norm.corr(), cmap='jet', annot=True, fmt=".1f", annot_kws={"size":12})
plt.title("Correlation of Features", size=14, c='b')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

pj_3 = pj_2_minmax_norm
pj_3 = pj_2_minmax_norm.loc[:, ["Visits", "No_Shows", "Spend_Visit"]]
print(pj_3)
print(pj_3.shape)

#Elbow_method (capire il miglior numero di K) ci dice se ci sono dati differenti e se questi il numero di cluster è giusto
sum_square_distance = []
K = range(2, 10)
for k in K:
     kmeans = KMeans(n_clusters=k)
     kmeans.fit(pj_3)
     sum_square_distance.append(kmeans.inertia_)
plt.plot(K, sum_square_distance, 'o--b', markersize=10)
plt.xlabel('Values of K')
plt.ylabel('Sum of squared distance for Opt K')
plt.show()

#Scelta di K mediante silhouette score
for k in K:
         kmeans = KMeans(n_clusters=k)
         c_s = kmeans.fit_predict(pj_3)
         sih_avg = silhouette_score(pj_3, c_s)
         print(f'cluster {k} ---- sih score = {sih_avg}')
         vis = SilhouetteVisualizer(kmeans, colors='yellowbrick')
         vis.fit(pj_3)
         vis.show()


#Clustering con k=5
kmeans = KMeans(n_clusters=5)
pj_3["Cluster"] = kmeans.fit_predict(pj_3)
pj_3["Cluster"] = pj_3["Cluster"].astype("category")
sns.relplot(x="Visits", y="Spend_Visit",hue="Cluster", data=pj_3, height=8)
plt.show()

#Centri cluster
km_cluster=KMeans(n_clusters=5)
(km_cluster.fit(pj_3))
print(km_cluster.cluster_centers_)

#Grafico a dispersione con centri dei cluster indicati
plt.scatter(pj_3.values[:,0], pj_3.values[:,2], c=km_cluster.labels_, cmap='rainbow')
plt.scatter(km_cluster.cluster_centers_[:,0], km_cluster.cluster_centers_[:,2], c='k', s=100)
plt.title("Visits vs Spend_Visit", c='b', size=16)
plt.xlabel("Visits", size=14)
plt.ylabel("Spend_Visit", size=14)
plt.legend()
plt.show()


#Stampo i cluster in base ad output voluto
pj_final = pj.copy()
pj_final["label"]=km_cluster.labels_
print(pj_final)
pj_final_2 = pj_final.loc[:, ["ID Customers", "label"]]
print(pj_final_2)


#Agglomerative_clustering + dendrogram
hierarchy=AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_pred_hierarchy=hierarchy.fit_predict(pj_final.drop(["Gender"], axis=1))
print(y_pred_hierarchy)
print("############")
print(hierarchy.labels_)

dendro=linkage(pj_final.drop(["Gender"], axis=1),'ward')
annots=pj_final.drop(["Gender"], axis=1).index
plt.rcParams['figure.figsize']=[12,10]
dendrogram(dendro, orientation='top', labels=annots, distance_sort='descending', show_leaf_counts=True)
plt.show()

#Grafico a dispersione [Cluster vs No_shows]
pj_3["No_Shows"] = pj_3["No_Shows"]
sns.catplot(x="No_Shows", y="Cluster", data=pj_3, kind="boxen", height=10)
plt.show()


#Classi di customers suddivisi in L,Ul,SL
Cluster_UL = pj_final_2.loc[(pj_final_2['label'] == 1)  | (pj_final_2['label'] == 3) ]
Cluster_L = pj_final_2.loc[(pj_final_2['label'] == 0) | (pj_final_2['label'] == 2) ]
Cluster_SL = pj_final_2.loc[pj_final_2['label'] == 4]
print('Clienti Super_Loyal')
print(Cluster_SL)
print('Clienti Loyal')
print(Cluster_L)
print('Clienti Un_Loyal')
print(Cluster_UL)

#Clustering con GMM per confronto tra modelli
pj_4 = pj_2_minmax_norm.loc[:,['Spend_Visit', 'Visits']]
gmm = GaussianMixture (n_components= 30)
gmm.fit(pj_4)

#Predicitions
labels = gmm.predict(pj_4)
frame = pd.DataFrame(pj_4)
frame['cluster'] = labels
frame.columns = ['Spend_Visit', 'Visits', 'cluster']

color=['blue','green','red', 'black', 'yellow']
for k in range(0,5):
    pj_4 = frame[frame["cluster"]==k]
    plt.scatter(pj_4["Visits"],pj_4["Spend_Visit"],c=color[k])
plt.title("Visits vs Spend_Visit", c='b', size=16)
plt.xlabel("Visits", size=14)
plt.ylabel("Spend_Visit", size=14)
plt.show()

#Analisi valori BIC,AIC vs silhouette coeff.
pj_4 = pj_3.loc[:,['Spend_Visit', 'Visits']]
def get_km(k, pj_4):
    km = KMeans(n_clusters=k, random_state=37)
    km.fit(pj_4)
    return(km)

def get_bic_aic(k, pj_4):
    gmm = GaussianMixture(n_components=k, init_params='kmeans')
    gmm.fit(pj_4)
    return(gmm.bic(pj_4), gmm.aic(pj_4))
Y = 998 #samples

def get_score(k, pj_4, Y):
    km = get_km(k, pj_4)
    y_pred = km.predict(pj_4)
    bic, aic = get_bic_aic(k, pj_4)
    sil = silhouette_score(pj_4, y_pred)
    return k, bic, aic, sil

df = pd.DataFrame([get_score(k, pj_4, Y) for k in range(2, 11)], columns=['k', 'BIC', 'AIC', 'silhouette'])
print(df)

plt.style.use('ggplot')

def plot_compare(df, y1, y2, x, fig, ax1):
    ax1.plot(df[x], df[y1], color='tab:red')
    ax1.set_title(f'{y1} and {y2}')
    ax1.set_xlabel(x)
    ax1.set_ylabel(y1, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.plot(df[x], df[y2], color='tab:blue')
    ax2.set_ylabel(y2, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

def plot_contrast(df, y1, y2, x, fig, ax):
    a = np.array(df[y1])
    b = np.array(df[y2])

    r_min, r_max = df[y1].min(), df[y1].max()
    scaler = MinMaxScaler(feature_range=(r_min, r_max))
    b = scaler.fit_transform(b.reshape(-1, 1))[:,0]

    diff = np.abs(a - b)
    ax.plot(df[x], diff)
    ax.set_title('Scaled Absolute Difference')
    ax.set_xlabel(x)
    ax.set_ylabel('absolute difference')

def plot_result(df, y1,  y2, x):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    plot_compare(df, y1, y2, x, fig, axes[0])
    plot_contrast(df, y1, y2, x, fig, axes[1])
    plt.tight_layout()
plot_result(df,'BIC','silhouette' , 'k')
plt.show()