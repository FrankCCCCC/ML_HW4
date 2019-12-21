import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
# from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.externals import joblib

from mlxtend.plotting import plot_decision_regions

# nltk.download('stopwords')

path_data = ''

def read_data(path_data):
    raw_data = pd.read_csv(path_data + "Google_AI_published_research.csv")
    # print(raw_data.iloc[0, :].values)
    return raw_data

def select_samples(data, select_samples, is_shuffle):
    data = shuffle(data, random_state=0)
    if select_samples == -1:
        return data
    out = data[:select_samples]
    return out

def cleaner(data):
    # shape = data.shape
    # row = shape[0]
    # col = shape[1]
    # for r in range(row):
        # for c in range(col):
    cleaned_data = re.sub('[\W]', '', data)
    return cleaned_data

def clean_array(data):
    rows = len(data)
    for r in range(rows):
        data[r] = re.sub('[\W]', ' ', data[r])
    return data

def count_vectorize(data):
    count = CountVectorizer()
    # print(data.shape)
    count_vec = count.fit_transform(data).toarray()
    # print(count.vocabulary_)
    # print(count_vec)

    return count_vec, count

def tfidf(count_vec):
    tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
    tfidf_vec = tfidf.fit_transform(count_vec).toarray()
    # print(tfidf_vec)
    return tfidf_vec, tfidf

def tokenize_porter(docs):
    porter = PorterStemmer()
    # row0 = np.array([porter.stem(word) for word in docs[0].split()])
    token_matrix = np.array([])

    # print(token_matrix)
    for sentence in docs[:100]:
        # print(sentence)
        row = [porter.stem(word) for word in sentence.split()]
        row = np.array([" ".join(row)])
        # print(row)
        token_matrix = np.append(token_matrix, row, axis=0)

    print(token_matrix)
    return token_matrix

def tokenizer(sentence):
    porter = PorterStemmer()
    return [porter.stem(word) for word in sentence.split()]

def show_explained_var(decomposer, n_comp):
    explained_var = np.sort(np.array(decomposer.explained_variance_ratio_))[::-1]
    plt.bar(range(1, n_comp + 1), explained_var)
    plt.step(range(1, n_comp + 1), np.cumsum(explained_var))
    plt.show()

    return explained_var

def show_lda_topic(count, lda):
    n_top_words =5
    feature_names = count.get_feature_names()
    # print(np.array(feature_names).shape)
    # print(feature_names)
    # print(lda.components_.shape)
    # print(lda.components_)
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic %d" % (topic_idx + 1))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

def lda_clf_fit_predict(data, pipe_lda_clf):
    processed_data = pipe_lda_clf.fit_transform(data)
    classify_result = np.zeros(processed_data.shape[0])
    i = 0
    for row in processed_data:
        classify_result[i] = np.argsort(row)[-1:][0]
        i = i + 1
    return processed_data, classify_result

def silhouette_plot(x, y):
    cluster_labels = np.unique(y)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(x, y, metric='euclidean')
    y_ax_lower = 0
    y_ax_upper = 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_lower) / 2.)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    print("Silhouette: ", silhouette_avg)
    plt.axvline(silhouette_avg, color='red', linestyle='--')

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('silhouette coefficient')

    plt.tight_layout()
    plt.show()

def elbow_plot(x):
    range_array = range(1, 100, 10)
    distortions = []
    for i in range_array:
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
        ac = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='complete')
        model = km
        y = model.fit(x)
        distortions.append(model.inertia_)
    plt.plot(range_array, distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout()
    plt.show()

def tune_lda_pre(data, next_func=None, is_plot_show=1):
    print("LDA Tuning...")
    res = [('-1', -1)]
    best_res = [('-1', -1)]
    best_y = -1
    flag = False
    if is_plot_show == 2:
        flag = True
    for max_dff in [.1]:
        for max_featuresf in [5000]:
            for n_topicsf in [50, 60, 70]:
                count = CountVectorizer(stop_words='english', max_df=max_dff, max_features=max_featuresf)
                lda = LatentDirichletAllocation(n_topics = n_topicsf, random_state = 0)
                pipe = Pipeline([('count', count), ('lda', lda)])
                y = pipe.fit_transform(data)
                res = next_func(y, [('max_df', max_dff), ('max_features', max_featuresf), ('n_topics', n_topicsf)], flag)
                if res[0][1] > best_res[0][1]:
                    best_res = res
                    best_y = y
    print("Pipe Tuning Result: ")
    for e in best_res:
        if e[0] == 'best y':
            if is_plot_show >= 1:
                silhouette_plot(best_y, e[1])
        else:
            print(e[0], ': ', e[1])

def tune_tfidf_svd_pre(data, next_func=None, is_plot_show=1):
    print("TFIDF SVD Tuning...")
    res = [('-1', -1)]
    best_res = [('-1', -1)]
    best_y = -1
    flag = False
    if is_plot_show == 2:
        flag = True
    for normf in ['l1']:
        for ngram_rangef in [(1, 1)]:
            for n_componentsf in [50, 60, 70, 80]:
                for n_iterf in [15]:
                    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, ngram_range=ngram_rangef, norm='l2')
                    svd = TruncatedSVD(n_components = n_componentsf, n_iter = 5, random_state = 0)
                    pipe = Pipeline([('tfidf', tfidf), ('svd', svd)])
                    y = pipe.fit_transform(data)
                    res = next_func(y, [('norm', normf), ('ngram_range', ngram_rangef), ('n_components', n_componentsf), ('n_iter', n_iterf)], flag)
                    if res[0][1] > best_res[0][1]:
                        best_res = res
                        best_y = y
    print("Pipe Tuning Result: ")
    for e in best_res:
        if e[0] == 'best y':
            if is_plot_show >= 1:
                silhouette_plot(best_y, e[1])
        else:
            print(e[0], ': ', e[1])

def tune_dbscan(data, last_step, is_plot_show=False):
    best_last_step = [(-1, -1)]
    best_score = -1
    best_eps = -1
    best_min_samples = -1
    best_clusters = -1
    best_y = -1
    print("Tuning...")
    for epsf in [0.2, 0.3, 0.5, 0.8, 1]:
        for min_samplesf in [3, 4, 5, 6, 8, 10, 15, 20]:
            dbscan = DBSCAN(eps=epsf, min_samples=min_samplesf, metric='euclidean')
            y = dbscan.fit_predict(data)
            score = silhouette_score(data, y)
            if score > best_score:
                best_last_step = last_step
                best_score = score
                best_eps = epsf
                best_min_samples = min_samplesf
                best_y = y
                best_clusters = np.unique(y).shape[0]
    print("DBSCAN Tune Result:")
    print("Best silhouette_score: ", best_score)
    for param in best_last_step:
        print(param[0], ': ', param[1])
    print("Best eps: ", best_eps)
    print("Best min_samples: ", best_min_samples)
    print("Cluster Result: ", best_clusters)
    if is_plot_show:
        silhouette_plot(data, best_y)
    return [('best score', best_score)] + best_last_step + [('best eps', best_eps), ('best min_samples', best_min_samples), ('best clusters', best_clusters), ('best y', best_y)]

def tune_ac(data, last_step, is_plot_show=False):
    best_last_step = [(-1, -1)]
    best_score = -1
    best_n_cluster = -1
    best_affinity = -1
    best_linkage = -1
    best_clusters = -1
    best_y = -1
    print("AgglomerativeClustering Tuning...")
    for n_clusterf in [20, 50, 60, 70, 80, 100, 200]:
        for affinityf in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']:
            for linkagef in ['complete', 'average', 'single']:
                ac = AgglomerativeClustering(n_clusters=n_clusterf, affinity=affinityf, linkage=linkagef)
                y = ac.fit_predict(data)
                score = silhouette_score(data, y)
                if score > best_score:
                    best_last_step = last_step
                    best_score = score
                    best_n_cluster = n_clusterf
                    best_affinity = affinityf
                    best_linkage = linkagef
                    best_y = y
                    best_clusters = np.unique(y).shape[0]
    print("AgglomerativeClustering Tune Result:")
    print("Best silhouette_score: ", best_score)
    for param in best_last_step:
        print(param[0], ': ', param[1])
    print("Best n_cluster: ", best_n_cluster)
    print("Best affinity: ", best_affinity)
    print("Best linkage: ", best_linkage)
    print("Cluster Result: ", best_clusters)
    if is_plot_show:
        silhouette_plot(data, best_y)
    return [('best score', best_score)] + best_last_step + [('best n_cluster', best_n_cluster), ('best affinity', best_affinity), ('best linkage', best_linkage), ('best clusters', best_clusters), ('best y', best_y)]

def tune_kmp(data, last_step, is_plot_show=False):
    best_last_step = [(-1, -1)]
    best_score = -1
    best_n_cluster = -1
    best_n_init = -1
    best_max_iter = -1
    best_tol = -1
    best_y = -1
    print("K-Means++ Tuning...")
    for n_clusterf in [20, 50, 60, 70, 80, 100, 200]:
        for n_initf in [10, 20]:
            for max_iterf in [300, 500, 800]:
                for tolf in [1e-3, 1e-04, 1e-5]:
                    kmeans_plus = KMeans(n_clusters=n_clusterf, init='k-means++', n_init=n_initf, max_iter=max_iterf, tol=tolf, random_state=0)
                    y = kmeans_plus.fit_predict(data)
                    score = silhouette_score(data, y)
                    if score > best_score:
                        best_last_step = last_step
                        best_score = score
                        best_n_cluster = n_clusterf
                        best_n_init = n_initf
                        best_max_iter = max_iterf
                        best_tol = tolf
                        best_y = y
                        best_clusters = np.unique(y).shape[0]
    print("AgglomerativeClustering Tune Result:")
    print("Best silhouette_score: ", best_score)
    for param in best_last_step:
        print(param[0], ': ', param[1])
    print("Best n_cluster: ", best_n_cluster)
    print("Best n_init: ", best_n_init)
    print("Best max_iter: ", best_max_iter)
    print("Cluster Result: ", best_clusters)
    if is_plot_show:
        silhouette_plot(data, best_y)
    return [('best score', best_score)] + best_last_step + [('best n_cluster', best_n_cluster), ('best n_init', best_n_init), ('best max_iter', best_max_iter), ('best tol', best_tol), ('best clusters', best_clusters), ('best y', best_y)]


# t = "eoigrhsaivs ss\n\r()ddd()ad \n oqhwo\r"
# print(t)
# text = cleaner(t)
# print(text)

# Global Parameters
n_comp = 50
pca_comp = 150
selected_samples = 1000
n_cluster = 20

# Read Data
raw_data = read_data(path_data)
data = raw_data.values.ravel()
data = select_samples(data, selected_samples, True)
data = clean_array(data)

# Preprocessing
# data = tokenize_porter(data)
# count_vec, count = count_vectorize(data)
# tfidf_vec, tfidf = tfidf(count_vec)
tfidf_vectorizer = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, norm='l2')
count = CountVectorizer(stop_words='english', max_df=.1, max_features=5000)

pca = PCA(n_components = pca_comp)
svd = TruncatedSVD(n_components = n_comp, n_iter = 5, random_state = 0)
lda = LatentDirichletAllocation(n_topics = n_comp, random_state = 0)

standard_scaler = StandardScaler()

# CLuster Algo
lr_clf = LogisticRegression(random_state=0)
kmeans = KMeans(n_clusters=n_cluster, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
kmeans_plus = KMeans(n_clusters=n_cluster, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
ac = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean', linkage='complete')
dbscan = DBSCAN(eps = 0.5, min_samples=10, metric='euclidean')
# optics = OPTICS(min_samples=2)
spc = SpectralClustering(n_clusters=n_cluster, assign_labels='discretize', random_state=0)
lda_clf = LatentDirichletAllocation(n_topics=n_cluster, random_state=0)

model = ac

# Pipeline
lda_pipe_array = [('count', count), ('lda', lda)]
pre_pipe_array = [('tfidf', tfidf_vectorizer), ('svd', svd)]
pipe_lda = Pipeline(lda_pipe_array)
pipe_pre = Pipeline(pre_pipe_array)

pipe_km_clf = Pipeline(lda_pipe_array + [('km', kmeans)])
pipe_kmp_clf = Pipeline(lda_pipe_array + [('kmp', kmeans_plus)])
pipe_ac_clf = Pipeline(lda_pipe_array + [('ac', ac)])
pipe_dbscan_clf = Pipeline(lda_pipe_array + [('dbscan', dbscan)])
# pipe_optics_clf = Pipeline(lda_pipe_array + [('optics', optics)])
pipe_spc_clf = Pipeline(lda_pipe_array + [('spc', spc)])
pipe_lda_clf = Pipeline([('count', count), (('ldaClf', lda_clf))])
processed_data = pipe_lda.fit_transform(data)
# print(processed_data.shape)
# print(processed_data[0, :])
# print(processed_data[0, :].shape)
# processed_data = pipe_lda.fit_transform(data)
# print(lda_data.shape)
# print(lda_data)
# show_lda_topic(count, lda)


# y = pipe_kmp_clf.fit_predict(data)
# y = pipe_dbscan_clf.fit_predict(data)
# show_explained_var(pca, pca_comp)
# processed_data, y = lda_clf_fit_predict(data, pipe_lda_clf)
# y = dbscan.fit_predict(processed_data)
# y = optics.fit_predict(processed_data)
# y = spc.fit_predict(processed_data)
# silhouette_plot(processed_data, y)
# elbow_plot(processed_data)

# tune_dbscan(data, tune_lda_pre, False)
tune_lda_pre(data, tune_kmp)
# tune_tfidf_svd_pre(data, tune_ac)

# param_grid_dbscan = [
#             {'count__max_df': [.1],
#             'count__max_features': [5000]
#             },
#             {'lda__n_topics': [20, 50, 100, 500, 1000, 5000]},
#             {'dbscan__eps': [0.1, 0.2, 0.5, 1, 1.5, 2],
#             'dbscan__min_samples': [2, 3, 5, 10, 15, 20]
#             }
#             ]

# gs = GridSearchCV(pipe_dbscan_clf, param_grid_dbscan, scoring='normalized_mutual_info_score', cv=5, verbose=1, n_jobs=-1)
# gs.fit(data)
# print('Best param: ', gs.best_params_)
# print('Best score: ', gs.best_score_)