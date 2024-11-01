from urllib.request import urlopen 
import json
from pprint import pprint
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder

def scrape_data(url):
    with urlopen(url) as response:
        data = response.read()
        return json.loads(data)

# sklearn pca
def pca_func(data):
    pca_data = PCA(n_components=2).fit_transform(data)
    return pca_data

# manual PCA
def custom_pca(data):
    data = np.array(data)
    # referred to https://medium.com/@nahmed3536/a-python-implementation-of-pca-with-numpy-1bbd3b21de2e
    # data -> (n,d) 
    # standardize the data
    standardized_data = (data - data.mean(axis = 0)) / data.std(axis = 0)
    # cov mat
    covariance_matrix = np.cov(standardized_data, ddof = 1, rowvar = False)
    # eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    order_of_importance = np.argsort(eigenvalues)[::-1] 
    sorted_eigenvalues = eigenvalues[order_of_importance]
    sorted_eigenvectors = eigenvectors[:,order_of_importance]
    explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    # number of principal components
    k = 2 
    reduced_data = np.matmul(standardized_data, sorted_eigenvectors[:,:k]) # transform the original data
    return reduced_data


# t-SNE sklearn
def tsne_func(X):
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(X)
    return X_embedded

# def data_encoder(data):
#     new_arr = []
#     encoder = OneHotEncoder()
#     enc_classes =  encoder.fit_transform(np.array(list(data)).reshape(-1, 1)).toarray()
#     new_arr = np.repeat(enc_classes, 10, axis=0)
#     print(new_arr.shape)
#     return new_arr

# LDA function
def lda_func(X, classes):
    # classes = data_encoder(classes)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    classes = np.repeat(colors, 10, axis=0)
    # print(classes.shape)
    # print(classes)
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_data = lda.fit_transform(X, classes)
    return lda_data
    # return 0


def parse_data(data):
    embeddings = []
    for i in data:
        # print(f"POS: {i}, Words: {data[i].keys()}, Number of words: {len(data[i])})")
        for k in data[i].keys():
            # print(f"\tWord: {k}, Embedding Shape: {len(data[i][k])}")
            embeddings.append(data[i][k])
        # print(data[i], i)
        # print(f"Word: {data[i]}, Embedding: {data_dict['nouns']['lowell'][i]}")
    pprint(np.array(embeddings).shape)
    return data.keys(), np.array(embeddings)

def plot_data(data, title:str, keys:list[str]):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    print(keys)
    plt.figure(figsize=(8, 6))
    for i in range(0, len(data), 10):
        segment = data[i:i+10]
        color = colors[i // 10 % len(colors)]
        plt.scatter(segment[:, 0], segment[:, 1], color=color, label=f'{keys[i // 10 % len(keys)]}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{title}.jpg'))


def main():

    qstr = "https://users.cs.utah.edu/~jeffp/teaching/HDDA/homeworks/glove_word_embeddings.json"
    data_dict = scrape_data(qstr)
    keys, parsed_data = parse_data(data_dict)

    # dim reduction using PCA
    test_data = pca_func(parsed_data)
    plot_data(test_data, "PCA", list(keys))

    test_data = custom_pca(parsed_data)
    plot_data(test_data, "Custom PCA", list(keys))


    # dim reduction using t-SNE
    test_data = tsne_func(parsed_data)
    plot_data(test_data, "t-SNE", list(keys))

    # dim reduction using LDA  
    test_data = lda_func(parsed_data, keys)
    plot_data(test_data, "LDA", list(keys))

if __name__ == "__main__":
    main()
