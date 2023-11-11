import os
from os.path import join
import copy
import warnings

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import sklearn
import matplotlib.pyplot as plt

# 데이터불러옴
data = pd.read_excel('C:/Users/user/Desktop/222.xlsx', header=0, sheet_name=' 베피스+기저귀')


# 요약내용컬럼 단어별 파싱, 벡터화
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['요약 내용'])
# print('요약내용 : ', X)


# 단어 리스트
word_list = vectorizer.get_feature_names()
print('전체단어수 : ', len(word_list))
print(word_list)


# 군집화
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
# cluster_labels = kmeans.fit_predict(X)

# # t-SNE를 사용하여 2차원으로 축소
# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(X)
#
# # 군집화 결과와 t-SNE 결과를 합침
# result_df = pd.DataFrame(X_tsne, columns=['X', 'Y'])
# result_df['Cluster'] = cluster_labels
#
# # 산점도로 시각화
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(result_df['X'], result_df['Y'], c=result_df['Cluster'], cmap='viridis', s=50)
# plt.legend(*scatter.legend_elements(), title='Clusters')
# plt.title('t-SNE Visualization of Clusters')
# plt.show()


# 행별 군집플래그 부여
cluster_labels = kmeans.labels_
data['Cluster'] = cluster_labels  # 'Cluster' 컬럼에 군집 레이블 추가
print(data[['요약 내용', 'Cluster']])


# 결과를 엑셀 파일로 저장
data.to_excel('C:/Users/user/Desktop/clustered_data.xlsx', index=False)

# column_count = data.shape[1]
# for
# first_row_words = data['요약 내용'][0]
# first_row_vector = vectorizer.transform([first_row_words])
# cluster_label = kmeans.predict(first_row_vector)[0] + 1
# print('1번째행 클러스터라벨 :: ',cluster_label)

# column_count = data.shape[1]
# print('컬럼갯수 : ',column_count)
#
# summary_text = data['요약 내용'][0]
#
# # 단어별로 파싱하여 리스트로 만듦
# word_list = summary_text.split()
# print('단어별 파싱 : ', word_list)
#
# print(data)