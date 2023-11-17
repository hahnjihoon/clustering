import os
import sys
import warnings

from konlpy.tag import Okt

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG,  # 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def clustering(adres_list, filna_list):
    # 데이터불러옴
    # data = pd.read_excel('C:/Users/user/Desktop/222.xlsx', header=0, sheet_name=' 베피스+기저귀')
    data = pd.read_excel(adres_list, header=0, sheet_name=filna_list)

    # 1.기존 요약된 배열대로
    summary_data = pd.DataFrame(data["No."])
    summary_data = summary_data.assign(제목=data["제목"])
    summary_data = summary_data.assign(요약내용=data["요약 내용"])
    summary_data['공백1'] = ''
    summary_data['공백2'] = ''
    print("summary_data ::", summary_data)
    # 2.원본에 열만추가
    # data['공백1'] = ''
    # data['공백2'] = ''

    # 요약내용을 단어별로 추출
    okt = Okt()
    okt_data = okt.nouns(str(summary_data['요약내용']))
    unique_list = list(set(okt_data))
    print("okt_data :: ", unique_list)
    # okt = Okt()
    # okt_data = okt.nouns(str(data['요약 내용']))
    # unique_list = list(set(okt_data))
    # print("okt_data :: ", unique_list)

    # 요약내용컬럼 벡터화
    vectorizer = TfidfVectorizer()  # 벡터 초기화
    X = vectorizer.fit_transform(data['요약 내용'])
    print("X :::", X)

    # 단어 파싱 리스트
    word_list = vectorizer.get_feature_names() #벡터를 단어로바꿈
    # word_list = vectorizer.get_feature_names_out() #3.10.x 부터 out추가
    print('word_list :: ', word_list)

    # 군집화
    kmeans = KMeans(n_clusters=5, random_state=42) #default42
    kmeans.fit(X)

    # 행별 군집플래그 부여
    cluster_labels = kmeans.labels_
    summary_data['cluster_label'] = cluster_labels  # 'Cluster' 컬럼에 군집 레이블 추가
    # data['cluster_label'] = cluster_labels

    # 행별 단어 모음 추가
    parsing_row_words_list = []
    row_count = len(summary_data['요약내용'])
    # row_count = len(data['요약 내용'])

    for i in range(row_count):
        # row_bactor = TfidfVectorizer() #벡터화모듈초기화
        # print('rowbactor ::: ', row_bactor)
        # X = row_bactor.fit_transform([data['요약 내용'][i]]) #1행문장 벡터화
        # print('xxxxxxxx :: ', X)
        # parsing_row_word = row_bactor.get_feature_names_out() #벡터에서 글자가져옴
        # print('parsing_row_word ::: ', parsing_row_word)
        # parsing_row_words_list.append(parsing_row_word) #배열에 리스트추가
        # print('parsing_row_words_list ::: ', parsing_row_words_list)

        nouns_row = okt.nouns(str(summary_data['요약내용'][i]))
        parsing_row_words_list.append(nouns_row)
        print('parsing_row_words_list::::: ', parsing_row_words_list)
        # nouns_row = okt.nouns(str(data['요약 내용'][i]))
        # parsing_row_words_list.append(nouns_row)
        # print('parsing_row_words_list::::: ', parsing_row_words_list)

    summary_data['tokens'] = parsing_row_words_list

    # 결과를 엑셀 파일로 저장
    output_file = os.path.splitext(adres_list)[0] + '_clustered.xlsx'
    summary_data.to_excel(output_file, index=False, sheet_name=filna_list)


def clusteringOver(adres_list, filna_list):
    for adres in adres_list :
        i=0
        # originExcel = pd.read_excel(adres, header=0, sheet_name=filna_list)
        output_file = os.path.splitext(adres)[i] + '_clustered.xlsx'
        print(f'for{i}:adres :: ',adres)
        # with pd.ExcelWriter('C:/Users/Rainbow Brain/Desktop/clustered_data.xlsx', engine='openpyxl', mode='a') as writer:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for filna in filna_list:
                print(f'for{i}:filna :: ', filna)
                data = pd.read_excel(adres, header=0, sheet_name=filna)

                # 요약내용컬럼 벡터화
                vectorizer = TfidfVectorizer()  # TfidfVectorizer 초기화
                X = vectorizer.fit_transform(data['요약 내용'])
                # print('요약내용 : ', X, flush=True)
                # logger.info('요약내용 : %s', X)

                # 단어 파싱 리스트
                word_list = vectorizer.get_feature_names()
                print('전체단어수 : ', len(word_list), flush=True)
                logger.info('전체단어수 : %d', len(word_list))
                print(word_list, flush=True)
                logger.info('%s', word_list)

                # 군집화
                kmeans = KMeans(n_clusters=5, random_state=42)
                kmeans.fit(X)

                # 행별 군집플래그 부여
                cluster_labels = kmeans.labels_
                data['cluster_label'] = cluster_labels  # 'Cluster' 컬럼에 군집 레이블 추가
                # print(data[['요약 내용', 'cluster_flag']], flush=True)

                # 행별 단어 모음 추가
                parsing_row_words_list = []
                row_count = len(data['요약 내용'])
                summary_data = pd.DataFrame(data["요약 내용"])
                okt = Okt()
                okt_data = okt.nouns(str(summary_data['요약 내용']))
                unique_list = list(set(okt_data))
                print("unique_list :: ", unique_list)
                for i in range(row_count):
                    nouns_row = okt.nouns(str(summary_data['요약 내용'][i]))
                    parsing_row_words_list.append(nouns_row)
                    # row_bactor = TfidfVectorizer()
                    # X = row_bactor.fit_transform([data['요약 내용'][i]])
                    # parsing_row_word = row_bactor.get_feature_names_out()
                    # parsing_row_words_list.append(parsing_row_word)

                data['tokens'] = parsing_row_words_list

                # 결과를 엑셀 파일로 저장
                # data.to_excel('C:/Users/Rainbow Brain/Desktop/clustered_data.xlsx', index=False, sheet_name=filna)
                data.to_excel(writer, index=False, sheet_name=f"{filna}")
                logger.info('결과 저장 완료: %s', f"{filna}")

if __name__ == "__main__":
    # adres = sys.argv[1]
    # filna = sys.argv[2]
    #
    # # clustering 함수 호출
    # clustering(adres, filna)
    # if len(sys.argv) < 4 or sys.argv.index('--') >= len(sys.argv):
    #     print("Usage: python cluster.py <address1> <address2> ... -- <keyword1> <keyword2> ...")
    #     sys.exit(1)
    if len(sys.argv) == 3:
        adres_list = sys.argv[1]
        filna_list = sys.argv[2]
        print('adres_list ::', adres_list)
        print('filna_list ::::', filna_list)
        clustering(adres_list, filna_list)
    elif len(sys.argv) > 4:
        keyword_index = sys.argv.index('--keyword')
        adres_list = sys.argv[1:keyword_index]
        filna_list = sys.argv[keyword_index + 1:]
        print('adres_list ::::', adres_list)
        print('filna_list ::::', filna_list)
        clusteringOver(adres_list, filna_list)