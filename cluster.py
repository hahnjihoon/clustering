import os
import sys
import warnings
import pandas as pd
import logging

from konlpy.tag import Okt
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.DEBUG,  # 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def clustering(adres_list, filna_list):
    # utf-8, ascii 확인
    # encode = chardet.detect(adres_list)
    # encode2 = chardet.detect(filna_list)

    # 데이터불러옴
    data = pd.read_excel(adres_list, header=0, sheet_name=filna_list)
    # data['요약 내용'] = data['요약 내용'].fillna('')

    # 추가 날짜컬럼
    data["날짜 (yyyymmdd)"] = pd.DataFrame(data["날짜 (yyyymmdd)"])
    # if not pd.DataFrame(data["날짜 (yyyymmdd)"]).empty and len(str(data["날짜 (yyyymmdd)"][0])) > 8:
    #     data["날짜 (yyyymmdd)"] = pd.to_datetime(data["날짜 (yyyymmdd)"])
    #     data["날짜 (yyyymmdd)"] = data["날짜 (yyyymmdd)"].dt.strftime('%Y%m%d')
    #
    #     data["게시일 (post_date)"] = pd.DataFrame(data["게시일 (post_date)"])
    #     data["게시일 (post_date)"] = pd.to_datetime(data["게시일 (post_date)"])
    #     data["게시일 (post_date)"] = data["게시일 (post_date)"].dt.strftime('%Y%m%d')
    # else:
    #     print("데이터가 하나도 없다dddddd") # 결과를 엑셀 파일로 저장
    #     output_file = os.path.splitext(adres_list)[0] + '_clustered.xlsx'
    #     data.to_excel(output_file, index=False, sheet_name=filna_list)
    #     exit()
    data["날짜 (yyyymmdd)"] = pd.DataFrame(data["날짜 (yyyymmdd)"])
    if not pd.DataFrame(data["날짜 (yyyymmdd)"]).empty and len(str(data["날짜 (yyyymmdd)"][0])) > 8:
        data["날짜 (yyyymmdd)"] = pd.to_datetime(data["날짜 (yyyymmdd)"])
        data["날짜 (yyyymmdd)"] = data["날짜 (yyyymmdd)"].dt.strftime('%Y%m%d')

        data["게시일 (post_date)"] = pd.DataFrame(data["게시일 (post_date)"])
        data["게시일 (post_date)"] = pd.to_datetime(data["게시일 (post_date)"])
        data["게시일 (post_date)"] = data["게시일 (post_date)"].dt.strftime('%Y%m%d')
        exit()

    if pd.DataFrame(data["날짜 (yyyymmdd)"]).empty:
        print("데이터가 하나도 없다(키워드1개)")
        # 결과를 엑셀 파일로 저장
        output_file = os.path.splitext(adres_list)[0] + '_clustered.xlsx'
        data.to_excel(output_file, index=False, sheet_name=filna_list)
        exit()

    # token용 배열생성
    summary_data = pd.DataFrame(data["제목"])
    summary_data = summary_data.assign(요약내용=data["요약 내용"])

    #빈칸 공백처리
    data['요약 내용'].fillna('', inplace=True)
    print("요약내용 있는지없는지 :: ", data['요약 내용'])

    # 요약 내용이 5개 미만인 경우
    if len(data['요약 내용']) < 5:
        print("데이터가 적어서 군집이 형성이 안돼요")
        data['cluster_label'] = ''
        data['tokens'] = ''
        output_file = os.path.splitext(adres_list)[0] + '_clustered.xlsx'
        data.to_excel(output_file, index=False, sheet_name=filna_list)
        exit()

    if not any(data['요약 내용'].apply(lambda x: bool(x.strip()))):
        #요약내용에 .apply(함수)를 적용시키는데 어떤함수냐 lambda x : x를 .strip시작끝공백제거해서 그결과가 빈문자면 flase 있으면true
        print("요약내용이 없어서 제목으로 만들어요")
        vectorizer = TfidfVectorizer()  # 벡터 초기화
        X = vectorizer.fit_transform(data['제목'])

        # 단어 파싱 리스트
        word_list = vectorizer.get_feature_names_out()  # 벡터를 단어로바꿈
        # print("word_list ::: ", word_list.astype(str).tolist())

        # 군집화
        kmeans = KMeans(n_clusters=5, random_state=42)  # default42
        kmeans.fit(X)

        # 행별 군집플래그 부여
        cluster_labels = kmeans.labels_
        data['cluster_label'] = cluster_labels

        # 행별 단어 모음 추가
        parsing_row_words_list = []
        row_count = len(data['제목'])

        okt = Okt()
        for i in range(row_count):
            nouns_row = okt.nouns(str(data['제목'][i]))
            if not nouns_row:
                nouns_row = word_tokenize(data['제목'][i])
            parsing_row_words_list.append(nouns_row)

        data['tokens'] = parsing_row_words_list
        # print('parsing_row_words_list::::: ', parsing_row_words_list)

        # 결과를 엑셀 파일로 저장
        output_file = os.path.splitext(adres_list)[0] + '_clustered.xlsx'
        data.to_excel(output_file, index=False, sheet_name=filna_list)
        exit()

    # 요약내용컬럼 벡터화
    vectorizer = TfidfVectorizer()  # 벡터 초기화
    X = vectorizer.fit_transform(data['요약 내용'])

    # 단어 파싱 리스트
    word_list = vectorizer.get_feature_names_out()  # 벡터를 단어로바꿈
    # print("word_list ::: ", word_list.astype(str).tolist())
        
    # 군집화
    kmeans = KMeans(n_clusters=10, random_state=42) #default42
    kmeans.fit(X)

    # 행별 군집플래그 부여
    cluster_labels = kmeans.labels_
    data['cluster_label'] = cluster_labels

    # 행별 단어 모음 추가
    parsing_row_words_list = []
    row_count = len(summary_data['요약내용'])

    okt = Okt()
    for i in range(row_count):
        nouns_row = okt.nouns(str(summary_data['요약내용'][i]))
        if not nouns_row:
            nouns_row = word_tokenize(summary_data['요약내용'][i])
        parsing_row_words_list.append(nouns_row)

    data['tokens'] = parsing_row_words_list
    # print('parsing_row_words_list::::: ', parsing_row_words_list)

    # 결과를 엑셀 파일로 저장
    output_file = os.path.splitext(adres_list)[0] + '_clustered.xlsx'
    data.to_excel(output_file, index=False, sheet_name=filna_list)

def clustering_lot(adres_list, filna_list):
    try:
        for adres in adres_list:
            i = 0
            output_file = os.path.splitext(adres)[i] + '_clustered.xlsx'
            print(f'파일명 {i} :: ', adres)
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                for filna in filna_list:
                    data = pd.read_excel(adres, header=0, sheet_name=filna)
                    print(f'시트명 {filna} :: ', adres)
                    # 빈칸 공백처리
                    data['요약 내용'].fillna('', inplace=True)

                    # 추가 날짜컬럼
                    data["날짜 (yyyymmdd)"] = pd.DataFrame(data["날짜 (yyyymmdd)"])
                    if not pd.DataFrame(data["날짜 (yyyymmdd)"]).empty and len(str(data["날짜 (yyyymmdd)"][0])) > 8:
                        data["날짜 (yyyymmdd)"] = pd.to_datetime(data["날짜 (yyyymmdd)"])
                        data["날짜 (yyyymmdd)"] = data["날짜 (yyyymmdd)"].dt.strftime('%Y%m%d')

                        data["게시일 (post_date)"] = pd.DataFrame(data["게시일 (post_date)"])
                        data["게시일 (post_date)"] = pd.to_datetime(data["게시일 (post_date)"])
                        data["게시일 (post_date)"] = data["게시일 (post_date)"].dt.strftime('%Y%m%d')
                        continue

                    if pd.DataFrame(data["날짜 (yyyymmdd)"]).empty:
                        print("데이터가 하나도 없다")
                        # 결과를 엑셀 파일로 저장
                        data.to_excel(writer, index=False, sheet_name=f"{filna}")
                        continue

                    # 요약 내용이 5개 미만인 경우
                    if len(data['요약 내용']) < 5:
                        print("데이터가 적어서 군집이 형성이 안돼요")
                        data['cluster_label'] = ''
                        data['tokens'] = ''
                        data.to_excel(writer, index=False, sheet_name=f"{filna}")
                        continue

                    if not any(data['요약 내용'].apply(lambda x: bool(x.strip()))):
                        print("요약내용이 없어서 제목으로 만들어요")
                        vectorizer = TfidfVectorizer()  # 벡터 초기화
                        X = vectorizer.fit_transform(data['제목'])

                        # 단어 파싱 리스트
                        word_list = vectorizer.get_feature_names_out()  # 벡터를 단어로바꿈
                        # print("word_list ::: ", word_list.astype(str).tolist())

                        # 군집화
                        kmeans = KMeans(n_clusters=10, random_state=42)  # default42
                        kmeans.fit(X)

                        # 행별 군집플래그 부여
                        cluster_labels = kmeans.labels_
                        data['cluster_label'] = cluster_labels

                        # 행별 단어 모음 추가
                        parsing_row_words_list = []
                        row_count = len(data['제목'])

                        okt = Okt()
                        for i in range(row_count):
                            nouns_row = okt.nouns(str(data['제목'][i]))
                            if not nouns_row:
                                nouns_row = word_tokenize(data['제목'][i])
                            parsing_row_words_list.append(nouns_row)

                        data['tokens'] = parsing_row_words_list
                        # print('parsing_row_words_list::::: ', parsing_row_words_list)

                        # 결과를 엑셀 파일로 저장
                        data.to_excel(writer, index=False, sheet_name=f"{filna}")
                        continue

                    # 요약내용컬럼 벡터화
                    vectorizer = TfidfVectorizer()  # TfidfVectorizer 초기화
                    X = vectorizer.fit_transform(data['요약 내용'])

                    # 단어 파싱 리스트
                    word_list = vectorizer.get_feature_names_out()

                    # 군집화
                    kmeans = KMeans(n_clusters=5, random_state=42)
                    kmeans.fit(X)

                    # 행별 군집플래그 부여
                    cluster_labels = kmeans.labels_
                    data['cluster_label'] = cluster_labels  # 'Cluster' 컬럼에 군집 레이블 추가

                    # 행별 단어 모음 추가
                    parsing_row_words_list = []
                    row_count = len(data['요약 내용'])
                    summary_data = pd.DataFrame(data["요약 내용"])
                    okt = Okt()
                    for i in range(row_count):
                        nouns_row = okt.nouns(str(summary_data['요약 내용'][i]))
                        if not nouns_row:
                            nouns_row = word_tokenize(summary_data['요약 내용'][i])

                        parsing_row_words_list.append(nouns_row)

                    data['tokens'] = parsing_row_words_list

                    # 결과를 엑셀 파일로 저장
                    data.to_excel(writer, index=False, sheet_name=f"{filna}")
    except Exception as e:
        print("군집화중 오류발생")
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        adres_list = sys.argv[1]
        filna_list = sys.argv[2]
        print('adres_list ::', adres_list)
        print('filna_list ::', filna_list)
        clustering(adres_list, filna_list)
    elif len(sys.argv) > 4:
        keyword_index = sys.argv.index('--keyword')
        adres_list = sys.argv[1:keyword_index]
        filna_list = sys.argv[keyword_index + 1:]
        print('adres_list ::::', adres_list)
        print('filna_list ::::', filna_list)
        clustering_lot(adres_list, filna_list)