import os
import sys
import warnings
import pandas as pd
import logging
import string
import re
from collections import Counter

from konlpy.tag import Okt
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.DEBUG,  # 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


# 파일1개 시트1개일때
def clustering(adres_list, filna_list):
    # utf-8, ascii 확인
    # encode = chardet.detect(adres_list)
    # encode2 = chardet.detect(filna_list)

    # 데이터불러옴
    data = pd.read_excel(adres_list, header=0, sheet_name=filna_list)

    cluster_number = 25
    data_length = len(data)

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
    summary_data = pd.DataFrame(data["제목"])  # 제목만 있는 새로운 프레임
    summary_data = summary_data.assign(요약내용=data["요약 내용"])  # 제목에 요약내용추가 컬럼2개짜리 프레임

    # 빈칸 공백처리
    data['제목'].fillna('', inplace=True)
    # print("요약내용 있는지없는지 :: ", data['요약 내용'])

    # 요약 내용이 5개 미만인 경우
    # if len(data['제목']) < 5:
    #     print("데이터가 적어서 군집이 형성이 안돼요")
    #     data['cluster_label'] = ''
    #     data['tokens'] = ''
    #     output_file = os.path.splitext(adres_list)[0] + '_clustered.xlsx'
    #     data.to_excel(output_file, index=False, sheet_name=filna_list)
    #     exit()

    if not any(data['제목'].apply(lambda x: bool(x.strip()))):
        # 요약내용에 .apply(함수)를 적용시키는데 어떤함수냐 lambda x : x를 .strip시작끝공백제거해서 그결과가 빈문자면 flase 있으면true
        print("제목이 없어서 요약내용으로 만들어요")
        vectorizer = TfidfVectorizer()  # 벡터 초기화
        X = vectorizer.fit_transform(data['요약 내용'])

        # 단어 파싱 리스트
        word_list = vectorizer.get_feature_names_out()  # 벡터를 단어로바꿈
        # print("word_list ::: ", word_list.astype(str).tolist())

        if data_length < cluster_number:
            cluster_number = data_length

        # 군집화
        kmeans = KMeans(n_clusters=cluster_number, random_state=42)  # default42
        kmeans.fit(X)

        # 행별 군집플래그 부여
        cluster_labels = kmeans.labels_
        data['cluster_label'] = cluster_labels

        # 행별 단어 모음 추가
        parsing_row_words_list = []
        row_count = len(data['요약 내용'])

        okt = Okt()
        for i in range(row_count):
            nouns_row = okt.nouns(str(data['요약 내용'][i]))
            if not nouns_row:
                nouns_row = word_tokenize(data['요약 내용'][i])
            parsing_row_words_list.append(nouns_row)

        data['tokens'] = parsing_row_words_list
        # print('parsing_row_words_list::::: ', parsing_row_words_list)

        # 라벨 카운팅
        label_counts = data['cluster_label'].value_counts()
        most_common_label = label_counts.idxmax()

        # 카운팅된 가장많은수를 가진 숫자를 0으로 0은 그숫자로 치환
        data['cluster_label'] = data['cluster_label'] = data['cluster_label'].replace(
            {most_common_label: 0, 0: most_common_label})

        # 결과를 엑셀 파일로 저장
        output_file = os.path.splitext(adres_list)[0] + '_clustered.xlsx'
        # data.to_excel(output_file, index=False, sheet_name=filna_list)
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            data.to_excel(writer, index=False, sheet_name=filna_list)

            # 모든 단어를 하나의 리스트로 합치기
            all_words = []  # 토큰합친배열
            all_keywords = []  # 원본키워드컬럼
            all_usermail = []  # 유저이메일
            for tokens, keyword, email in zip(data['tokens'], data['키워드'], data['유저 이메일']):
                all_words.extend(tokens)
                all_keywords.extend([keyword] * len(tokens))
                all_usermail.extend([email] * len(tokens))

            # 공백 제거 및 특정 단어 삭제
            deleteWord = ['a', 'the', 'The', 'is', 'all', 'of', 's', 'for', 'in', 'With', 'with', 'Their', 'Its',
                          'will',
                          'Will', 'to', 't', 'To', 'has', 'It', 'Has', 'be', 'but', 'Be', 'But', 'and', 'And',
                          'Now', 'now',
                          'A',
                          'on', 'On', 'More', 'more', 'then', 'Then', 'That', 'that', 'Why', 'why', 'Yes', 'yes',
                          'no',
                          'No']

            # all_words = [re.sub(r'[^a-zA-Z가-힣\s]', '', word).strip() for word in all_words if
            #              len(re.sub(r'[^a-zA-Z가-힣\s]', '', word).strip()) > 1 and word not in deleteWord]
            # all_keywords = [keyword for word, keyword in zip(all_words, all_keywords) if word.strip()]
            # all_usermail = [email for word, email in zip(all_words, all_usermail) if word.strip()]

            word_counts = Counter(all_words)

            # 하나의 열로 만들어서 각 단어를 쉼표로 구분하여 각 행에 할당
            data2 = pd.DataFrame({'수집일': [data['날짜 (yyyymmdd)'][0]] * len(all_words)})
            data2 = data2.assign(이메일=all_usermail)
            data2 = data2.assign(키워드=all_words)
            data2 = data2.assign(분류=all_keywords)
            data2 = data2.assign(타입=all_keywords)
            data2 = data2.assign(빈도수=[word_counts[word] for word in all_words], )
            # data2 = data2.groupby('키워드').apply(lambda x: x.loc[x['빈도수'].idxmax()]).reset_index(drop=True)
            data2_no_duplicates = data2.drop_duplicates(subset=['이메일', '키워드'])
            data2_filtered = data2_no_duplicates[data2_no_duplicates['키워드'].str.len() > 1]
            data2_filtered = data2_filtered[~data2_filtered['키워드'].isin(deleteWord)]
            data2_filtered.to_excel(writer, index=False, sheet_name='news Keywords')
        exit()

    print('정상일때 로직시작 ::')
    # 요약내용컬럼 벡터화
    vectorizer = TfidfVectorizer()  # 벡터 초기화
    X = vectorizer.fit_transform(data['제목'])

    # 단어 파싱 리스트
    word_list = vectorizer.get_feature_names_out()  # 벡터를 단어로바꿈
    # print("word_list ::: ", word_list.astype(str).tolist())
    # print('단어파싱리스트 ::', word_list)

    if data_length < cluster_number:
        cluster_number = data_length

    # 군집화
    kmeans = KMeans(n_clusters=cluster_number, random_state=42)  # default42
    kmeans.fit(X)

    # 행별 군집플래그 부여
    cluster_labels = kmeans.labels_
    data['cluster_label'] = cluster_labels
    # print('행별 군집플래그 ::', data['cluster_label'])

    # 행별 단어 모음 추가
    parsing_row_words_list = []  # 밑에서담을 빈배열초기화
    row_count = len(summary_data['제목'])
    print('row_count ::', row_count)

    okt = Okt()
    for i in range(row_count):
        nouns_row = okt.nouns(str(summary_data['제목'][i]))  # 명사만 추출해서담는데 (한국어)
        if not nouns_row:
            nouns_row = word_tokenize(summary_data['제목'][i])  # 추출안됐으면 word_tokenize로 다시돌려 (영어)
        parsing_row_words_list.append(nouns_row)

    data['tokens'] = parsing_row_words_list
    # print('parsing_row_words_list::::: ', parsing_row_words_list)


    # 라벨 카운팅
    label_counts = data['cluster_label'].value_counts()
    most_common_label = label_counts.idxmax()
    # print('label_counts :: ', label_counts)
    # print('most_common_label :: ', most_common_label)

    # 카운팅된 가장많은수를 가진 숫자를 0으로 0은 그숫자로 치환
    # data['cluster_label'] = data['cluster_label'].apply(lambda x: most_common_label if x == 0 else 0)
    data['cluster_label'] = data['cluster_label'] = data['cluster_label'].replace(
        {most_common_label: 0, 0: most_common_label})
    # print('치환된 라벨값(0이항상많아야됨) :: ', data['cluster_label'])

    # print('저장전data :: ', data)
    # 결과를 엑셀 파일로 저장
    output_file = os.path.splitext(adres_list)[0] + '_clustered.xlsx'
    # data.to_excel(output_file, index=False, sheet_name=filna_list)
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        data.to_excel(writer, index=False, sheet_name=filna_list)

        # 모든 단어를 하나의 리스트로 합치기
        all_words = []  # 토큰합친배열
        all_keywords = []  # 원본키워드컬럼
        all_usermail = []  # 유저이메일
        for tokens, keyword, email in zip(data['tokens'], data['키워드'], data['유저 이메일']):
            all_words.extend(tokens)
            # 각 토큰에 해당하는 키워드와 이메일을 추가
            all_keywords.extend([keyword] * len(tokens))
            all_usermail.extend([email] * len(tokens))

        # 공백 제거 및 특정 단어 삭제
        deleteWord = ['a', 'the', 'The', 'is', 'all', 'of', 's', 'for', 'in', 'With', 'with', 'Their', 'Its',
                      'will',
                      'Will', 'to', 't', 'To', 'has', 'It', 'Has', 'be', 'but', 'Be', 'But', 'and', 'And',
                      'Now', 'now',
                      'A',
                      'on', 'On', 'More', 'more', 'then', 'Then', 'That', 'that', 'Why', 'why', 'Yes', 'yes',
                      'no',
                      'No']

        # all_words = [re.sub(r'[^a-zA-Z가-힣\s]', '', word).strip() for word in all_words if
        #              len(re.sub(r'[^a-zA-Z가-힣\s]', '', word).strip()) > 1 and word not in deleteWord]
        # all_keywords = [keyword for word, keyword in zip(all_words, all_keywords) if word.strip()]
        # all_usermail = [email for word, email in zip(all_words, all_usermail) if word.strip()]
        # 이 중복제거속에서 에러가 남

        word_counts = Counter(all_words)

        # 하나의 열로 만들어서 각 단어를 쉼표로 구분하여 각 행에 할당
        data2 = pd.DataFrame({'수집일': [data['날짜 (yyyymmdd)'][0]] * len(all_words)})
        data2 = data2.assign(이메일=all_usermail)
        data2 = data2.assign(키워드=all_words)
        data2 = data2.assign(분류=all_keywords)
        data2 = data2.assign(타입=all_keywords)
        data2 = data2.assign(빈도수=[word_counts[word] for word in all_words],)
        # data2 = data2.groupby('키워드').apply(lambda x: x.loc[x['빈도수'].idxmax()]).reset_index(drop=True)
        # data2 = data2[data2['키워드'].str.len() > 1]
        # data2 = data2[~data2['키워드'].isin(deleteWord)]
        # data2 = data2.drop_duplicates(subset=['키워드'])
        # data2 = data2.groupby(['이메일', '키워드']).size().reset_index(name='count')
        # data2 = data2.drop_duplicates(subset=['이메일', '키워드'], keep='first')
        data2_no_duplicates = data2.drop_duplicates(subset=['이메일', '키워드'])
        data2_filtered = data2_no_duplicates[data2_no_duplicates['키워드'].str.len() > 1]
        data2_filtered = data2_filtered[~data2_filtered['키워드'].isin(deleteWord)]
        # 쳇지피티 중복제거 코드

        data2_filtered.to_excel(writer, index=False, sheet_name='news Keywords')

# 파일n개 시트n개일때
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
                    # if len(data['제목']) < 5:
                    #     print("데이터가 적어서 군집이 형성이 안돼요")
                    #     data['cluster_label'] = ''
                    #     data['tokens'] = ''
                    #     data.to_excel(writer, index=False, sheet_name=f"{filna}")
                    #     continue
                    cluster_number = 25
                    data_length = len(data)

                    if not any(data['제목'].apply(lambda x: bool(x.strip()))):
                        print("제목이 없어서 요약내용으로 만들어요")
                        vectorizer = TfidfVectorizer()  # 벡터 초기화
                        X = vectorizer.fit_transform(data['요약 내용'])

                        # 단어 파싱 리스트
                        word_list = vectorizer.get_feature_names_out()  # 벡터를 단어로바꿈
                        # print("word_list ::: ", word_list.astype(str).tolist())

                        if data_length < cluster_number:
                            cluster_number = data_length

                        # 군집화
                        kmeans = KMeans(n_clusters=cluster_number, random_state=42)  # default42
                        kmeans.fit(X)

                        # 행별 군집플래그 부여
                        cluster_labels = kmeans.labels_
                        data['cluster_label'] = cluster_labels

                        # 행별 단어 모음 추가
                        parsing_row_words_list = []
                        row_count = len(data['요약 내용'])

                        okt = Okt()
                        for i in range(row_count):
                            nouns_row = okt.nouns(str(data['요약 내용'][i]))
                            if not nouns_row:
                                nouns_row = word_tokenize(data['요약 내용'][i])
                            parsing_row_words_list.append(nouns_row)

                        data['tokens'] = parsing_row_words_list
                        # print('parsing_row_words_list::::: ', parsing_row_words_list)

                        # 라벨 카운팅
                        label_counts = data['cluster_label'].value_counts()
                        most_common_label = label_counts.idxmax()

                        # 카운팅된 가장많은수를 가진 숫자를 0으로 0은 그숫자로 치환
                        data['cluster_label'] = data['cluster_label'] = data['cluster_label'].replace(
                            {most_common_label: 0, 0: most_common_label})

                        # 결과를 엑셀 파일로 저장
                        data.to_excel(writer, index=False, sheet_name=f"{filna}")

                        # second_sheet_data = pd.DataFrame({
                        #     '번호': range(1, len(data) + 1),
                        #     '이름': ['' for _ in range(len(data))]
                        # })
                        # second_sheet_data.to_excel(writer, index=False, sheet_name='secondSheet')
                        continue

                    # 요약내용컬럼 벡터화
                    vectorizer = TfidfVectorizer()  # TfidfVectorizer 초기화
                    X = vectorizer.fit_transform(data['제목'])

                    # 단어 파싱 리스트
                    word_list = vectorizer.get_feature_names_out()

                    if data_length < cluster_number:
                        cluster_number = data_length

                    # 군집화
                    kmeans = KMeans(n_clusters=cluster_number, random_state=42)
                    kmeans.fit(X)

                    # 행별 군집플래그 부여
                    cluster_labels = kmeans.labels_
                    data['cluster_label'] = cluster_labels  # 'Cluster' 컬럼에 군집 레이블 추가

                    # 행별 단어 모음 추가
                    parsing_row_words_list = []
                    row_count = len(data['제목'])
                    summary_data = pd.DataFrame(data["제목"])
                    okt = Okt()
                    for i in range(row_count):
                        nouns_row = okt.nouns(str(summary_data['제목'][i]))
                        if not nouns_row:
                            nouns_row = word_tokenize(summary_data['제목'][i])

                        parsing_row_words_list.append(nouns_row)

                    data['tokens'] = parsing_row_words_list

                    # 라벨 카운팅
                    label_counts = data['cluster_label'].value_counts()
                    most_common_label = label_counts.idxmax()

                    # 카운팅된 가장많은수를 가진 숫자를 0으로 0은 그숫자로 치환
                    data['cluster_label'] = data['cluster_label'] = data['cluster_label'].replace(
                        {most_common_label: 0, 0: most_common_label})

                    # 결과를 엑셀 파일로 저장
                    data.to_excel(writer, index=False, sheet_name=f"{filna}")

                # 모든 단어를 하나의 리스트로 합치기
                all_words = []  # 토큰합친배열
                all_keywords = []  # 원본키워드컬럼
                all_usermail = []  # 유저이메일
                for tokens, keyword, email in zip(data['tokens'], data['키워드'], data['유저 이메일']):
                    all_words.extend(tokens)
                    all_keywords.extend([keyword] * len(tokens))
                    all_usermail.extend([email] * len(tokens))

                # 공백 제거 및 특정 단어 삭제
                deleteWord = ['a', 'the', 'The', 'is', 'all', 'of', 's', 'for', 'in', 'With', 'with', 'Their', 'Its',
                              'will',
                              'Will', 'to', 't', 'To', 'has', 'It', 'Has', 'be', 'but', 'Be', 'But', 'and', 'And',
                              'Now', 'now',
                              'A',
                              'on', 'On', 'More', 'more', 'then', 'Then', 'That', 'that', 'Why', 'why', 'Yes', 'yes',
                              'no',
                              'No']

                # all_words = [re.sub(r'[^a-zA-Z가-힣\s]', '', word).strip() for word in all_words if
                #              len(re.sub(r'[^a-zA-Z가-힣\s]', '', word).strip()) > 1 and word not in deleteWord]
                # all_keywords = [keyword for word, keyword in zip(all_words, all_keywords) if word.strip()]
                # all_usermail = [email for word, email in zip(all_words, all_usermail) if word.strip()]

                word_counts = Counter(all_words)

                # 하나의 열로 만들어서 각 단어를 쉼표로 구분하여 각 행에 할당
                data2 = pd.DataFrame({'수집일': [data['날짜 (yyyymmdd)'][0]] * len(all_words)})
                data2 = data2.assign(이메일=all_usermail)
                data2 = data2.assign(키워드=all_words)
                data2 = data2.assign(분류=all_keywords)
                data2 = data2.assign(타입=all_keywords)
                data2 = data2.assign(빈도수=[word_counts[word] for word in all_words], )
                # data2 = data2.groupby('키워드').apply(lambda x: x.loc[x['빈도수'].idxmax()]).reset_index(drop=True)
                data2_no_duplicates = data2.drop_duplicates(subset=['이메일', '키워드'])
                data2_filtered = data2_no_duplicates[data2_no_duplicates['키워드'].str.len() > 1]
                data2_filtered = data2_filtered[~data2_filtered['키워드'].isin(deleteWord)]
                data2.to_excel(writer, index=False, sheet_name='news Keywords')

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
