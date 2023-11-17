# This is a sample Python script.
import asyncio
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import subprocess
import sys

# async def main():
#     input('address :: ')
#     address = sys.argv[1].lower()  # 경로
#
#     input('sheename :: ')
#     sheename = sys.argv[2].lower()  # 파일명
#     # sheetname = sys.argv[3].lower()  # 시트명(키워드)
#
#     await clustering(address, sheename)
#
#
# if __name__ == '__main__':
#
#     asyncio.run(main())

if __name__ == '__main__':

    # #sys.argv[0]은 스크립트 이름
    # adres = sys.argv[1]
    # filna = sys.argv[2]
    # print("주소 :: ", adres)
    # print("키워드 :: ", filna)
    #
    # command = ['python', 'cluster.py', adres, filna]
    # # command = ['python', 'cluster.py']
    # # print("command :: ", command)
    #
    # #인코딩문제해결
    # # if type(command) == type(u''):
    # #     command = command.encode('utf-8')
    #
    # # subprocess를 사용하여 cluster.py 실행
    # process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # # print("process :: ", process)
    # output, error = process.communicate()
    # # print("output :: ", output)
    # # print("error :: ", error)
    #
    # # 실행 결과 출력
    # try:
    #     print("cluster.py 실행 성공!")
    #     print("Output:", output.decode('utf-8', errors='replace'))
    # except UnicodeDecodeError:
    #     print("cluster.py 실행 성공! (디코딩 중 에러 발생)")
    #     print("Output:", output, errors='replace')
    #
    # if process.returncode != 0:
    #     print("cluster.py 실행 중 오류 발생.")
    #     print("Error:", error.decode('utf-8', errors='replace'))

    #=================================================================================================================
    # sys.argv[0]은 스크립트 이름

    if len(sys.argv) < 3:
        print("fail, enter like cf - python main.py 'C:/Users/Rainbow Brain/Desktop/네이버뉴스.xlsx' --keyword ' 베피스+기저귀' ' 키퍼스+기저귀' ' 코디(codi)'")
        sys.exit(1)

    elif len(sys.argv) == 3:
        # sys.argv[0]은 스크립트 이름
        adres = sys.argv[1]
        filna = sys.argv[2]
        print("주소 :: ", adres)
        print("키워드 :: ", filna)

        command = ['python', 'cluster.py', adres, filna]

    else:
        adres = sys.argv[1:sys.argv.index('--keyword')]
        filna = sys.argv[sys.argv.index('--keyword') + 1:]

        command = ['python', 'cluster.py'] + adres + ['--keyword'] + filna

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    output, error = process.communicate()

    # 실행 결과 출력
    try:
        print("cluster.py 실행 성공!")
        # print("Output:", output.decode('utf-8', errors='replace'))
        print("Output:", output)
    except UnicodeDecodeError:
        print("cluster.py 실행 성공! (디코딩 중 에러 발생)")
        print("Output:", output)

    if process.returncode != 0:
        print("cluster.py 실행 중 오류 발생.")
        print("Error:", error)