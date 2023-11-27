# -*- coding: utf-8 -*-
# This is a sample Python script.
import asyncio
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import subprocess
import sys

import chardet

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("fail, enter like cf - python main.py 'C:/Users/Rainbow Brain/Desktop/네이버뉴스.xlsx' --keyword ' 베피스+기저귀' ' 키퍼스+기저귀' ' 코디(codi)'")
        sys.exit(1)

    elif len(sys.argv) == 3:
        # sys.argv[0]은 스크립트 이름
        adres = sys.argv[1]
        filna = sys.argv[2]
        result = chardet.detect(adres.encode())
        print("adres_encode :: ", result)

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