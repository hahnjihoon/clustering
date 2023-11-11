# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import subprocess


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    command = ['python', 'cluster.py']

    # subprocess를 사용하여 cluster.py 실행
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    # 실행 결과 출력
    if process.returncode == 0:
        print("cluster.py 실행 성공!")
        # print("Output:", output.decode('utf-8'))
    else:
        print("cluster.py 실행 중 오류 발생.")
        # print("Error:", error.decode('utf-8'))
