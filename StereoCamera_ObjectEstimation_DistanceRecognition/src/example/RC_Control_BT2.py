import serial

# 시리얼 포트 설정
ser = serial.Serial(
    port='/dev/cu.usbserial-1130',  # Windows의 경우 'COMx' (예: COM3), macOS/Linux의 경우 '/dev/ttyUSB0' 또는 '/dev/ttyS0'
    baudrate=9600,        # 보드레이트 설정 (일반적으로 9600이 사용됨)
    timeout=1             # 타임아웃 설정 (초 단위)
)
# 시리얼 포트가 열려 있는지 확인
if ser.is_open:
    print("시리얼 포트가 열려 있습니다.")



# 데이터 송신
def sending_Data():  
    send_data = str(input("전송할 메시지를 입력해주세요: "))
    ser.write(send_data.encode('utf-8'))  # 문자열을 바이트로 인코딩하여 전송

# 데이터 수신
def receiving_Data():
    received_data = ser.readline().decode('utf-8').strip()  # 수신된 데이터는 바이트이므로 디코딩 필요
    print(f"수신된 데이터: {received_data}")

# 시리얼 포트 닫기
def closing():
    ser.close()
    print("\n\ndisconnected")
    print("disconnected")
    print("disconnected\n\n")



# main 함수
def main():
    coin = 1
    while coin > 0:
        sending_Data()
        coin -= 1        
'''
    coin = 1
    print("Receiving")
    while coin > 0:
        receiving_Data()
        coin -= 1
'''

##############################

if __name__ == "__main__":
    main()
