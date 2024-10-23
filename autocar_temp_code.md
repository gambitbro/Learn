```python

from pop.Pilot import AutoCar
from time import sleep

car = AutoCar()     # 객체 생성

# IMU 센서(관성 측정 장치)
    # 캘리브레이션
car.setCalibration()
    # 오일러 값 읽기
car.setSensorStatus(euler=1)
eu = car.getEuler()
print(eu[0],eu[1],eu[2])
car.setSensorStatus(euler=0)
    # 서보 모터(스티어링: -1.0 ~ 1.0)
car.steering = 1.0      # 오른쪽
sleep(1)
car.steering = -1.0     # 왼쪽
sleep(1)
car.steering = 0
    # DC 모터(주행: 20 ~ 100)
car.forward(50)
sleep(1)
car.backward(100)
sleep(1)
car.stop()
    # LED
car.setLamp(1,0)    # 앞1 뒤0
sleep(1)
car.setLamp(0,1)    # 앞0 뒤1
sleep(1)
car.setLamp(0,0)
    # Ultrasonic (10 ~ 180cm)
for _ in range(10):
    us = car.getUltrasonic()
    print(us[0], us[1])
    sleep(0.1)

    # Buzzer
car.alarm(scale=4,pitch=8,duration=0.5)
sleep(0.5)
car.alarm(scale=4,pitch=8,duration=0.01)

# ?
from pop.Pilot import AutoCar
from time import sleep

car = AutoCar()
car.setSensorStatus(euler=1)
car.forward(80)
car.steering = 0.0

for i in range(100):
    eu = car.getEuler()
    sleep(0.1)
    print(eu)

car.setSensorStatus(euler=0)
car.stop()

```

