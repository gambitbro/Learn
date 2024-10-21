```python
int
float
as
if not
```


```python
# 비디오 저장
import cv2  # OpenCV 라이브러리를 가져온다.
from pop import Util    # pop 모듈에서 Util 클래스 불러오기.

Util.enable_imshow()    # imshow() 활성화. pop라이브러리에서 전처리한다.

cam = Util.gstrmer(width=640,height=480)    # GStreamer를 사용하여 비디오 스트림 설정
camera = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)   # 카메라 장치를 cv2.VideoCapture로 연다. cam은 Util.gstrmer에서 설정한 GStreamer 비디오 스트림을 나타낸다.

if not camera.isOpened():
    print("Not found camera")   # 카메라가 열렸는지 확인, 안되어있으면 프린트

fourcc = cv2.VideoWriter_fourcc(*"PIM1")    # 비디오 코덱을 MPEG-1으로 설정
out = cv2.VideoWriter("soda.avi", fourcc, 30, (640,480))    # 비디오 파일로 저장하기 위해 cv2.VideoWriter의 객체를 생성한다. soda.avi라는 이름, MPEG-1코덱, 30프레임, 640x480해상도로 설정한다.

for _ in range(120):    # 120번 루프. 1초당 30프레임이므로 약 4초분량 비디오.
    ret, frame = camera.read()  #카메라를 읽고 프레임을 저장, ret은 불리언.
    framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #컬러->그레이

    out.write(frame)    #컬러(원본)을 일단 설정해놓은 out파일(soda.avi)에 저장

    cv2.imshow("soda", frameGray)   #"soda"라는 창에 그레이컬러 비디오 보여주기

camera.release()    # 카메라 해제
cv2.destroyAllWindows() # 모든 OpenCV 창 닫기
```

```python
# 카메라 활용
import cv2
from pop import Util

Util.enable_imshow()

cam = Util.gstrmer(width=640,height=480)

camera = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
if not camera.isOpened():
    print("Not found camera")

# 폭, 높이 정보 띄우기
width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("init width:%d, init height:%d" %(width, height))

for _ in range(120):
    ret, frame = camera.read()
    if not ret:
        break
    cv2.imshow("soda", frame)

camera.release()
cv2.destroyAllWindows()
```

```python
# 이미지 표시
import cv2
from pop import Util

Util.enable_imshow()

Util.creatIMG()     # 테스트용 이미지 파일 생성

# 이미지 정보 출력
image = cv2.imread("img.jpg", cv2.IMREAD_COLOR)
height, width, channel = image.shape
print("width:%d, height:%d, channel:%d" %(width, height, channel))

# 이미지 보기
filename = "img.jpg"
imgColor = cv2.imread(filename, cv2.IMREAD_COLOR)
imgGray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

cv2.imshow("Color", imgColor)
cv2.imshow("GrayScale", imgGray)
```

```python
# Edge 검출
import cv2
from pop import Util

Util.enable_imshow()

cam = Util.gstrmer(width=640, height=480)
camera = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
if not camera.isOpened():
    print("Not found camera")

for _ in range(120):
    ret, frame = camera.read()
    if ret == False:
        print("Not found camera")
        break
    else:
        imgCanny = cv2.Canny(frame, 100, 200)    # 100은 첫 번째 임계값, 200은 두 번째 임계값이다. 100이하의 경사도(gradient)를 가지는 픽셀들은 엣지가 아니라고 판단하여 제거, 200이상의 경사도를 가지는 픽셀들은 강한 엣지로 간주하여 유지한다. 100과 200 사이의 경사도를 가지는 픽셀들은 주변에 강한 엣지가 있으면 엣지로 간주, 그렇지 않으면 엣지가 아니라고 판단한다.
        cv2.imshow("soda", imgCanny)

camera.release()
cv2.destroyAllWindows()
```

```python
# Haar Cascades를 이용한 얼굴 인식
import cv2
from pop import Util

Util.enable_imshow()

haar_face = '/경로/파일.xml'
face_cascade = cv2.CascadeClassifier(haar_face)     # 얼굴 검출기 만들기

cam = Util.gstrmer(width=640, height=480)
bunny = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
if bunny.isOpened() == False:
    print("Not found camera")

for _ in range(300):
    ret, img = bunny.read()
    if not ret:
        print("Not found camera")
        break
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출 수행, scaleFactor=1.3은 1.3배씩 이미지를 축소하며 얼굴을 찾는다는 뜻, 다양한 얼굴을 감지할 수 있다. minNeighbors=1은 감지된 영역을 얼굴로 확인하기 위한 주변 이웃의 개수. minSize=(100,100)은 감지할 얼굴의 최소 크기를 설정, 최소 100x100크기 이상인 얼굴만 검출한다.
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1, minSize=(100,100))

        #감지된 얼굴들의 좌표와 크기를 순차적으로 반복. faces는 감지된 얼굴들의 좌표와 크기를 튜플 형식으로 반환한다. (x, y)는 얼굴 영역의 좌표, w는 너비, h는 높이.
        for (x, y, w, h) in faces:
            # 주어진 이미지 특정 좌표에 사각형을 그린다(얼굴 주위). (x,y)는 왼쪽 상단 모서리, (x+w,y+h)는 오른쪽 하단 모서리 좌표. (255,0,0)은 사각형의 색상을 나타낸다(파란색). 2는 사각형의 두께이다.
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        # 'img'라는 창에 img 출력.
        cv2.imshow('img', img)
    
camera.release()
cv2.destroyAllWindows()
```