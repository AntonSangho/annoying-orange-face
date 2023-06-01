import cv2
# dblib의 얼굴 탐지랑 랜드마크 탐지 사용 
import dlib
# 사이즈 변환 
from imutils import face_utils, resize
import numpy as np

orange_img = cv2.imread('orange.jpg')
# 이미지가 크기 때문에 조정
orange_img = cv2.resize(orange_img, dsize=(512, 512))

# 얼굴영역 탐지하는 부분 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#cap = cv2.VideoCapture(0)
# 동영상을 동작시키고 열굴만 따서 할수 있음
cap = cv2.VideoCapture('01.mp4')

while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        break
    # 얼굴영역을 인식해주는 부분 
    # faces에 좌표정보가 들어감
    faces = detector(img)

    result = orange_img.copy()

    #얼굴이 한개 이상일 때 사용하는 부분인데 기본적으로 한개이상임
    if len(faces) > 0:
        # 0번째 index를 face에 저장하고
        face = faces[0]
        # 얼굴의 왼쪽,위쪽,오른쪽,아래쪽을 저장해준다음에  
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        # face만 crop(자르기)
        face_img = img[y1:y2, x1:x2].copy()

        shape = predictor(img, face)
        # shape에 face landmark 68(https://needjarvis.tistory.com/641)의 얼굴 정보가 담기게된다.
        # face_utils에 있는 shape를 numpy로 변경해주는 부분 
        shape = face_utils.shape_to_np(shape)

        # 아래 주석 해제하고 eye, mouth 주석처리한 부분 하고 cv2.imshow('face', face_img) 주석해제하면 얼굴에 점이 찍힘 
        for p in shape:
            cv2.circle(face_img, center=(p[0] - x1, p[1] - y1), radius=2, color=255, thickness=-1)

        # eyes
        # 눈을 자르기 위해 점에서 x축 36에서 39까지 범위자르고, y축 37에서 41까지 자르기
        le_x1 = shape[36, 0]
        le_y1 = shape[37, 1]
        le_x2 = shape[39, 0]
        le_y2 = shape[41, 1]
        # 너무 타이트하지 않게 자리기 위해 마진을 두었음
        le_margin = int((le_x2 - le_x1) * 0.18)

        re_x1 = shape[42, 0]
        re_y1 = shape[43, 1]
        re_x2 = shape[45, 0]
        re_y2 = shape[47, 1]
        re_margin = int((re_x2 - re_x1) * 0.18)

        # 왼쪽눈와 오른쪽 눈 마진을 줘서 crop을 한다.
        left_eye_img = img[le_y1-le_margin:le_y2+le_margin, le_x1-le_margin:le_x2+le_margin].copy()
        right_eye_img = img[re_y1-re_margin:re_y2+re_margin, re_x1-re_margin:re_x2+re_margin].copy()

        left_eye_img = resize(left_eye_img, width=100)
        right_eye_img = resize(right_eye_img, width=100)

        # poison blending(https://velog.io/@claude_ssim/%EA%B3%84%EC%82%B0%EC%82%AC%EC%A7%84%ED%95%99-Image-Blending)이라고 해서 티가 안나게 합성을 해준다. 
        result = cv2.seamlessClone(
            left_eye_img,
            result,
            np.full(left_eye_img.shape[:2], 255, left_eye_img.dtype),
            (100, 200),
            # 아래 옵션을 줘서 알아서 합성하도록 한다.
            cv2.MIXED_CLONE
        )

        # result는 오렌지 이미지를 카피한거 
        result = cv2.seamlessClone(
            right_eye_img,
            result,
            np.full(right_eye_img.shape[:2], 255, right_eye_img.dtype),
            (250, 200),
            cv2.MIXED_CLONE
        )

        # mouth
        mouth_x1 = shape[48, 0]
        mouth_y1 = shape[50, 1]
        mouth_x2 = shape[54, 0]
        mouth_y2 = shape[57, 1]
        mouth_margin = int((mouth_x2 - mouth_x1) * 0.1)

        # crop을 해서 마우스 이미지 저장 
        mouth_img = img[mouth_y1-mouth_margin:mouth_y2+mouth_margin, mouth_x1-mouth_margin:mouth_x2+mouth_margin].copy()

        mouth_img = resize(mouth_img, width=250)

        result = cv2.seamlessClone(
            mouth_img,
            result,
            np.full(mouth_img.shape[:2], 255, mouth_img.dtype),
            (180, 320),
            cv2.MIXED_CLONE
        )

        #cv2.imshow('left', left_eye_img)
        #cv2.imshow('right', right_eye_img)
        #cv2.imshow('mouth', mouth_img)
        #cv2.imshow('face', face_img)

        cv2.imshow('result', result)

    # cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break