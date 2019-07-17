# -*- coding: utf-8 -*-

import json
import cv2

from keras_train import Model
from public_data import MODEL_PATH,CASCADE_PATH,RECT_COLOR,NAME_COLOR,ip_url,JSON_PATH,UNKNOWN_COLOR


class Face_recognition:
    def __init__(self, capture_id=0):
        # 捕获指定摄像头的实时视频流
        self.cap = cv2.VideoCapture(capture_id)
        self.model = Model()
        self.model.load_model(file_path=MODEL_PATH)
        self.cascade = cv2.CascadeClassifier(CASCADE_PATH)
        # self.last_frame = None
        with open(JSON_PATH, 'r') as f:
            self.name_list_dict = json.loads(f.read())

    def actual_time_recognition(self, capture_id=0, window_name='face_recognition'):
        cv2.namedWindow(window_name)
        while True:
            ret, frame = self.cap.read()  # 读取一帧视频
            if ret is True:
                # if frame is None:
                #     frame = self.last_frame
                # else:
                #     self.last_frame = frame
                # 图像灰化
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                print('图像获取失败')
                self.cap = cv2.VideoCapture(capture_id)
                continue

            # 利用分类器识别出哪个区域为人脸
            faces = self.cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            if len(faces) > 0:
                for face in faces:
                    x, y, w, h = face

                    # 截取脸部图像提交给模型识别这是谁
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    probability, name_label = self.model.face_predict(image)
                    # print(name_label)
                    name = self.name_list_dict[str(name_label)]

                    # print('name_number:', name_number)
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), RECT_COLOR, thickness=2)

                    # 文字提示是谁
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if probability > 0.7:
                        cv2.putText(frame, name, (x + 30, y + 30), font, 1, NAME_COLOR, 2)
                    else:
                        cv2.putText(frame, 'unknow', (x + 30, y + 30), font, 1, UNKNOWN_COLOR, 2)

            cv2.imshow(window_name, frame)

            # 等待10毫秒看是否有按键输入
            k = cv2.waitKey(10)
            # 如果输入q则退出循环
            if k & 0xFF == ord('q'):
                break

        # 释放摄像头并销毁所有窗口
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = Face_recognition()
    fr.actual_time_recognition()

