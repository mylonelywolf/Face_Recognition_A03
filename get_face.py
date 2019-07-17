import cv2
from public_data import ip_url,CASCADE_PATH,RECT_COLOR,NAME_COLOR


class Face_Geter:
    def __init__(self, save_path, capture_id=0, is_save=False):
        self.save_path = save_path
        self.capture_id = capture_id
        self.is_save = is_save
        self.window_name = 'Get_Face'
        self.get_num = 1000
        self.cap = cv2.VideoCapture(self.capture_id)
        self.cascade = cv2.CascadeClassifier(CASCADE_PATH)
        # self.last_frame = None

    def get_faces(self):
        """获取人脸(数据集)"""

        # 定义窗口名称
        cv2.namedWindow(self.window_name)

        num = 0
        while True:
            # 读取当前一帧图像数据
            ret, frame = self.cap.read()
            if ret is True:
                # 图像灰化
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                print('图像获取失败')
                continue

            # 获取当前帧中的人脸
            # scaleFactor每次图像尺寸减小的比例
            # minNeighbors每一个目标需要循环检测到的次数
            # minSize目标最小尺寸
            # maxSize目标最大尺寸
            face_positions = self.cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            if len(face_positions) > 0:  # 如果检测到人脸
                x, y, w, h = face_positions[0]  # 每次只获取一张人脸数据

                # 将当前帧保存为图片
                if num <= self.get_num and self.is_save:
                    img_path = '{}/{}.jpg'.format(self.save_path, num)
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    if image is not None:
                        cv2.imwrite(img_path, image)
                        num += 1

                # 画出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), RECT_COLOR, 2)

                # 显示当前捕捉到了多少人脸图片
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, '{}'.format(num), (x + 30, y + 30), font, 1, NAME_COLOR, 4)

            # 显示图像
            cv2.imshow(self.window_name, frame)
            c = cv2.waitKey(10)
            if c & 0xFF == ord('q'):
                break
            elif c & 0xFF == ord('s'):
                self.is_save = True

        # 释放摄像头并销毁所有窗口
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    getter = Face_Geter('./data/JRJ')
    getter.get_faces()
