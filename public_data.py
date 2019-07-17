
UNKNOWN_COLOR = (255, 0, 255)  # 未识别出人脸时显示的字体颜色
NAME_COLOR = (255, 0, 255)  # 识别出结果时显示的字体颜色
RECT_COLOR = (0, 255, 0)  # 人脸框颜色
CASCADE_PATH = "./data/haarcascade_frontalface_default.xml"  # 人脸识别分类器地址
MODEL_PATH = './model/face.model'  # 训练出的模型保存地址
JSON_PATH = './data/label_and_name.txt'  # 字典保存地址
IMAGE_SIZE = 64  # 用来训练的图片规格
file_url = 'D:/test_1.mp4'
ip_url = 'rtsp://admin:admin@172.29.7.240:8554/live'  # ip摄像头地址
clarify_flag=0  #超分辨率任务进度标志