import random

from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np

from get_localFiles import FileOperator
from public_data import IMAGE_SIZE, MODEL_PATH


# 定义各数据集
class Dataset:
    def __init__(self, data_path):
        # 数据集加载路径
        self.data_path = data_path

        # 训练集
        self.train_images = None
        self.train_labels = None

        # 验证集
        self.valid_images = None
        self.valid_labels = None

        # 测试集
        self.test_images = None
        self.test_labels = None

        # 当前库采用的维度顺序
        self.input_shape = None

        self.face_num = None

        self.file_operator = FileOperator()

    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3):
        """加载数据集并对其进行预处理"""

        # 加载数据集到内存
        images, labels, face_num = self.file_operator.load_dataset(self.data_path)
        self.face_num = face_num

        # 随机划分训练集和验证集
        train_images, valid_images, train_labels, valid_labels = \
            train_test_split(images, labels, test_size=0.3, random_state=random.randint(0, 100))
        # 随机划分测试集
        _, test_images, _, test_labels = \
            train_test_split(images, labels, test_size=0.5, random_state=random.randint(0, 100))

        # 当前的维度顺序如果为'th'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
        # 这部分代码就是根据keras库要求的维度顺序重组训练数据集
        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

            # 输出训练集、验证集、测试集的数量
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')

            # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将类别标签进行one-hot编码使其向量化
            # print(train_labels, 'train labels')
            # print(valid_labels, 'valid labels')
            # print(test_labels, 'test labels')

            train_labels = np_utils.to_categorical(train_labels, self.face_num)
            valid_labels = np_utils.to_categorical(valid_labels, self.face_num)
            test_labels = np_utils.to_categorical(test_labels, self.face_num)

            # print(train_labels,'train labels')
            # print(valid_labels,'valid labels')
            # print(test_labels,'test labels')

            # 像素数据浮点化以便归一化
            train_images = train_images.astype('float32')
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')

            # 将其归一化,图像的各像素值归一化到0~1区间
            train_images /= 255
            valid_images /= 255
            test_images /= 255

            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels = test_labels


# CNN网络模型类
class Model:
    def __init__(self):
        self.model = None
        self.file_operator = FileOperator()

    def build_model(self, dataset, nb_classes=6):
        """建立模型"""

        # 构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.model = Sequential()

        # 以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
        # 2*(2*卷积层+池化层)+Flatten层+2*全连接层
        self.model.add(Convolution2D(32, 3, 3, border_mode='same',
                                     input_shape=dataset.input_shape))
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())  # 把多维的输入一维化,用于卷积层到全连接层的过渡
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        # 输出模型概况
        self.model.summary()

    def train(self, dataset, batch_size=20, nb_epoch=10, image_data_generator=True):
        """训练模型"""

        # SGD:随机梯度下降,但容易在最小点震荡,容易陷入局部最优解
        # lr:学习率
        # decay:学习率衰减值
        # moment:动量因子，可以加快收敛，减小震荡，一般取0.9
        # nesterov:确定是否使用Nesterov动量
        sgd = SGD(lr=0.0007, decay=1e-6,momentum=0.9, nesterov=True)
        # loss:定义损失函数,必须且仅能指定一个
        # optimizer:指定优化方式,必须且仅能指定一个
        # metrics:指定衡量模型的指标,accuracy指自定义操作函数,可以同时指定多个操作函数
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # 完成实际的模型配置工作

        # 不使用数据提升
        if not image_data_generator:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(dataset.valid_images, dataset.valid_labels),
                           shuffle=True)
        # 使用数据提升
        else:
            # 参照自https://blog.csdn.net/hnu2012/article/details/54017564
            # 数据提升可以通过对数据的操作(旋转，偏移等)解决分类数据过少导致的模型过拟合问题
            IDG = ImageDataGenerator(
                featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
                samplewise_center=False,  # 是否使输入数据的每个样本均值为0
                featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
                zca_whitening=False,  # 是否对输入数据施以ZCA白化
                rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range=0.2,  # 同上，只不过这里是垂直
                horizontal_flip=True,  # 是否进行随机水平翻转
                vertical_flip=False)  # 是否进行随机垂直翻转

            # 对训练数据集进行整体的数据提升
            IDG.fit(dataset.train_images)

            # fit_generator为训练模型生产批次数据，其功能和fit相同，但需要使用生成器来批次包装数据
            # fit_generator较fit在处理大批量的数据时更有优越性
            self.model.fit_generator(
                IDG.flow(dataset.train_images, dataset.train_labels,batch_size=batch_size),  # 数据提升后的训练集逐个输出训练
                samples_per_epoch=dataset.train_images.shape[0],
                nb_epoch=nb_epoch,
                validation_data=(dataset.valid_images, dataset.valid_labels))

    def save_model(self, file_path=MODEL_PATH):
        """保存模型"""
        self.model.save(file_path)

    def load_model(self, file_path=MODEL_PATH):
        """加载模型"""
        self.model = load_model(file_path)

    def evaluate_model(self, dataset):
        # evaluate返回损失值和模型的指标值，即上面定义的accuracy
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    def face_predict(self, image):
        """识别人脸"""
        # 依然是根据后端系统确定维度顺序
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = self.file_operator.resize_image(image)  # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # 与模型训练不同，这次只是针对1张图片进行预测
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = self.file_operator.resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

            # 浮点并归一化
        image = image.astype('float32')
        image /= 255

        # predict_proba返回预测样本为某个标签的概率
        # predict_classes返回预测样本可能性最大的标签
        result_probability = self.model.predict_proba(image)
        # print('result:', result_probability, max(result_probability[0]))

        # 给出类别预测：0-9
        result = self.model.predict_classes(image)

        # 返回类别预测结果
        return max(result_probability[0]),result[0]


if __name__ == '__main__':
    np.set_printoptions(threshold=5000)  # print数组时不会省略输出(5000以下)
    dataset = Dataset('./data')
    dataset.load()
    model = Model()
    model.build_model(dataset, dataset.face_num)
    model.train(dataset)
    model.save_model()
    model.evaluate_model(dataset)
