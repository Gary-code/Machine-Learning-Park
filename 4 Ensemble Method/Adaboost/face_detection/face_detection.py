import cv2

# OpenCV自带的基于Haar特征和AdaBoost的人脸检测方法
if __name__ == '__main__':

    # 为 haarcascade 加载级联分类器训练文件
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    
    # 加载测试图片
    img = cv2.imread('./Lena.png')

    # 将测试图像转换为灰度图，因为opencv人脸检测器需要灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 开始检测
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 画出检测的区域
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 展示成果
    cv2.imshow('img', img)
    cv2.imwrite('./result.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()