import cv2
import time
import predict as predict
from keras.models import load_model

model = load_model("intraoralArea.hdf5")
def detect():
    # 创建人脸检测的对象
    classifier_face = cv2.CascadeClassifier("/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_alt.xml")
    # 连接摄像头的对象，0表示摄像头的编号
    camera = cv2.VideoCapture(0)

    t = time.time()
    num = 0;
    while True:
        if time.time()-t < 1:
            num=num+1
        else :
            print("当前fps="+str(num))
            num=0
            t = time.time()

        # 读取当前帧
        ret, img = camera.read()

        faceRects_face = classifier_face.detectMultiScale(img, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, (20, 20))
        # 检测器：detectMultiScale参数（图像，每次缩小图像的比例，匹配成功所需要的周围矩形框的数目，检测的类型，匹配物体的大小范围）
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if len(faceRects_face) > 0:
            # 检测到人脸q
            for faceRect_face in faceRects_face:
                x, y, w, h = faceRect_face
                # draw face rectangle
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 225, 225), 2)
                mx = x + (int)(w / 4)
                mw = (int)(w / 2)
                my = y + (int)(h * 0.65)
                mh = (int)(h / 4)

                img = cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (255, 225, 0), 2)

                img_mouth = img[my:my + mh, mx:mx + mw]
                # img_mouth_gray = gray_proceess(img_mouth)
                # img_mouth_seg = seg_process(img_mouth)

                # seg_img = cv2.resize(seg_img, (256, 256))
                img_mouth = cv2.resize(img_mouth,(256,128))

                # cv2.imshow('mouth_gray', img_mouth_gray)
                img_mouth_seg = seg_process(img_mouth)
                cv2.imshow('before', img_mouth)
                img_mouth_seg = cv2.resize(img_mouth_seg , (256, 128))
                cv2.imshow("mouth_seg", img_mouth_seg/255)
        cv2.imshow('face', img)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break;

    camera.release()
    cv2.destroyAllWindows()

def gray_proceess(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def seg_process(img):

    return predict.getPredictValueByImg(img, model)

if __name__=='__main__':
    detect()

