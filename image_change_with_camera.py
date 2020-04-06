import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os
import time


def btnHelloClicked():
    labelHello.config(text="Hello Tkinter!")


def resize(w, h, w_box, h_box, im):
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return im.resize((width, height), Image.ANTIALIAS)

def gray_proceess(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 创建人脸检测的对象
classifier_face = cv2.CascadeClassifier("/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_alt.xml")
# 连接摄像头的对象，0表示摄像头的编号
camera = cv2.VideoCapture(0)

t = time.time()
num = 0;

top = tk.Tk()
# -------------- image 1 --------------
while True:
    fps_text=""
    if time.time()-t < 1:
        num=num+1
    else :
        fps_text="当前fps="+str(num)
        print("当前fps="+str(num))
        num=0
        t = time.time()

    # 读取当前帧
    ret, img = camera.read()

    faceRects_face = classifier_face.detectMultiScale(img, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, (20, 20))
    # 检测器：detectMultiScale参数（图像，每次缩小图像的比例，匹配成功所需要的周围矩形框的数目，检测的类型，匹配物体的大小范围）
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA

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
            img_mouth_gray = gray_proceess(img_mouth)

            # seg_img = cv2.resize(seg_img, (256, 256))
            img_mouth = cv2.resize(img_mouth,(256,128))


            current_image = Image.fromarray(cv2image)  # 将图像转换成Image对象
            img = ImageTk.PhotoImage(image=current_image)

            # img_mouth_gray_tk = ImageTk.PhotoImage(image=img_mouth_gray)
            # label1 = tk.Label(top, image=img_mouth_gray_tk, width=256, height=128)
            # label1.grid(row=1, column=1)
            # img_mouth_tk = ImageTk.PhotoImage(image=img_mouth)
            # label2 = tk.Label(top, image=img_mouth_tk, width=256, height=128)
            # label2.grid(row=0, column=1)
            img_tk = ImageTk.PhotoImage(image=img)
            label3 = tk.Label(top, image=img_tk, width=256, height=256)
            label3.grid(row=0, column=0)
            label_text = fps_text
            labelHello = tk.Label(top, text=label_text, height=5, width=20, fg="blue")
            labelHello.grid(row=1, column=0)

            top.update()

top.mainloop()

