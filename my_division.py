from tkinter import *
import cv2
import os
from PIL import Image, ImageTk
from tkinter import ttk

class APP:
    def __init__(self):
        self.camera = None  # 摄像头
        self.root = Tk()
        self.root.title('FACE')
        self.root.geometry('%dx%d' % (1600, 600))
        self.createFirstPage()
        mainloop()

    def createFirstPage(self):
        self.page1 = Frame(self.root)
        self.page1.pack()
        Label(self.page1, text='欢迎使用人脸识别系统', font=('粗体', 20)).pack()
        image = Image.open("/home/shike/A_graduationProject/introralArea_seg_project2/IntraoralArea_seg_project/IntraoralArea_project/mouth.png")  # 随便使用一张图片 不要太大
        photo = ImageTk.PhotoImage(image=image)
        self.data1 = Label(self.page1, width=780, image=photo)
        self.data1.image = photo
        self.data1.pack(padx=5, pady=5)

        self.button12 = Button(self.page1, width=18, height=2, text="录入新的人脸", bg='green', font=("宋", 12),
                               relief='raise', command=self.createSecondPage)
        self.button12.pack(side=LEFT, padx=25, pady=10)

    def createSecondPage(self):
        self.camera = cv2.VideoCapture(0)
        self.page1.pack_forget()
        self.page2 = Frame(self.root)
        self.page2.pack()
        Label(self.page2, text='欢迎使用人脸识别系统', font=('粗体', 20)).pack()
        self.data2 = Label(self.page2)
        self.data2.pack(padx=5, pady=5)

        self.button21 = Button(self.page2, width=18, height=2, text="返回", bg='gray', font=("宋", 12),
                               relief='raise', command=self.backFirst)
        self.button21.pack(padx=25, pady=10)

        self.video_loop(self.data2)

    def video_loop(self, panela):
        success, img = self.camera.read()  # 从摄像头读取照片
        if success:
            cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
            current_image = Image.fromarray(cv2image)  # 将图像转换成Image对象
            imgtk = ImageTk.PhotoImage(image=current_image)
            panela.imgtk = imgtk
            panela.config(image=imgtk)
            self.root.after(1, lambda: self.video_loop(panela))

    def backFirst(self):
        self.page2.pack_forget()
        self.page1.pack()
        # 释放摄像头资源
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    demo = APP()

