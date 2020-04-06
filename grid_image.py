from tkinter import *
import time

tk=Tk()
var=IntVar()

# #标签控件，显示文本和位图，展示在第一行
# Label(tk,text="First").grid(row=0,sticky=E)#靠右
# Label(tk,text="Second").grid(row=1,sticky=W)#第二行，靠左
#
# #输入控件
# Entry(tk).grid(row=0,column=1)
# Entry(tk).grid(row=1,column=1)

#插入图片
photo=PhotoImage(file="mouth.png")
label=Label(image=photo)
label.image=photo
# label.grid(row=0,column=1,rowspan=2,columnspan=2,sticky=W+E+N+S, padx=5, pady=5) #合并两行，两列，居中，四周外延5个长度
label.grid(row=0,column=1)

label_text = Label(text="fps=5")
label_text.grid(row=1,column=1)

photo_mouth=PhotoImage(file="mouth.png")
label_mouth=Label(image=photo_mouth)
label_mouth.image=photo_mouth
label_mouth.grid(row=0,column=2,sticky=W+E+N+S, padx=5, pady=5)

photo_mouth_after=PhotoImage(file="mouth.png")
label_mouth_after=Label(image=photo_mouth_after)
label_mouth_after.image=photo_mouth_after
label_mouth_after.grid(row=1,column=2,sticky=W+E+N+S, padx=5, pady=5)

label_text.configure(text="fps=10")

#主事件循环
mainloop()



