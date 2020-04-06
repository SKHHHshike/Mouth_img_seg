import os
import glob
import cv2

def togrey(img,outdir):
    src = cv2.imread(img)
    try:
        dst = cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
        cv2.imwrite(os.path.join(outdir,os.path.basename(img)), dst)
    except Exception as e:
        print(e)

# for file in glob.glob('D:/ÎÄŒþŒÐÂ·Ÿ¶/test24/*.png'):
#     togrey(file,'D:/ÎÄŒþŒÐÂ·Ÿ¶/test8/')

for file in glob.glob('/home/shike/A_graduationProject/introralArea_seg_project2/IntraoralArea_seg_project/IntraoralArea_project/data/intraoralArea/train_seg/label/*.png'):
    togrey(file, '/home/shike/A_graduationProject/introralArea_seg_project2/IntraoralArea_seg_project/IntraoralArea_project/data/intraoralArea/train_seg_eight_code/label/')