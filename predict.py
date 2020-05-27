from __future__ import division

from data import *
from model import *
from keras.models import load_model
import time
import os
import cv2

from IntraoralArea_project.data import testGeneratorForImg, saveResultForOtherModel, testGenerator, \
    getResultForSingleImg
from IntraoralArea_project.model import get_model

n_classes = 2
original_model = get_model(width =256, height = 128, model_name = 'unet_mini', n_classes = n_classes)
# original_model = get_model(model_name = 'unet', n_classes = n_classes)
img_target_size = (original_model.input_width, original_model.input_height)
mask_target_size = (original_model.output_width, original_model.output_height)

def predict():
    model = load_model("intraoralArea.hdf5")
    test_path = "data/intraoralArea/test_20_0.5"
    predict_img_num = 23
    testGene = testGenerator(test_path, predict_img_num, target_size = img_target_size)
    # testGene = testGenerator("data/intraoralArea/test_seg", 3)
    results = model.predict_generator(testGene, predict_img_num, verbose=1)

    saveResultForOtherModel(test_path, results, output_width = mask_target_size[0], output_height = mask_target_size[1], n_classes=n_classes)
    # saveResult("data/intraoralArea/test_seg", results)
    print("预测完毕，计算准确率：")
    pa_sum=0
    mpa_sum = 0
    iou_sum = 0
    for i in range(predict_img_num):
        img_result = cv2.imread(os.path.join(test_path, "%d_predict.png" % i), cv2.IMREAD_COLOR)
        # img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img_result = cv2.resize(img_result, img_target_size)

        img_origin = cv2.imread(os.path.join(test_path, "%d_origin.png" % i), cv2.IMREAD_COLOR)
        # img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img_origin = cv2.resize(img_origin, img_target_size)

        pa = calPa(img_origin,img_result)
        print(str(i)+"--pa---"+str(pa))
        pa_sum+=pa

        mpa = calMPa(img_origin,img_result)
        print(str(i)+"--mpa---"+str(mpa))
        mpa_sum+=mpa

        iou = calIoU(img_origin,img_result)
        print(str(i)+"--iou---"+str(iou))
        iou_sum+=iou
    print("pa_sum="+str(pa_sum)+"---pa_equal="+str(pa_sum/predict_img_num))
    print("mpa_sum=" + str(mpa_sum) + "---mpa_equal=" + str(mpa_sum / predict_img_num))
    print("iou_sum=" + str(iou_sum) + "---iou_equal=" + str(iou_sum / predict_img_num))

def calPa(img_origin,img_result):
    img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
    img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
    i=1
    equal=0
    for x in range(mask_target_size[1]):  # 图片的高
        for y in range(mask_target_size[0]):  # 图片的宽
            px_origin = img_origin[x, y]
            px_result = img_result[x, y]
            if(px_origin==px_result):
                equal+=1
            i+=1

    print("equal="+str(equal)+", i="+str(i))
    simi = equal/i
    return simi

def calMPa(img_origin,img_result):
    img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
    img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
    i1 = 1
    i2 = 1
    equal1 = 0
    equal2 = 0
    for x in range(mask_target_size[1]):
        for y in range(mask_target_size[0]):
            px_origin = img_origin[x, y]
            px_result = img_result[x, y]
            if(px_origin==0):
                if(px_result==0):
                    equal1 += 1
                i1+=1
            if(px_origin>1):
                if(px_result>1):
                    equal2 += 1
                i2 += 1
    pa1 = equal1/i1
    pa2 = equal2/i2
    mpa = pa1/2 + pa2/2
    print("equal1=" + str(equal1) + ", i1=" + str(i1))
    print("equal2=" + str(equal2) + ", i2=" + str(i2))
    return mpa

def calIoU(img_origin,img_result):
    img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
    img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
    i = 1
    equal = 0
    for x in range(mask_target_size[1]):
        for y in range(mask_target_size[0]):
            px_origin = img_origin[x, y]
            px_result = img_result[x, y]
            if(px_origin==px_result):
                equal+=1
            i+=1

    i = i*2 - equal
    print("equal=" + str(equal) + ", i=" + str(i))
    simi = equal / i
    return simi




def getPredictValueByImg(img, model):
    #t1 = time.time()
    testGene = testGeneratorForImg(img, as_gray = False, target_size = img_target_size)
    #t2 = time.time()
    ##verbose: 日志显示模式，0 或 1。
    results = model.predict_generator(testGene, 1, verbose=0)
    #t3 = time.time()
    #print("t2-t1=" + str(t2 - t1) + "t3-t2=" + str(t3 - t2))
    return getResultForSingleImg(results[0], output_width = mask_target_size[0], output_height = mask_target_size[1], n_classes = n_classes)




if __name__=='__main__':
    predict()

    # test_path = "data/intraoralArea/test_20_0.5"
    # img_result = cv2.imread(os.path.join(test_path, "0_predict.png"), cv2.IMREAD_COLOR)
    # img_result = cv2.resize(img_result, img_target_size)
    #
    # img_origin = cv2.imread(os.path.join(test_path, "0_origin.png"), cv2.IMREAD_COLOR)
    # img_origin = cv2.resize(img_origin, img_target_size)
    #
    # simi= calMPa(img_origin,img_result)
    # print("simi="+str(simi))