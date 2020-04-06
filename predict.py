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
original_model = get_model(width =128, height = 64, model_name = 'lw_rf_net_unet', n_classes = n_classes)
# original_model = get_model(model_name = 'unet', n_classes = n_classes)
img_target_size = (original_model.input_width, original_model.input_height)
mask_target_size = (original_model.output_width, original_model.output_height)

def predict():
    model = load_model("intraoralArea.hdf5")
    testGene = testGenerator("data/intraoralArea/test_mini", 3, target_size = img_target_size)
    # testGene = testGenerator("data/intraoralArea/test_seg", 3)
    results = model.predict_generator(testGene, 3, verbose=1)

    saveResultForOtherModel("data/intraoralArea/test_mini", results, output_width = mask_target_size[0], output_height = mask_target_size[1], n_classes=n_classes)
    # saveResult("data/intraoralArea/test_seg", results)

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