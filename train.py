from model import *
from model2 import *
from data import *
from keras.callbacks import ModelCheckpoint

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from IntraoralArea_project.data import trainGenerator
from IntraoralArea_project.model import get_model

n_classes = 2
#设定输入图片的width，height。设定使用的模型类别，设定分类的个数。
model = get_model(width = 128, height = 64,  model_name = 'unet', n_classes = n_classes)
# model = unet()

def train():
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')

    img_target_size = (model.input_height,model.input_width)
    mask_target_size = (model.output_height,model.output_width)
    myGene = trainGenerator(2,'data/intraoralArea/train_252','image','label',
                            data_gen_args, n_classes=n_classes,target_size = img_target_size,
                            mask_target_size = mask_target_size, save_to_dir = None)

    model_checkpoint = ModelCheckpoint('intraoralArea.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene, steps_per_epoch=150, epochs=1, callbacks=[model_checkpoint])

if __name__=='__main__':
    train()