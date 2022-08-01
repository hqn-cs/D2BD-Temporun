import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_path: str, img_W: int = 28, img_H: int = 28, batch_size: int = 128):
    
    dataloder = tf.keras.preprocessing.image_dataset_from_directory(
                    data_path,
                    labels="inferred",
                    label_mode="int", 
                    color_mode="rgb",
                    batch_size=batch_size,
                    image_size=(img_H, img_W),
                    shuffle=True,
                    seed=1111)
                    
    return dataloder

def load_ImageGen(data_path, test, h=28, w=28, batch_size=32): 
    if not test:
        data_generator =  ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range = 90,
            brightness_range = (0.1, .8),
            vertical_flip = True,
            horizontal_flip=True
            
        )
    
    else:
        data_generator = ImageDataGenerator(rescale=1./255)


    if not test:
        return data_generator.flow_from_directory(data_path, 
                                                class_mode="categorical",
                                                target_size= (h,w),
                                                batch_size = batch_size,
                                            )
    else:
        return data_generator.flow_from_directory(data_path,
                                                    class_mode = "categorical",
                                                    batch_size = batch_size
                                                    )
def plot_accuracy(history, path_save):
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'{path_save}/accuracy.png')


def plot_loss(history, path_save):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'{path_save}/loss.png')


