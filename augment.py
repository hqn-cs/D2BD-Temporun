import argparse
import matplotlib.pyplot as plt
from pkg_resources import require 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_data(data_path, _save_path, cls):

    generator = ImageDataGenerator( 
        rescale=1./255,
        
        #shear_range=0.2,
        zoom_range=0.6,
        rotation_range = 10,
        brightness_range = (0.1, .8),
        #vertical_flip = True,
        #horizontal_flip=True,
        data_format = "channels_last"
    )

    (x,_) = generator.flow_from_directory(
        directory=data_path,
        target_size = (28,28),
        color_mode = 'rgb',
        classes = [str(cls)],
        save_to_dir = _save_path,
        save_format = "jpeg",
        batch_size=128
    )

if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("--data_path", required = True)
    parse.add_argument("--save_path", required = True)
    parse.add_argument("--cls")
    parser = parse.parse_args()

    print("Start processing data..")
    augment_data(parser.data_path, parser.save_path, parser.cls)
    print("End!")



