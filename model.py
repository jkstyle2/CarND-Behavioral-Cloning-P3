'''
todo :
1) batch normalization 이란. 이해하기
2) with open, as 구문
3) fit_generator() 활용법 이해하기. augmented data set에 새로운 batch size로 iteration하는거 같은데.. , yield개념 추
4) adam쓰면 learning rate는 고려 안하는건가본데 왜그런지?
'''

import csv
import cv2
import argparse
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from NvidiaModel import NvidiaModel
import utils
from nn import model

class Pipelines:

    # Initialize
    def __init__(self, model=None, base_directory='', epochs=2):
        self.data = []  # store lines read in from 'driving_log.csv'
        self.model = model  # model that we want to use
        self.epochs = epochs  # number of epochs we want our model to train for
        self.batch_size = 128  # batch size that I want to use
        self.training_samples = []  # store the result of running 'train_test_split' on data stored in self.data
        self.validation_samples = []  # store the result of running 'train_test_split' on data stored in self.data
        self.steering_recovery_factor = 0.2  # steering angle adjustment for left/right camera images during data augmentation
        self.data_directory = base_directory  # holds the root directory of where the images and driving log are stored
        self.image_directory = self.data_directory + '/IMG/'  # helper variables holding the paths to the images folder
        self.driving_log_directory = self.data_directory + '/driving_log.csv'  # helper variables holding the paths to the driving log folder

    # Import Data into the Pipelines
    def import_data(self, skipHeader=False):

        with open(self.driving_log_directory) as csvfile:  # open the driving_log.csv file
            reader = csv.reader(csvfile)

            if skipHeader:
                next(reader)  # skip the column names at the first row

            for line in reader:
                self.data.append(line)  # reading each row into self.data. Each row contains absolute paths to camera images and steering angle

        return None

    # Split Data into Training and Validation Sets
    def split_data(self, split_ratio=0.2):
        '''
        Given the rows of data from driving_log.csv stored in self.data,
        the data is then split into a training and validation sets
        '''
        train, validation = train_test_split(self.data, test_size=split_ratio)

        # assign each set to the instance variables
        self.training_samples = train
        self.validation_samples = validation

        return None


    # Data Generator
    def data_generator(self, samples, batch_size=128):
        '''
        To reiterate, generator is used to avoid loading all of the augmented images into memory.
        Instead, it creates a new batch of augmented data only when it is called.
        '''

        num_samples = len(samples)

        while True:
            shuffle(samples)

            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset: offset+batch_size]

                images = []
                steering_angles = []
                for batch_sample in batch_samples:
                    augmented_images, augmented_angles = self.process_batch(batch_sample)
                    images.extend(augmented_images)  # Note the difference between 'extend' and 'append'
                    steering_angles.extend(augmented_angles)

                X_train = np.array(images)
                y_train = np.array(steering_angles)

                yield shuffle(X_train, y_train) # when returns, it restarts after yield while memorizing the last data.


    # Data Augmentation
    def process_batch(self, batch_sample):  # batch_sample: a list containing the paths to the center/left/right images and steering angle.
        '''
        Samples in each batch are processed.
        The center, left, right, and horizontally flipped images are used and processed.
        '''

        # New augmented images and steering angles
        images = []
        steering_angles = []

        # original measured steering_angle
        steering_angle = np.float32(batch_sample[3])  # 4th column in csv

        for image_directory_index in range(3):
            image_file_name = batch_sample[image_directory_index].split('/')[-1]  # split using '/' and [-1] indicates the last element in the list

            # addition from center/left/right images
            image = cv2.imread(self.image_directory + image_file_name)  # opencv reads images in BGR color
            cropped_image = utils.crop_img(image)
            blurred_image = utils.blur_img(cropped_image)
            resized_image = utils.resize_img(blurred_image)
            yuv_image = utils.bgr2yuv(resized_image)
            images.append(yuv_image)

            # # rgb_image = utils.bgr2rgb(image)  # bgr to rgb
            # # blurred_image = utils.blur_img(rgb_image)  # gaussian blurring
            # blurred_image = utils.blur_img(image)  # gaussian blurring
            # cropped_resized = utils.crop_and_resize(blurred_image)  # crop and resize (the input size is reshaped to (70, 160, 3))
            # # yuv_image = utils.rgb2yuv(cropped_resized)  # rgb to yuv (as nVidia suggested)
            # yuv_image = utils.bgr2yuv(cropped_resized)  # rgb to yuv (as nVidia suggested)
            # images.append(yuv_image)
            # # images.append(cropped_resized)

            # steer angle correction on left/right images to balance the dataset with non-zero steering angles
            if image_directory_index == 1:  # left camera
                steering_angles.append(steering_angle + self.steering_recovery_factor)

            elif image_directory_index == 2:  # right camera
                steering_angles.append(steering_angle - self.steering_recovery_factor)

            else:  # center camera
                steering_angles.append(steering_angle)

            # if the magnitude of steering angle is bigger than 0.2, flipped image of center is added.
            # if image_directory_index == 0 and abs(steering_angle) > 0.2:
            if image_directory_index == 0:
                images.append(utils.flip_img(yuv_image))
                # images.append(utils.flip_img(cropped_resized))
                steering_angles.append(steering_angle * (-1.0))

        return images, steering_angles


    def run(self):

        self.import_data(skipHeader=True)
        self.split_data()

        self.model.compile(loss='mse', optimizer='adam')  # learning process configuration
        self.model.fit_generator(generator = self.data_generator(samples=self.training_samples, batch_size=self.batch_size),
                                 steps_per_epoch = len(self.training_samples) * 2 // self.batch_size, # (the number of samples/batch_size)
                                 validation_data = self.data_generator(samples=self.validation_samples, batch_size=self.batch_size),
                                 validation_steps = len(self.validation_samples) // self.batch_size,
                                 epochs = self.epochs)

        self.model.save('model.h5')


def main():
    parser = argparse.ArgumentParser(description='Train a car to drive itself')
    parser.add_argument(
        '--data-base-directory',
        type=str,
        default='./data',
        help='Path to image directory and driving log')

    args = parser.parse_args()

    # Instantiate the pipelines
    pipeline = Pipelines(model=NvidiaModel(), base_directory=args.data_base_directory, epochs=10)

    # Feed driving log data into the pipelines
    # pipeline.import_data(skipHeader=True)

    # Start training
    pipeline.run()


if __name__ == '__main__':
    main()

