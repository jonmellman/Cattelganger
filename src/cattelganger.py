from __future__ import division
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers, optimizers, metrics, Sequential
import cv2
from .facenet import FaceNet
from .datasets import create_datasets_from_tfrecord
import matplotlib.pyplot as plt

tf.enable_eager_execution()

# TODO: Make the human file location dynamic, and eventually read from camera.
HUMAN_FACE_FILE_LOCATION = './vggface2/val/15169058.jpg'

# TODO: Use a larger database instead of files.
CAT_FACE_FILE_LOCATIONS = ['../cat2.jpg', '../cat3.jpg']


parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str, default='Resnet50')
parser.add_argument('--image-size', type=int, default=160)
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--embedding_size', type=int, default=512)
opt = parser.parse_args()
facenet = FaceNet(opt)


def _crop_to_face(image, cascade_path):
    SIZE = 160

    face_cascade = cv2.CascadeClassifier(cascade_path)

    bounding_boxes = face_cascade.detectMultiScale(image, 1.25, 6)

    if (len(bounding_boxes)) == 0:
        raise Exception('No faces detected in input image!')

    x, y, w, h = bounding_boxes[0]

    crop_img = image[y:y+h, x:x+w]
    resized_image = cv2.resize(crop_img, (SIZE, SIZE))

    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    return image_rgb


def crop_to_human_face(image):
    return _crop_to_face(image, '../haarcascade_frontalface_default.xml')


def crop_to_cat_face(image):
    return _crop_to_face(image, '../haarcascade_frontalcatface.xml')


def show_cat(index):
    cat_photo_file = '../cat' + str((index.numpy()) + 1) + '.jpg'

    cat_photo = cv2.imread(cat_photo_file)
    cat_photo = cv2.cvtColor(cat_photo, cv2.COLOR_BGR2RGB)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cat_photo)
    plt.show()


human_face = tf.image.convert_image_dtype(crop_to_human_face(
    cv2.imread(HUMAN_FACE_FILE_LOCATION)), tf.float32)


cat_faces = tf.stack([
    # TODO: Refactor to use list comprehension
    tf.image.convert_image_dtype(crop_to_cat_face(cv2.imread(CAT_FACE_FILE_LOCATIONS[0])), tf.float32),
    tf.image.convert_image_dtype(crop_to_cat_face(cv2.imread(CAT_FACE_FILE_LOCATIONS[1])), tf.float32)
], axis=0)

human_face_prediction = facenet.model(human_face[None, ...])
cat_face_predictions = facenet.model(cat_faces)
distances = tf.norm(human_face_prediction -
                    cat_face_predictions, axis=1, ord=2)
cat_index = tf.argmin(distances)

show_cat(cat_index)
