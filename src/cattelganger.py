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


parser = argparse.ArgumentParser()
parser.add_argument('--model-name',type=str, default='Resnet50')
parser.add_argument('--image-size',type=int, default=160)
parser.add_argument('--num_classes',type=int, default=100)
parser.add_argument('--embedding_size',type=int, default=512)
opt = parser.parse_args()
facenet = FaceNet(opt)


def _crop_to_face(image, cascade_path):
	SIZE = 160
	
	face_cascade = cv2.CascadeClassifier(cascade_path)

	bounding_boxes = face_cascade.detectMultiScale(image, 1.25, 6)

	if (len(bounding_boxes)) == 0:
		raise Exception('No faces detected in input image!')

	x, y, w, h = bounding_boxes[0]

	# w = max(SIZE, w)
	# h = max(SIZE, h)

	# x = x - ((SIZE - w2) / 2)
	# y = y - ((SIZE - h2) / 2)

	
	crop_img = image[y:y+h, x:x+w]
	resized_image = cv2.resize(crop_img, (SIZE, SIZE)) 

	image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

	# fig, ax = plt.subplots(1)
	# ax.imshow(image_rgb)

	# plt.xticks([])
	# plt.yticks([])
	# plt.show()

	return image_rgb


def crop_to_human_face(image):
	return _crop_to_face(image, '../haarcascade_frontalface_default.xml')

def crop_to_cat_face(image):
	return _crop_to_face(image, '../haarcascade_frontalcatface.xml')


# human_face = crop_to_human_face(cv2.imread('./vggface2/val/15169058.jpg'))
cat_face = crop_to_cat_face(cv2.imread('../cat3.jpg'))
prediction = facenet.model(cat_face[None, ...])
print(prediction)

# for batch_id, (batch_images_validate, batch_labels_validate) in enumerate(val_datasets):
# 	# batch_images_validate: [1, 160, 160, 3], dtype=float32
# 	import pdb; pdb.set_trace()
# 	prediction = facenet.model(batch_images_validate)
# 	print(prediction)