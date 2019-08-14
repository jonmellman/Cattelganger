import dlib
import cv2
from imutils import face_utils
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
from pycatfd import catfd

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

image = cv2.imread('../human1.jpg')
rects = detector(image, 1)

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

# Cat features are
# chin, eyes, nose

FEATURES = ['jaw', 'left_eye', 'right_eye', 'nose']

def compute_feature_averages(shape):
	return np.concatenate([
		np.mean(shape[slice(*FACIAL_LANDMARKS_IDXS[feature])], axis=0, keepdims=True)
		for feature in FEATURES	
	], axis=0).astype(np.int32)
	

if (len(rects) > 1):
	raise Exception('Multiple faces detected!')

if (len(rects) < 1):
	raise Exception('No faces detected!')

rect = rects[0]


# determine the facial landmarks for the face region, then
# convert the facial landmark (x, y)-coordinates to a NumPy
# arraya
shape = predictor(image, rect)
shape = face_utils.shape_to_np(shape)

# convert dlib's rectangle to a OpenCV-style bounding box
# [i.e., (x, y, w, h)], then draw the face bounding box
(x, y, w, h) = face_utils.rect_to_bb(rect)
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the face number
cv2.putText(image, "Face #{}".format(1), (x - 10, y - 10),
	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# loop over the (x, y)-coordinates for the facial landmarks
# and draw them on the image
for (x, y) in shape:
	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

shape = compute_feature_averages(shape)

for (x, y) in shape:
	cv2.circle(image, (x, y), 5, (255, 0, 255), -1)
 
# show the output image with the face detections + facial landmarks
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()


# Need to scale all cat images to be same size
# Then scale human image to be same size as cat images before comparison

# For each cat image in dataset:
# 	Scale image to set size
# 	Run catfd.detect within bounding box
#   Store landmark points? Filesystem for now

# For human image:
#  Scale to set size
#  Run feature detector
#  Compare with all cat landmarks and return cat with minimum diff
	# tf.norm(human_features - cat_features)


cat_features = catfd.detect(input_image='../cat2.jpg', output_path=None, use_json=True, annotate_faces=False,
           annotate_landmarks=False, face_color=None, landmark_color=None, save_chip=False)

cat_landmarks = cat_features.landmarks