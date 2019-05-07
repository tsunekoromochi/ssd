import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import numpy as np
from scipy.misc import imread
import tensorflow as tf

from ssd import SSD300
from ssd_utils import BBoxUtility

def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def mosaic_area(src, x_min, y_min, x_max, y_max, ratio=0.1):
    dst = src.copy()
    dst[y_min:y_max, x_min:x_max] = mosaic(dst[y_min:y_max, x_min:x_max], ratio)
    return dst

np.set_printoptions(suppress=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))


voc_classes = ['face', 'hand']
NUM_CLASSES = len(voc_classes) + 1

input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
# model.load_weights('weights_SSD300.hdf5', by_name=True)
model.load_weights('./checkpoints/weights.99-2.78.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)

vid = cv2.VideoCapture('./free_data/lonestartx_free.mp4')
if not vid.isOpened():
    raise IOError(("Couldn't open video file or webcam. If you're "
    "trying to open a webcam, make sure you video_path is an integer!"))

vidw = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
vidh = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = vid.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('./free_data/output.avi', int(fourcc), fps, (int(vidw), int(vidh)))

while True:
# for i in range(10):  # for test
    retval, img = vid.read()
    if not retval:
        print("Done!")
        break

    inputs = []
    images = []
    images.append(img)
    img_org = img
    img = cv2.resize(img, (300, 300))
    img = image.img_to_array(img)
    inputs.append(img.copy())
    inputs = preprocess_input(np.array(inputs))

    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)

    for i, img in enumerate(images):
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.1]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))

            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = voc_classes[label - 1]
            # print(label_name)
            # cv2.rectangle(img_org, (xmin, ymin), (xmax+1, ymax+1), (0, 0, 255), 1)  # red box
            img_org = mosaic_area(img_org, xmin, ymin, xmax+1, ymax+1, ratio=0.1) # mosaic

    out.write(img_org)

vid.release()
out.release()

