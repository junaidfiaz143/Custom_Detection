import os
import cv2
import numpy as np
import tensorflow as tf
import time
from PIL import ImageFont, ImageDraw, Image

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, 'frozen_inference_graph.pb')

NUM_CLASSES = 2 #PISTOL:1, NOT-PISTOL:2

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

video = cv2.VideoCapture("secret_video.mp4")

starting_time = time.time()
frame_id = 0

label = "label"
color = (0, 0, 0)

fontpath = "clan_med.ttf"     
font = ImageFont.truetype(fontpath, 24)

THRESHOLD = 0.95

while(True):
    ret, frame = video.read()

    frame_expanded = np.expand_dims(frame, axis=0)

    frame_id += 1
    height, width, channels = frame.shape

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    for index, score in enumerate(scores[0]):
        if score > THRESHOLD:

            ymin = boxes[0][index][0]
            xmin = boxes[0][index][1]

            ymax = boxes[0][index][2]
            xmax = boxes[0][index][3]
            (left, right, top, bottom) = (xmin * width, xmax * width, 
                                          ymin * height, ymax * height)
            
            left_top = (int(left), int(top))
            right_bottom = (int(right), int(bottom))

            if classes[0][index] == 1:
                label = "PISTOL" + " " + str(round(score * 100, 0)) + "%"
                color = (0, 0, 255)
            else:
                label = "NOT PISTOL" + " " + str(round(score * 100, 0)) + "%"
                color = (0, 255, 255)

            draw.text((int(left), int(top-24)), label, font = font, fill = color)
            draw.rectangle([(left_top), (right_bottom)], width = 4, outline = color)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time

    draw.text((10, 10),  "FPS@ " + str(round(fps, 2)), font = font, fill = (0, 255, 255))
    frame = np.array(img_pil)

    cv2.imshow('Weapon Detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()