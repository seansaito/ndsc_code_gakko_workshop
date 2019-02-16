import colorsys
import logging
import os
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from keras import backend as K
from keras.models import load_model
from prettytable import PrettyTable

### CONSTANTS
repo_path = os.path.curdir
model_data_dir = os.path.join(repo_path, 'model_data')
model_path = os.path.join(model_data_dir, 'yolo.h5')
anchors_path = os.path.join(model_data_dir, 'yolo_anchors.txt')
classes_path = os.path.join(model_data_dir, 'coco_classes.txt')
font_path = os.path.join(repo_path, 'font/FiraMono-Medium.otf')
font_size = 14
img_path = os.path.join(repo_path, 'images/dog.jpg')
# This constant determines the minimum score required for a detection to be a considered
detection_score_threshold = 0.3
# This constant determines threshold for pruning away bounding box predictions that have
# overlap with previous selections
nms_iou_threshold = 0.45
# Thickness of bounding boxes
thickness = 4
image_shape = (416, 416)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


def yolo_head(feats, anchors, num_classes, input_shape):
    """
    Convert final layer features to bounding box parameters.
    """
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """
    Get corrected boxes according to the image's original shapes
    In other words, we scale the predicted bounding boxes to the input's original dimensions
    """
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

def yolo_predict(yolo_outputs,
                 anchors,
                 num_classes,
                 image_shape,
                 max_boxes=20,
                 score_threshold=.6,
                 iou_threshold=.5):
    """
    Get prediction from YOLO model on given input and return filtered boxes.
    """
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        # Get the boxes and the confidences for each box
        box_xy, box_wh, box_confidence, box_class_probs = yolo_head(yolo_outputs[l],
                                                                    anchors[anchor_mask[l]],
                                                                    num_classes,
                                                                    input_shape)
        _boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
        _boxes = K.reshape(_boxes, [-1, 4])
        _box_scores = box_confidence * box_class_probs
        _box_scores = K.reshape(_box_scores, [-1, num_classes])

        boxes.append(_boxes)
        box_scores.append(_box_scores)

    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        # TensorFlow function for non max suppression of detection candidates
        # This is for filtering overlapping bounding boxes
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_image(image, size):
    """
    Utility function resizing image to fit YOLO model. The function ensures that the aspect
    ratio is unchanged via padding
    """
    image_width, image_height = image.size
    to_width, to_height = size
    scale = min(to_width / image_width, to_height / image_height)
    new_width = int(image_width * scale)
    new_height = int(image_height * scale)

    image = image.resize((new_width, new_height), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((to_width - new_width) // 2, (to_height - new_height) // 2))
    return new_image


class YOLODetector(object):

    def __init__(self):
        self.model_path = model_path
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.build_model()

    def _get_class(self):
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def build_model(self):
        # Load model and weights
        self.yolo_model = load_model(model_path, compile=False)
        logger.info('Model loaded from {}.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # Fixed seed for consistent colors across runs
        np.random.seed(1234)
        # Shuffle colors to decorrelate adjacent classes
        np.random.shuffle(self.colors)
        # Reset seed to default
        np.random.seed(None)

        # Placeholder tensor for image shape
        self.input_image_shape = K.placeholder(shape=(2,))
        # Generate output tensor targets for filtered bounding boxes.
        boxes, scores, classes = yolo_predict(yolo_outputs=self.yolo_model.output,
                                              anchors=self.anchors,
                                              num_classes=len(self.class_names),
                                              image_shape=self.input_image_shape,
                                              score_threshold=detection_score_threshold,
                                              iou_threshold=nms_iou_threshold)
        return boxes, scores, classes

    def detect_image(self, image):
        # main function for object detection in an image
        start = timer()

        # Preprocess the image
        boxed_image = preprocess_image(image, image_shape)
        image_data = np.array(boxed_image, dtype='float32')
        logger.info("Preprocessed image looks like: {}".format(image_data.shape))
        # We normalize the image to a range of (0, 1)
        image_data /= 255.
        # Add a batch dimension to match the 4-D input tensor shape
        image_data = np.expand_dims(image_data, 0)

        # Run the model predictions to obtain the following:
        # predicted_boxes       bounding box predictions
        # predicted_scores      confidence of a particular class
        # predicted_classes     the label of the predicted class
        predicted_boxes, predicted_scores, predicted_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                # This flag is to tell the model that it's test mode
                K.learning_phase(): 0
            })

        logger.info('Found {} boxes for image'.format(len(predicted_boxes)))
        logger.info('Generating rectangles')

        # Prepare the prediction
        font = ImageFont.truetype(font=font_path, size=font_size)
        result_table = PrettyTable()
        result_table.field_names = ["Class", "Probability", "Bounding Box"]

        # Drawing time
        for index, class_label in reversed(list(enumerate(predicted_classes))):
            predicted_class = self.class_names[class_label]
            box = predicted_boxes[index]
            score = predicted_scores[index]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            result_table.add_row([predicted_class, '{:.3f}'.format(score),
                                  '({}, {}) ({}, {})'.format(left, top, right, bottom)])

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # A simple way of drawing thicker lines :)
            for offset in range(thickness):
                draw.rectangle(
                    [left + offset, top + offset, right - offset, bottom - offset],
                    outline=self.colors[class_label])

            # Draw the tiny rectangle in which we write the class text
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[class_label])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        logger.info('Time taken: {:.3f} seconds'.format(end - start))

        logger.info('Printing predictions:')
        print(result_table)

        return image


if __name__ == '__main__':
    image = Image.open(img_path)
    yolo_model = YOLODetector()
    detected_image = yolo_model.detect_image(image)
    detected_image.show()
