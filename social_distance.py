import numpy as np
import cv2
import math
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from object_detection.utils import ops as utils_ops

# User defined variables
loc_model = 'efficientdet_d7_coco17_tpu-32/saved_model'     # location of the inference graph, taken from tensorflow zoo
video_path = 'soccer_test.avi'             # Video we wish to run the physical distancing model on
save_as = 'soccer_test_detected.avi'       # The name we want to save it under
score_min = 0.3         # This is the minimum score for detection of humans.
dist_min = 25           # If pixel distance is smaller than dist_min between 2 people, we display "danger" and red block
dist_danger = 40        # Between pixel distance dist_min and dist_danger, we display "caution" and orange block
mult_max = [1, 1]       # Perspective multiplier, based on y coordinate of location in the screen.


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


# Setting up the model
print("Setting up the model")
tf.keras.backend.clear_session()
model = tf.saved_model.load(loc_model)

# Reading all of the frames from the video, so we can run detection on it
print("Reading all the frames from the video")
# Read all the frames from the video
vidcap = cv2.VideoCapture(video_path)
success, image = vidcap.read()
all_img = []
while success:
    all_img.append(image)
    success, image = vidcap.read()
    print("Number of images we have gotten from video : {}, ".format(len(all_img)))


print("\nDETECTION PHASE \n")
# run each frame from the video through the model
img_detected = []
count = 0
for img in all_img:
    output_dict = run_inference_for_single_image(model, img)
    im = Image.fromarray(img)
    all_blocks = []
    for kn in range(0, len(output_dict['detection_scores'])):
        scores = output_dict['detection_scores'][kn]
        if scores > score_min and output_dict['detection_classes'][kn] == 1:
            all_blocks.append([list(output_dict['detection_boxes'][kn]), output_dict['detection_classes'][kn], scores])

    # parse all_found_lights in order to draw our stuff
    font = ImageFont.truetype("arial.ttf", 15)  # the font to draw the label and score on the image
    im_draw = ImageDraw.Draw(im)
    im_width = im.size[0]
    im_height = im.size[1]

    # find the center for each
    all_boxes_to_draw = []      # [[x1,y1,x2,y2,x3,y3, score]]
    for item in all_blocks:     # find all boxes and their coordinates
        width = abs(item[0][1]-item[0][3])
        height = abs(item[0][0]-item[0][2])
        if width < 0.3 and height < 0.4:
            box_to_draw = [int(item[0][1] * im_width), int(item[0][0] * im_height), int(item[0][3] * im_width),
                           int(item[0][2] * im_height),
                           int(item[0][1] * im_width) + int((int(item[0][3] * im_width)-int(item[0][1] * im_width))/2),
                           int(item[0][0] * im_height) + int((int(item[0][2] * im_height)-int(item[0][0] * im_height))/2),
                           item[2]]
                           # outsides of the bounding box
            all_boxes_to_draw.append([box_to_draw])

    todraw = []
    all_blocked_lines = []
    for item1 in all_boxes_to_draw:
        c_draw = 'green'
        item1[0][6] = ''
        for item2 in all_boxes_to_draw:
            if item1 is not item2:
                # find out what the dist_min and dist_danger are, based on perspective
                mult_cur = mult_max[0]+((min(item1[0][5],item2[0][5])+abs(item1[0][5]-item2[0][5])/2)/im_height)*(mult_max[1]-mult_max[0])
                if math.sqrt(math.pow(item1[0][4]-item2[0][4],2)+math.pow(item1[0][5]-item2[0][5], 2)) < dist_min*mult_cur:
                    c_draw = 'red'
                    all_blocked_lines.append([item1[0][4], item1[0][5], item2[0][4], item2[0][5], 'red'])
                    item1[0][6] = 'Danger'
                elif math.sqrt(math.pow(item1[0][4] - item2[0][4], 2) + math.pow(item1[0][5] - item2[0][5], 2)) < dist_danger*mult_cur:
                    if c_draw == 'green':
                        c_draw = 'orange'
                        item1[0][6] = 'Caution'
                    all_blocked_lines.append([item1[0][4], item1[0][5], item2[0][4], item2[0][5], 'orange'])

        todraw.append([item1[0], c_draw])

    for item in todraw:
        box_to_draw = [item[0][0], item[0][1], item[0][2], item[0][3]]
        center_to_draw = [item[0][4], item[0][5]]
        color_to_draw = item[1]
        score_to_draw = item[0][6]
        im_draw.rectangle(box_to_draw, outline=color_to_draw, width=2)
        im_draw.text((box_to_draw[0], box_to_draw[1] - 15), str(score_to_draw), fill=color_to_draw, align='left', font=font)
        im_draw.ellipse((center_to_draw[0]-5, center_to_draw[1]-5, center_to_draw[0]+5, center_to_draw[1]+5), fill=color_to_draw)

    # draw the lines of violations
    for item in all_blocked_lines:
        line = [item[0], item[1], item[2], item[3]]
        if item[4] == 'red':
            lw = 4                      # thickness of the red line
        elif item[4] == 'orange':
            lw = 3                      # thickness of the orange line
        im_draw.line(line, item[4], lw)

    # Append to the detected instances
    img_detected.append(im)
    count += 1
    print("Number of images detected : {}, {}% done".format(count, round(count/len(all_img)*100)))

print("\nSAVE ALL IMAGES TO VIDEO\n")
size = (im_width, im_height)
out = cv2.VideoWriter(save_as, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
for i in range(len(img_detected)):
    open_cv_image = np.array(img_detected[i])
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    out.write(open_cv_image)
out.release()

