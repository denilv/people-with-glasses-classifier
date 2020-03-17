import cv2
import dlib
import numpy as np
from PIL import ExifTags, Image
from PIL.ImageOps import exif_transpose

face_detector = dlib.get_frontal_face_detector()

def find_face(img):
    rectangles, scores, detector_num = face_detector.run(img, 0)
    if len(rectangles) == 0:
        # no faces were found
        # try with another sampling method
        rectangles, scores, detector_num = face_detector.run(img, 1)
        if len(rectangles) == 0:
            return None
    rect = rectangles[0]
    bbox = [
        max(0, rect.top()), 
        max(0, rect.bottom()), 
        max(0, rect.left()), 
        max(0, rect.right())
    ]
    return bbox


def crop_img(img):
    bbox = find_face(img)
    if bbox is None:
        return img
    top, bottom, left, right = bbox
    return img[top: bottom, left: right]


def resize_shortest_edge(img, width, height):
    dim = None
    (h, w) = img.shape[:2]

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(img, dim)
    return resized


def open_img(fp):
    pil_img = Image.open(fp)
    try:
        pil_img =  exif_transpose(pil_img)
    except:
        pass
    return np.array(pil_img)
