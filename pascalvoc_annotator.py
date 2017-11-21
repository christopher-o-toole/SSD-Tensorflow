#!/usr/bin/env python3.5
############################################
# Multi-Rotor Robot Design Team
# Missouri University of Science and Technology
# Fall 2017
# Christopher O'Toole

import cv2
import numpy as np
import os
import argparse

from datasets.generate_pascalvoc_annotations import generate_pascvalvoc_annotation_from_image_file, ANNOTATION_DEFAULT_DIR
import xml.etree.ElementTree as ET

DEFAULT_WINDOW_NAME = 'Pascal VOC Annotator'
PASCAL_VOC_IMAGE_EXTENSION = '.jpg'

RECT_COLOR = (0, 255, 0)
RECT_THICKNESS = 2

QUIT_KEY = 'q'
CLEAR_KEY = 'c'
NEXT_KEY = 'n'
PREV_KEY = 'p'
UNDO_KEY = 'u'

def _parse_pascalvoc_annoation(path):
    tree = ET.parse(path)
    root = tree.getroot()

    bboxes = []
    labels = []

    for obj in root.findall('object'):
        labels.append(obj.find('name').text)

        bbox = obj.find('bndbox')
        bboxes.append((int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text), int(bbox.find('ymax').text)))
    
    return bboxes, labels


class PascalVOCAnnotator():
    def __init__(self, folder, obj_label, name=DEFAULT_WINDOW_NAME, window_type=cv2.WINDOW_AUTOSIZE):
        assert os.path.isdir(folder), 'PascalVOCAnnotator(): %s is not a directory' % (folder,)
        self._name = name
        self._type = window_type
        self._folder = folder
        self._img_paths = [os.path.join(folder, filename) for filename in os.listdir(folder) if PASCAL_VOC_IMAGE_EXTENSION in filename]
        assert self._img_paths, 'PascalVOCAnnotator(): no JPEG images were found in directory %s' % (folder,)
        self._cur_img = None
        self._cur_bboxes = []
        self._cur_labels = []
        self._changed = False
        self._bbbox_in_progress = []
        self._annotations = {}
        self._load_saved_annotations()
        self._obj_label = obj_label
        self.index = 0
    
    def _load_saved_annotations(self):
        if os.path.isdir(self.annotation_dir):
            for filename in os.listdir(self.annotation_dir):
                key = os.path.join(self._folder, os.path.basename(os.path.splitext(filename)[0]) + PASCAL_VOC_IMAGE_EXTENSION)
                self._annotations[key] = _parse_pascalvoc_annoation(os.path.join(self.annotation_dir, filename))


    def _next_image(self):
        self.index = self.index + 1
    
    def _prev_image(self):
        self.index = self.index - 1

    def _remove_annotation_file_when_empty(self):
        if hasattr(self, '_index'):
            annotation_file_path = os.path.join(self.annotation_dir, os.path.basename(os.path.splitext(self.cur_img_path)[0]) + '.xml')
            if not self._cur_bboxes or not self._cur_labels and os.path.isfile(annotation_file_path):
                os.remove(annotation_file_path)

    def _undo(self):
        if not self._bbbox_in_progress and self._cur_bboxes and self._cur_labels:
            self._changed = True
            del self._cur_bboxes[-1]
            del self._cur_labels[-1]
        

    def _clear(self):
        self._cur_img = None
        self._changed = True
        
        if self.cur_img_path in self._annotations:
            del self._annotations[self.cur_img_path]
        
        self.index = self.index

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._bbbox_in_progress = [x, y]*2
        elif event == cv2.EVENT_MOUSEMOVE and self._bbbox_in_progress:
            self._bbbox_in_progress[2] = x
            self._bbbox_in_progress[3] = y
        elif event == cv2.EVENT_LBUTTONUP and self._bbbox_in_progress:
            self._changed = True
            self._cur_bboxes.append(self._bbbox_in_progress)
            self._cur_labels.append(self._obj_label)
            self._bbbox_in_progress = []

    def _get_key(self):
        return chr(cv2.waitKey(1) & 0xFF)

    def _draw_rect(self, img, coords, color=RECT_COLOR, thickness=RECT_THICKNESS):
        xmin, ymin, xmax, ymax = coords
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    @property
    def index(self):
        return self._index

    @property
    def annotation_dir(self):
        return os.path.join(self._folder, ANNOTATION_DEFAULT_DIR)

    @index.setter
    def index(self, value):
        if self._cur_img is not None and self._cur_bboxes and self._changed:
            assert len(self._cur_labels) == len(self._cur_bboxes), 'PascalVocAnnotator(): label and bounding box collection sizes are mistmatched'
            generate_pascvalvoc_annotation_from_image_file(self.cur_img_path, self._cur_labels, self._cur_bboxes, self.annotation_dir, True)

        self._remove_annotation_file_when_empty()

        self._index = value % len(self._img_paths)
        self._cur_img = cv2.imread(self.cur_img_path)

        if not self.cur_img_path in self._annotations:
            self._annotations[self.cur_img_path] = ([], [])
        
        self._cur_bboxes, self._cur_labels = self._annotations[self.cur_img_path]
        assert self._cur_img is not None, 'PascalVocAnnotator(): could not read image at %s' % (self.cur_img_path,)
        self._changed = False

    @property
    def cur_img(self):
        return self._cur_img

    @property
    def cur_img_path(self):
        return self._img_paths[self._index]

    def update(self):
        should_quit = False
        cur_img_copy = self._cur_img.copy()
        
        for bbox in self._cur_bboxes + [self._bbbox_in_progress]:
            if bbox:
                self._draw_rect(cur_img_copy, bbox)
        
        cv2.imshow(self.name, cur_img_copy)
        key = self._get_key()

        if key == QUIT_KEY:
            should_quit = True
        elif key == NEXT_KEY:
            self._next_image()
        elif key == PREV_KEY:
            self._prev_image()
        elif key == CLEAR_KEY:
            self._clear()
        elif key == UNDO_KEY:
            self._undo()

        return should_quit

    def __enter__(self):
        cv2.namedWindow(self.name, self.type)
        cv2.setMouseCallback(self.name, self._mouse_callback)
        return self

    def __exit__(self, *args):
        if self._changed:
            self.index = self.index + 1
        
        cv2.destroyWindow(self.name)

    
def run(folder, obj_label, **kwargs):
    with PascalVOCAnnotator(folder, obj_label, **kwargs) as annotator:
        while not annotator.update():
            pass

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-f', '--folder', required=True, help='path to folder in which the images to be annotated are stored')
    arg_parser.add_argument('-l', '--label', required=True, help='object class label')
    args = vars(arg_parser.parse_args())
    run(args['folder'], args['label'])