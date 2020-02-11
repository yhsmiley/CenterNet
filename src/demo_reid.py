from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import CenterNet.src._init_paths

import os
import cv2

from detectors.detector_factory import detector_factory

class Demo_Reid(object):
  def __init__(self, opt):
    self.image_ext = ['jpg', 'jpeg', 'png', 'webp']
    self.video_ext = ['mp4', 'mov', 'avi', 'mkv']
    self.time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

    self.opt = opt
    self.detector_factory = detector_factory

    os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpus_str
    self.opt.debug = max(self.opt.debug, 1)
    Detector = self.detector_factory[self.opt.task]
    self.detector = Detector(self.opt)

  def run(self):
    show_win_name = 'input'
    cv2.namedWindow(show_win_name, cv2.WINDOW_NORMAL)

    if self.opt.demo == 'webcam' or \
      self.opt.demo[self.opt.demo.rfind('.') + 1:].lower() in self.video_ext:
      cam = cv2.VideoCapture(0 if self.opt.demo == 'webcam' else self.opt.demo)
      detector.pause = False
      while True:
          status, img = cam.read()
          if not status:
            break
          cv2.imshow(show_win_name, img)
          ret = detector.run(img)
          time_str = ''
          for stat in self.time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
          print(time_str)
          if cv2.waitKey(1) == 27:
              return  # esc to quit
    else:
      if os.path.isdir(self.opt.demo):
        image_names = []
        ls = os.listdir(self.opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in self.image_ext:
                image_names.append(os.path.join(self.opt.demo, file_name))
      else:
        image_names = [self.opt.demo]
      
      for (image_name) in image_names:
        ret = detector.run(image_name)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)

  def get_detections_dict(self, frames, classes=None):
    bbs = self.detector.run_reid(frames[0], classes=classes)

    return [bbs]