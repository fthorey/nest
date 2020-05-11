#!/usr/bin/env python2
import json
import os
import uuid
from datetime import datetime

import fire
import numpy as np

import rospy
import torch
import torch.nn as nn
from cv_bridge import CvBridge
from PIL import Image as pImage
from sensor_msgs.msg import Image
from torchvision import models
from torchvision import transforms as T

DATA_FOLDER = '/workdir/data'
ALPHA = 0.99


def imgmsg_to_arr(msg, encoding='rgb8'):
    bridge = CvBridge()
    msg.encoding = str(msg.encoding)
    return bridge.imgmsg_to_cv2(msg, encoding)


class Model(object):
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.cls_to_label = json.load(open('scripts/cls_to_id.json'))

    def detect(self, imgs):
        imgs = torch.stack([self.transform(img) for img in imgs], 0)
        with torch.no_grad():
            preds = self.model(imgs)
            probs = nn.functional.softmax(preds[0], dim=0)
        idxs = np.arange(1000)
        labels = self.cls_to_label.values()
        mask = np.isin(idxs, self.cls_to_label.keys())
        preds = zip(labels, np.array(probs[mask]))
        preds = sorted(preds, key=lambda x: -x[-1])[0]
        if preds[1] < 0.25:
            return ''
        return preds[0].replace(' ', '_')


class Recorder():
    def __init__(self):
        self.session = 'session_{}'.format(uuid.uuid4().hex)
        os.makedirs(os.path.join(DATA_FOLDER, self.session))
        self.idx = 0
        self.running_mean = 0
        self.model = Model()
        self.detection_idx = 0

    def save(self, img, tag):
        name = '{}T{}_{}.jpg'.format(datetime.now().strftime('%Y%m%d'),
                                     datetime.now().strftime('%H%M%S'), tag)
        rospy.loginfo('Logging image {}: {}'.format(self.idx, name))
        fname = os.path.join(DATA_FOLDER, self.session, name)
        img.save(fname)

    def record_one(self, *args, **kwargs):
        msg = rospy.wait_for_message('/cv_camera/image_raw', Image, timeout=1)
        arr = imgmsg_to_arr(msg)
        img = pImage.fromarray(arr)
        self.save(img, 'image')
        self.idx += 1

    def callback(self, msg):
        img = pImage.fromarray(imgmsg_to_arr(msg))
        imgs = [img]
        preds = self.model.detect(imgs)
        if 'cat' in preds:
            rospy.loginfo('Detected a cat')
            self.save(img, 'cat')
        elif 'dog' in preds:
            rospy.loginfo('Detected a dog')
            self.save(img, 'dog')
        self.detection_idx += 1
        if self.detection_idx % 100 == 0:
            rospy.loginfo('We have been running {} detections so far'.format(
                self.detection_idx))


def start(frequency=0.2):
    rospy.init_node('recorder')
    rospy.sleep(5)
    rec = Recorder()
    rospy.loginfo('*' * 50)
    rospy.loginfo('Starting session {}'.format(rec.session))
    rospy.loginfo('*' * 50)
    rospy.Timer(rospy.Duration(1 / frequency), rec.record_one)
    rospy.Subscriber("/cv_camera/image_raw", Image, rec.callback)
    rospy.spin()


if __name__ == '__main__':
    fire.Fire(start)
