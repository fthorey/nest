#!/usr/bin/env python2
import os
import uuid
from datetime import datetime

import fire
import numpy as np

import rospy
from cv_bridge import CvBridge
from PIL import Image as pImage
from sensor_msgs.msg import Image

DATA_FOLDER = '/workdir/data'
ALPHA = 0.99


def imgmsg_to_arr(msg, encoding='rgb8'):
    bridge = CvBridge()
    msg.encoding = str(msg.encoding)
    return bridge.imgmsg_to_cv2(msg, encoding)


class Recorder():
    def __init__(self):
        self.session = 'session_{}'.format(uuid.uuid4().hex)
        os.makedirs(os.path.join(DATA_FOLDER, self.session))
        self.idx = 0
        self.running_mean = 0

    def save(self, img, tag):
        name = '{}T{}_{}.jpg'.format(datetime.now().strftime('%Y%m%d'),
                                     datetime.now().strftime('%H%M%S'), tag)
        rospy.loginfo('Logging image {}: {}'.format(self.idx, name))
        fname = os.path.join(DATA_FOLDER, self.session, name)
        img.save(fname)

    def is_outlier(self, arr, threshold=10):
        if self.idx < 100:
            return False
        return abs(np.mean(arr) - self.running_mean) > 10

    def record_one(self, *args, **kwargs):
        msg = rospy.wait_for_message('/cv_camera/image_raw', Image, timeout=1)
        arr = imgmsg_to_arr(msg)
        img = pImage.fromarray(arr)
        self.save(img, 'image')
        if self.is_outlier(arr):
            self.save(img, 'outlier')
        else:
            self.running_mean = ALPHA * self.running_mean + (
                1 - ALPHA) * np.mean(arr)
        self.idx += 1


def start(frequency=0.2):
    rospy.init_node('recorder')
    rec = Recorder()
    rospy.loginfo('*' * 50)
    rospy.loginfo('Starting session {}'.format(rec.session))
    rospy.loginfo('*' * 50)
    rospy.Timer(rospy.Duration(1 / frequency), rec.record_one)
    rospy.spin()


if __name__ == '__main__':
    fire.Fire(start)
