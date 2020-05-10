#!/usr/bin/env python2
import os
import uuid
from datetime import datetime

import fire

import rospy
from cv_bridge import CvBridge
from PIL import Image as pImage
from sensor_msgs.msg import Image

DATA_FOLDER = '/workdir/data'


def imgmsg_to_arr(msg, encoding='rgb8'):
    bridge = CvBridge()
    msg.encoding = str(msg.encoding)
    return bridge.imgmsg_to_cv2(msg, encoding)


def imgmsg_to_img(msg, encoding='rgb8'):
    img = imgmsg_to_arr(msg, encoding=encoding)
    img = pImage.fromarray(img)
    return img


class Recorder():
    def __init__(self):
        self.session = 'session_{}'.format(uuid.uuid4().hex)
        os.makedirs(os.path.join(DATA_FOLDER, self.session))
        self.idx = 0

    def record_one(self, *args, **kwargs):
        msg = rospy.wait_for_message('/cv_camera/image_raw', Image, timeout=1)
        img = imgmsg_to_img(msg)
        name = '{}T{}_image.jpg'.format(datetime.now().strftime('%Y%m%d'),
                                        datetime.now().strftime('%H%M%S'))
        rospy.loginfo('Logging image {}: {}'.format(self.idx, name))
        fname = os.path.join(DATA_FOLDER, self.session, name)
        img.save(fname)
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
