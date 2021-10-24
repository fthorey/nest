#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import redis


def imgmsg_to_arr(msg, encoding="rgb8"):
    bridge = CvBridge()
    msg.encoding = str(msg.encoding)
    return bridge.imgmsg_to_cv2(msg, encoding)


def on_message(msg):
    img = imgmsg_to_arr(msg)
    encoded_frame = img.tobytes()
    con = redis.Redis()
    con.set("stream", encoded_frame)


def main():
    print("In main")
    rospy.init_node("ui", anonymous=True)
    rospy.sleep(5)
    rospy.loginfo("*" * 50)
    rospy.loginfo("Starting feed")
    rospy.loginfo("*" * 50)
    rospy.Subscriber("/cv_camera/image_raw", Image, on_message)
    rospy.spin()


if __name__ == "__main__":
    main()
