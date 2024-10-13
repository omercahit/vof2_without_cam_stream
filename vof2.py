import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import cv2
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import os
import numpy as np
import yaml
import tf_transformations as tf

class Vof2(Node):
    def __init__(self):
        super().__init__('vof2')

        cwd = os.path.dirname(os.path.realpath(__file__))
        cwd = (cwd.split(os.sep)[:-1])
        cwd = (os.sep.join(cwd))

        cwd = cwd + '/vof2_without_cam_stream/params/vof2_params.yaml'

        with open(cwd, 'r') as file:
            self.params = yaml.safe_load(file)

        print(self.params)
        
        self.frame = None
        self.d1 = []
        self.d2 = []
        self.prev_gray = None
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.prev_angle = 0
        self.scale=1000
        self.quat = [0,0,0,1]
        self.loc = self.params['initial_pose']

        self.T_base_camera = self.create_transformation_matrix(self.params['transform_to_base'][0],
        self.params['transform_to_base'][1])
        #####
        self.publisher_ = self.create_publisher(Image, '/camera/vof_image', 10)
        self.bridge = CvBridge()  # OpenCV -> ROS mesaj dönüşümü

        self.cap = cv2.VideoCapture(0)

        # Timer ile her 10 Hz'de bir görüntü yayınlama
        self.timer = self.create_timer(0.1, self.timer_callback)

    def create_transformation_matrix(self, translation, rotation):
        """
        translation: [x, y, z]
        rotation: [roll, pitch, yaw] in radians
        """
        tx, ty, tz = translation
        roll, pitch, yaw = rotation

        Rx = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
        
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
        
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
        
        R = np.dot(Rz, np.dot(Ry, Rx))
        
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = [tx, ty, tz]
        
        return T

    def timer_callback(self):
        ret, self.frame = self.cap.read()
        if ret:
            self.frame = self.frame[60:660, 340:940]
            
            ros_image = self.bridge.cv2_to_imgmsg(self.frame, encoding='bgr8')
            self.publisher_.publish(ros_image)

            if self.prev_gray is None:
                self.prev_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                return
            
            lk_params = dict(winSize=(120, 120), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                    
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **lk_params)

            self.good_new = p1[st == 1]
            self.good_old = p0[st == 1]

            temp_d1s = []
            temp_d2s = []
            for i, (new, old) in enumerate(zip(self.good_new, self.good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                temp_d1s.append(a - c)
                temp_d2s.append(b - d)

                self.frame = cv2.arrowedLine(self.frame, (int(c), int(d)), (int(a), int(b)),
                (0, 255, 0), 2)

                self.frame = cv2.circle(self.frame, (int(a), int(b)), 5, (0, 0, 255), -1)

            temp_d1 = np.median(np.array(temp_d1s)) * self.params['calibration_multiplier']
            temp_d2 = np.median(np.array(temp_d2s)) * self.params['calibration_multiplier']
            self.d1.append(temp_d1)
            self.d2.append(temp_d2)
            self.loc[0] = self.loc[0] + temp_d1
            self.loc[1] = self.loc[1] + temp_d2

            if np.isnan(temp_d1) or np.isnan(temp_d2):
                return

            try:
                H, mask = cv2.findHomography(self.good_old, self.good_new, cv2.RANSAC, 5.0)
                theta = np.arctan2(H[1, 0], H[0, 0])
                #print(H[1, 0], H[0, 0])
                self.loc[2] = self.loc[2] + theta
                print(f"Açısal değişim: {np.degrees(self.loc[2])} derece")
            except:
                print("Not enough points to calculate angle")

            magnitude = np.sqrt(temp_d1**2 + temp_d2**2)
            self.angle = np.arctan2(temp_d2, temp_d1) - np.pi/2
            #angle = theta

            if magnitude > 0:
                T_camera_new_old = self.create_transformation_matrix([temp_d1, temp_d2, 0], [0, 0, self.loc[2]])

                self.angle += self.prev_angle

                T_base_camera_new = np.dot(self.T_base_camera, T_camera_new_old)
                T_base_new = np.dot(np.linalg.inv(self.T_base_camera), T_base_camera_new)
                new_position = T_base_new[0:3, 3]
                new_orientation = [np.arctan2(T_base_new[2, 1], T_base_new[2, 2]),
                                np.arctan2(-T_base_new[2, 0], np.sqrt(T_base_new[2, 1]**2 + T_base_new[2, 2]**2)),
                                np.arctan2(T_base_new[1, 0], T_base_new[0, 0])]

                self.quat = tf.quaternion_from_matrix(T_base_new)

                broadcaster = StaticTransformBroadcaster(self)

                tf_msg = TransformStamped()
                tf_msg.header.stamp = self.get_clock().now().to_msg()
                tf_msg.header.frame_id = "map"
                tf_msg.child_frame_id = self.params['tf_frame_name']
                tf_msg.transform.translation.x = self.loc[1] - 2.00
                tf_msg.transform.translation.y = self.loc[0] - 0.50
                tf_msg.transform.translation.z = 0 + 0.1
                tf_msg.transform.rotation.x = float(self.quat[0])
                tf_msg.transform.rotation.y = float(self.quat[1])
                tf_msg.transform.rotation.z = float(self.quat[2])
                tf_msg.transform.rotation.w = float(self.quat[3])

                broadcaster.sendTransform(tf_msg)

                self.prev_angle = self.angle

            self.prev_gray = gray

            cv2.imshow('img', self.frame)
            
        else:
            self.get_logger().error('Camera frame could not be read.')

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    image_publisher = Vof2()

    try:
        rclpy.spin(image_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        image_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

