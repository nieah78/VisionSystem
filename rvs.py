import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from robotvisionsystem_msgs.msg import Motor
from robotvisionsystem_msgs.msg import State
from robotvisionsystem_msgs.msg import Ray

import numpy as np
import cv2
from cv_bridge import CvBridge

import math

class RobotVisionSystem(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.pub_motor = self.create_publisher(Motor, '/car/motor', 10)

        self.sub_state = self.create_subscription(State, '/car/state', self._state, 10)
        self.sub_ray = self.create_subscription(Ray, '/car/sensor/ray', self._ray, 10)
        self.sub_front_image = self.create_subscription(Image, '/car/sensor/camera/front', self._image, 10)
        
        self.i = 0
        self.image = np.empty(shape=[1])

        self.timer = self.create_timer(0.01, self.rvs)

        self.line_roi = [[0, 640], [240, 400]]
        self.traffic_light_roi = [[270,360],[160,200]]

    def rvs(self):
        self.sub_front_image
        self.sub_ray
        self.sub_state

        msg = Motor()
        # if 0 < self.i <= 1:    
        #     msg.steer = 0.0
        #     msg.motorspeed = 1.0
        
        # elif 1 < self.i <= 5:    
        #     msg.steer = 30.0
        #     msg.motorspeed = 1.0

        # elif 5 < self.i <= 10:
        #     msg.steer = -30.0
        #     msg.motorspeed = 5.0

        ## _line 함수에서의 결과에 따라 모터를 움직여야 할 것 같다!
        ## 아마 steer, moterspeed, breakbool 이 세 가지만 이용해 동작시키는듯

        # 신호등 감지
        light_status = self._trafficlight(self.image)

        # 자동차 동작 제어
        if light_status == "RED":
            msg.motorspeed = 0.0  # 멈춤
        elif light_status == "GREEN":
            msg.motorspeed = 5.0  # 이동
        else:
            msg.motorspeed = 0.0  # 예외 처리 (급발진)

        self.pub_motor.publish(msg)
        # self.get_logger().info("Steer : %s MotorSpeed : %s Break : %s" % (msg.steer, msg.motorspeed, msg.breakbool))

        right_line_mean, left_line_mean = self._line(self.image, self.line_roi[0], self.line_roi[1])
        self.i += 1

        self.viewer(right_line_mean, left_line_mean)

        # self._image()
    
    def _trafficlight(self, image):
        try:
            # 신호등의 ROI 설정
            traffic_light_roi = image[160:200, 270:360]

            # ROI를 HSV 색공간으로 변환
            hsv = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)

            # 빨간불 범위 (HSV 색상 기준)
            lower_red1 = np.array([0, 100, 100])  # 빨간색 하한값
            upper_red1 = np.array([10, 255, 255])  # 빨간색 상한값
            lower_red2 = np.array([160, 100, 100])  # 빨간색 하한값 (2번째 범위)
            upper_red2 = np.array([180, 255, 255])  # 빨간색 상한값 (2번째 범위)

            # 초록불 범위 (HSV 색상 기준)
            lower_green = np.array([50, 100, 100])  # 초록색 하한값
            upper_green = np.array([70, 255, 255])  # 초록색 상한값

            # 빨간불 마스크
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            # 초록불 마스크
            mask_green = cv2.inRange(hsv, lower_green, upper_green)

            # 빨간불과 초록불 픽셀 개수 계산
            red_pixels = cv2.countNonZero(mask_red)
            green_pixels = cv2.countNonZero(mask_green)

            # 특정 영역 이상의 픽셀 개수로 빨간불/초록불 판별
            threshold = 100  # 픽셀 개수 기준값
            if red_pixels > threshold:
                print(f"Red Light Detected: {red_pixels} pixels")
                return "RED"
            elif green_pixels > threshold:
                print(f"Green Light Detected: {green_pixels} pixels")
                return "GREEN"
            else:
                print("Unable to Detect Light")
                return "UNKNOWN"
        except Exception as ex:
            print(f"[Error] [_trafficlight] Line : {ex.__traceback__.tb_lineno} | {ex}")
            return "ERROR"

    
    def _state(self, data):
        None
        # print(data)
    
    def _ray(self, data):
        None
        # print(data)
    
    def _line(self, image, width=[0,640], height=[240,400]):
        try:
            if len(image.shape) > 1:
                img_height, img_width, _ = image.shape
                if img_height > 0 and img_width > 0:
                    roi_image = image[height[0]:height[1], width[0]:width[1]]

                    # Gray scale
                    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

                    # GaussianBlur
                    blur_gray = cv2.GaussianBlur(gray,(5, 5), 0)

                    # ROI Area
                    edge_img = cv2.Canny(np.uint8(blur_gray), 60, 70)

                    # HoughLinesP
                    all_lines = cv2.HoughLinesP(edge_img, 1, math.pi/180,30,30,10)                  

                    # draw lines in ROI area
                    # calculate slope and do filtering
                    right_lines = []
                    left_lines = []
                    for line in all_lines:
                        x1, y1, x2, y2 = line[0]
                        slope = 0
                        if (x2 - x1) == 0:
                            slope = 0
                        else:
                            slope = float(y2-y1) / float(x2-x1)

                        if 0.1 < slope < 10: # right_line
                            right_lines.append(line[0])

                        elif -10 < slope < -0.1: # left_line
                            left_lines.append(line[0])
                    
                    rlm = np.array(right_lines).mean(axis=0)
                    llm = np.array(left_lines).mean(axis=0)

                    print(f"{rlm}, {llm}")

                    return rlm, llm
            else:
                return [0,0,0,0], [0,0,0,0]

        except Exception as ex:
            print(f"\033[31m[Error] [_line]\033[0m Line : {ex.__traceback__.tb_lineno} | {ex}")

    def _stopline(self):
        print()
        
    def _image(self, data):
        # print(data)
        self.image = CvBridge().imgmsg_to_cv2(data, 'bgr8')
    
    def viewer(self, right_line_mean, left_line_mean):
        cv2.imshow('img', self.image)

        line_image = self.image.copy()
        cv2.rectangle(line_image, (int(self.line_roi[0][0]), int(self.line_roi[1][0])), (int(self.line_roi[0][1]), int(self.line_roi[1][1])), (0,255,0), 3)
        cv2.rectangle(line_image, (int(self.traffic_light_roi[0][0]), int(self.traffic_light_roi[1][0])), (int(self.traffic_light_roi[0][1]), int(self.traffic_light_roi[1][1])), (0,255,0), 3)
        cv2.line(line_image, (int(right_line_mean[0]), int(self.line_roi[1][0])),
                             (int(right_line_mean[2]), int(self.line_roi[1][1])),
                             (255,0,0),
                             1)
        cv2.line(line_image, (int(left_line_mean[0]), int(self.line_roi[1][0])),
                             (int(left_line_mean[2]), int(self.line_roi[1][1])),
                             (255,0,0),
                             1)
        cv2.imshow('line', line_image)
        cv2.waitKey(33)


def main(args=None):
    rclpy.init(args=args)

    rvs = RobotVisionSystem()

    rclpy.spin(rvs)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    rvs.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
