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
import time

class RobotVisionSystem(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.pub_motor = self.create_publisher(Motor, '/car/motor', 10)

        self.sub_state = self.create_subscription(State, '/car/state', self._state, 10)
        self.sub_ray = self.create_subscription(Ray, '/car/sensor/ray', self._ray, 10)
        self.sub_front_image = self.create_subscription(Image, '/car/sensor/camera/front', self._image, 10)
        
        self.image = np.empty(shape=[1])

        self.current_state = 0  # 현재 경로 상태 (0: Start -> Way0, ..., 4: Way3 -> Start)
        self.is_turning = False  # 회전 동작 상태 (좌/우회전 중인지 여부)
        self.turn_timer = 0  # 회전 타이머 시작 시간
        self.timer = self.create_timer(0.01, self.rvs) # 회전 동작용 타이머

        self.line_roi = [[0, 640], [240, 400]]
        self.traffic_light_roi = [[270,360],[170,200]]

    def rvs(self):
        # 센서 데이터 업데이트
        self.sub_front_image
        self.sub_ray
        self.sub_state

        msg = Motor()

        # current_state 실시간 출력
        self.get_logger().info(f"현재 상태: {self.current_state}")

        # 신호등 감지
        light_status = self._trafficlight(self.image)  # RED, GREEN, UNKNOWN

        # 회전 동작이 진행 중일 때 타이머 체크
        if self.is_turning:
            self.get_logger().info(f"회전 시간: {time.time() - self.turn_timer}")
            if time.time() - self.turn_timer >= 6:  # 6초 회전 후 종료
                self.is_turning = False
                self.get_logger().info("회전 완료!")

                # 회전 완료 후 상태 업데이트
                if self.current_state == 0:
                    self.current_state = 1
                    
                elif self.current_state == 1:
                    self.current_state = 2

                elif self.current_state == 2:
                    self.current_state = 3

                elif self.current_state == 3:
                    self.current_state = 4

                self.get_logger().info(f"다음 경로로 넘어갑니다.")

            else:
                #self.pub_motor.publish(msg)
                return  # 회전 중에는 다른 동작 수행 안 함

        # 경로 상태별 동작
        if self.current_state == 0:  # Start -> Way0
            if not self.is_turning:
                if light_status == "GREEN":
                    self.get_logger().info("초록불을 인식했습니다. 좌회전을 시작합니다.")

                    self.is_turning = True
                    self.turn_timer = time.time()  # 회전 시작 시간 기록
                    msg.steer = -10.0
                    msg.motorspeed = 0.12
                    self.get_logger().info("좌회전 시작!")

        elif self.current_state == 1:  # Way0 -> Way1
            if not self.is_turning:
                if light_status == "RED":
                    self.stop(msg)
                elif light_status == "GREEN":
                    self.turn_right(msg)
                else:
                    self.go_straight(msg)

        elif self.current_state == 2:  # Way1 -> Way2
            self.handle_state_with_turn(msg, light_status, 3)

        elif self.current_state == 3:  # Way2 -> Way3
            self.handle_state_with_turn(msg, light_status, 4)

        elif self.current_state == 4:  # Way3 -> Start
            if light_status == "RED":
                self.stop(msg)
            else:
                self.go_straight(msg)
    
        self.pub_motor.publish(msg)


        right_line_mean, left_line_mean = self._line(self.image, self.line_roi[0], self.line_roi[1])

        self.viewer(right_line_mean, left_line_mean)

    def handle_state_with_turn(self, msg, light_status, next_state):
        """공통 상태 처리: 직진/RED 정지/GREEN 회전"""
        if not self.is_turning:
            if light_status == "RED":
                self.stop(msg)
            elif light_status == "GREEN":
                self.is_turning = True
                self.turn_right(msg)
            else:  # UNKNOWN 상태
                self.go_straight(msg)
        elif self.is_turning:  # 우회전 완료 후 상태 변경
            self.is_turning = False
            self.current_state = next_state

    def turn_left(self, msg):
        if not self.is_turning:  # 회전 중이 아닐 때만 회전 시작
            self.is_turning = True
            self.turn_timer = time.time()  # 회전 시작 시간 기록
            msg.steer = -20.0
            msg.motorspeed = 3.0
            self.get_logger().info("좌회전 시작!")

    def turn_right(self, msg):
        self.set_turn_motion(msg, 15.0, "우회전 시작!")

    def go_straight(self, msg):
        msg.steer = 0.0
        msg.motorspeed = 0.1
        self.get_logger().info("직진!")

    def stop(self, msg):
        msg.steer = 0.0
        msg.motorspeed = 0.0
        self.get_logger().info("정지!")

    def _trafficlight(self, image):
        try:
            # 신호등의 ROI 설정
            traffic_light_roi = image[170:200, 270:360]

            # ROI 영역 내에서 조건에 따라 RED 또는 GREEN 상태 구분
            red_dectect = (traffic_light_roi[:, :, 2] >= 200) & \
                            (traffic_light_roi[:, :, 1] <= 100) & \
                            (traffic_light_roi[:, :, 0] <= 100)
            yellow_detect = (traffic_light_roi[:, :, 1] >= 200) & \
                            (traffic_light_roi[:, :, 2] >= 200) & \
                            (traffic_light_roi[:, :, 0] <= 100)
            green_detect = (traffic_light_roi[:, :, 1] >= 200) & \
                            (traffic_light_roi[:, :, 2] <= 100) & \
                            (traffic_light_roi[:, :, 0] <= 100)

            # 조건에 부합하는 픽셀 수 계산
            red_pixels = np.sum(red_dectect)
            yellow_pixels = np.sum(yellow_detect)
            green_pixels = np.sum(green_detect)

            # 픽셀 개수에 따라 상태 결정
            threshold = 30  # 픽셀 개수 기준값
            if red_pixels > threshold:
                print(f"빨간불: {red_pixels} 픽셀")
                return "RED"
            
            elif yellow_pixels > threshold:
                print(f"노란불: {yellow_pixels} 픽셀")
                return "YELLOW"
            
            elif green_pixels > threshold:
                print(f"초록불: {green_pixels} 픽셀")
                return "GREEN"
            
            else:
                print("감지된 신호가 없습니다.")
                return "NOTHING"
                
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

                    #print(f"{rlm}, {llm}")

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
