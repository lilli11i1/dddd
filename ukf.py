#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rospy
from dataclasses import dataclass

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, Quaternion

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# ----------------------- Settings (오직 여기만 바꿔도 됨) -----------------------
@dataclass
class Settings:
    frame_id: str
    dt: float                 # 기본 주기(초)
    # 과정잡음 Q (상태: x, y, yaw(rad), v)
    q_pos: float              # x,y에 동일하게 적용 (m^2/s 단위 스케일)
    q_yaw_deg: float          # yaw 과정잡음(도) -> 내부에서 rad로 변환
    q_vel: float              # v 과정잡음
    # 관측잡음 R
    r_xy: float               # 오돔 x,y 관측 표준편차 (m)
    r_yaw_deg: float          # IMU yaw 관측 표준편차 (deg) -> rad 변환
    # 초기 공분산 P
    p_pos0: float             # 초기 x,y 분산
    p_yaw0_deg: float         # 초기 yaw 분산(도) -> rad 변환
    p_vel0: float             # 초기 v 분산

def load_settings(ns="~") -> Settings:
    """ROS 파라미터에서 Settings 로드 (모든 상수는 여기만)"""
    g = rospy.get_param
    return Settings(
        frame_id     = g(ns + "frame_id", "map"),
        dt           = float(g(ns + "dt", 0.05)),         # 20Hz
        q_pos        = float(g(ns + "q_pos", 0.01)),
        q_yaw_deg    = float(g(ns + "q_yaw_deg", 1.0)),   # deg
        q_vel        = float(g(ns + "q_vel", 0.05)),
        r_xy         = float(g(ns + "r_xy", 0.05)),
        r_yaw_deg    = float(g(ns + "r_yaw_deg", 2.0)),
        p_pos0       = float(g(ns + "p_pos0", 0.5)),
        p_yaw0_deg   = float(g(ns + "p_yaw0_deg", 10.0)), # deg
        p_vel0       = float(g(ns + "p_vel0", 0.5)),
    )

# ----------------------- 유틸 -----------------------
def quat_to_yaw(q: Quaternion) -> float:
    """geometry_msgs/Quaternion -> yaw(rad)"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def wrap_to_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def unwrap_yaw(meas, ref):
    d = meas - ref
    while d > math.pi:
        meas -= 2.0 * math.pi
        d = meas - ref
    while d < -math.pi:
        meas += 2.0 * math.pi
        d = meas - ref
    return meas

# ----------------------- 시스템/관측 모델 -----------------------
# 상태: x, y, yaw, v
def fx(x, dt):
    x = x.copy()
    x[0] += x[3] * math.cos(x[2]) * dt
    x[1] += x[3] * math.sin(x[2]) * dt
    x[2]  = wrap_to_pi(x[2])
    return x

def hx_xy(x):
    return np.array([x[0], x[1]])

def hx_yaw(x):
    return np.array([x[2]])

# ----------------------- 노드 -----------------------
class UKFNode:
    def __init__(self):
        rospy.init_node("ukf_node")
        self.s = load_settings("~")   # 상수 한 곳에서만 온다
        rospy.loginfo("UKF settings: %s", self.s)

        self.points = MerweScaledSigmaPoints(n=4, alpha=0.2, beta=2.0, kappa=0.0)
        self.ukf = UKF(dim_x=4, dim_z=2, fx=fx, hx=hx_xy, dt=self.s.dt, points=self.points)

        # 초기 상태/공분산
        self.ukf.x = np.array([0.0, 0.0, 0.0, 0.0])
        self.ukf.P = np.diag([
            self.s.p_pos0,                      # x
            self.s.p_pos0,                      # y
            (math.radians(self.s.p_yaw0_deg))**2,
            self.s.p_vel0
        ])

        # Q (dt 스케일은 예측시 적용)
        self.Q_base = np.diag([
            self.s.q_pos, self.s.q_pos,
            (math.radians(self.s.q_yaw_deg))**2,
            self.s.q_vel
        ])
        # R
        self.R_xy  = np.diag([self.s.r_xy**2, self.s.r_xy**2])
        self.R_yaw = np.array([[ (math.radians(self.s.r_yaw_deg))**2 ]])

        self.last_stamp = None

        # Pub/Sub
        self.pub_pose = rospy.Publisher("/ukf_pose", PoseStamped, queue_size=10)
        rospy.Subscriber("/odom", Odometry, self.on_odom, queue_size=50)
        rospy.Subscriber("/imu/data", Imu, self.on_imu, queue_size=50)

        # 주기 예측(센서 없어도 계속)
        self.timer = rospy.Timer(rospy.Duration(self.s.dt), self.on_timer)
        rospy.loginfo("UKF node up. Sub: /odom, /imu/data  Pub: /ukf_pose")

    # 공통 예측
    def predict(self, stamp):
        if self.last_stamp is None:
            dt = self.s.dt
        else:
            dt = (stamp - self.last_stamp).to_sec()
            if dt <= 0.0 or dt > 0.5:
                dt = self.s.dt
        self.last_stamp = stamp
        self.ukf.dt = dt
        self.ukf.Q = self.Q_base * max(dt, 1e-3)
        self.ukf.predict()

    def on_odom(self, msg: Odometry):
        self.predict(msg.header.stamp)
        z = np.array([msg.pose.pose.position.x,
                      msg.pose.pose.position.y])
        self.ukf.update(z, R=self.R_xy, hx=hx_xy)

        # 속도 힌트(선택): 오돔 twist → v
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        v_meas = math.hypot(vx, vy)
        self.ukf.x[3] = 0.9*self.ukf.x[3] + 0.1*v_meas

        self.publish_pose(msg.header.stamp)

    def on_imu(self, msg: Imu):
        stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0 else rospy.Time.now()
        self.predict(stamp)
        yaw = unwrap_yaw(quat_to_yaw(msg.orientation), self.ukf.x[2])
        z = np.array([yaw])
        self.ukf.update(z, R=self.R_yaw, hx=hx_yaw)
        self.publish_pose(stamp)

    def on_timer(self, event):
        now = rospy.Time.now()
        self.predict(now)
        self.publish_pose(now)

    def publish_pose(self, stamp):
        x, y, yaw, _ = self.ukf.x
        m = PoseStamped()
        m.header.stamp = stamp
        m.header.frame_id = self.s.frame_id
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.0
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = math.sin(yaw/2.0)
        m.pose.orientation.w = math.cos(yaw/2.0)
        self.pub_pose.publish(m)

def main():
    try:
        UKFNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
