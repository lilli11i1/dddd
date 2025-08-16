# =============================
# File: dnn_cmdvel_node.py
# Description: ROS1 Noetic node for TurtleBot3 that loads a TorchScript DNN
#              to predict (linear, angular) velocities from state and publishes /cmd_vel.
#              Requires: rospy, numpy, torch, joblib, geometry_msgs, nav_msgs, tf.transformations
# Run: rosrun your_pkg dnn_cmdvel_node.py _model_path:=/home/ubuntu/models/velocity_predictor.pt \
#      _input_scaler_path:=/home/ubuntu/models/input_scaler.pkl _output_scaler_path:=/home/ubuntu/models/output_scaler.pkl
# =============================

from __future__ import annotations
import os
import time
import math
import numpy as np
import rospy
import torch
import joblib
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion


def quat_to_yaw(q):
    return euler_from_quaternion([q.x, q.y, q.z, q.w])[2]


class DNNCmdVelNode:
    def __init__(self):
        # --- Parameters ---
        self.model_path = rospy.get_param('~model_path', '/home/ubuntu/models/velocity_predictor.pt')
        self.input_scaler_path = rospy.get_param('~input_scaler_path', '/home/ubuntu/models/input_scaler.pkl')
        self.output_scaler_path = rospy.get_param('~output_scaler_path', '/home/ubuntu/models/output_scaler.pkl')
        self.rate_hz = rospy.get_param('~rate', 50)
        self.linear_limit = rospy.get_param('~linear_limit', 0.8)    # m/s
        self.angular_limit = rospy.get_param('~angular_limit', 1.8)  # rad/s
        self.cmd_topic = rospy.get_param('~cmd_topic', '/cmd_vel')
        self.odom_topic = rospy.get_param('~odom_topic', '/odom')
        self.timeout_sec = rospy.get_param('~odom_timeout', 0.5)
        self.use_pose_xy = rospy.get_param('~use_pose_xy', True)  # if False, zeros x,y
        self.lowpass_alpha = rospy.get_param('~lowpass_alpha', 0.35)  # 0..1, higher = less smoothing

        # --- Load model ---
        if not os.path.exists(self.model_path):
            rospy.logfatal('Model file not found: %s', self.model_path)
            raise SystemExit(1)
        try:
            self.model = torch.jit.load(self.model_path, map_location='cpu')
            self.model.eval()
        except Exception as e:
            rospy.logfatal('Failed to load TorchScript model: %s', e)
            raise SystemExit(1)

        # --- Load scalers (sklearn MinMaxScaler saved with joblib) ---
        if not os.path.exists(self.input_scaler_path) or not os.path.exists(self.output_scaler_path):
            rospy.logfatal('Scaler files not found: %s, %s', self.input_scaler_path, self.output_scaler_path)
            raise SystemExit(1)
        try:
            self.input_scaler = joblib.load(self.input_scaler_path)
            self.output_scaler = joblib.load(self.output_scaler_path)
        except Exception as e:
            rospy.logfatal('Failed to load scalers: %s', e)
            raise SystemExit(1)

        # --- ROS I/O ---
        self.pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)
        self.sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=10)

        # --- State ---
        self.last_odom_time = None
        self.state = None  # (x, y, yaw, v)
        self.prev_cmd = np.zeros(2, dtype=np.float32)  # for smoothing

        rospy.loginfo('DNNCmdVelNode initialized. Model: %s', self.model_path)

    def odom_cb(self, msg: Odometry):
        # Pose
        if self.use_pose_xy:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
        else:
            x, y = 0.0, 0.0
        yaw = quat_to_yaw(msg.pose.pose.orientation)

        # Velocity magnitude in base frame (use twist)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        v = float(math.hypot(vx, vy))

        self.state = (float(x), float(y), float(yaw), v)
        self.last_odom_time = rospy.Time.now()

    def predict_cmd(self, state):
        # state: (x,y,theta,v)
        arr = np.array(state, dtype=np.float32).reshape(1, -1)
        # Normalize with training scaler
        try:
            arr_norm = self.input_scaler.transform(arr)
        except Exception as e:
            rospy.logwarn_throttle(2.0, 'Input scaler transform failed (%s). Using raw.', e)
            arr_norm = arr

        with torch.no_grad():
            x = torch.from_numpy(arr_norm).to(torch.float32)
            y_norm = self.model(x).cpu().numpy()  # shape (1,2)
        # Inverse scale to physical units
        try:
            y = self.output_scaler.inverse_transform(y_norm)
        except Exception as e:
            rospy.logwarn_throttle(2.0, 'Output inverse_transform failed (%s). Using raw.', e)
            y = y_norm

        cmd = y.reshape(-1)
        return cmd  # [v, w]

    def clamp(self, cmd):
        v, w = float(cmd[0]), float(cmd[1])
        v = max(-self.linear_limit, min(self.linear_limit, v))
        w = max(-self.angular_limit, min(self.angular_limit, w))
        return np.array([v, w], dtype=np.float32)

    def lowpass(self, cmd):
        # Simple IIR low-pass on (v, w)
        a = float(self.lowpass_alpha)
        smoothed = a * cmd + (1.0 - a) * self.prev_cmd
        self.prev_cmd = smoothed
        return smoothed

    def publish_twist(self, cmd_vw):
        msg = Twist()
        msg.linear.x = float(cmd_vw[0])
        msg.angular.z = float(cmd_vw[1])
        self.pub.publish(msg)

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            # Safety: stop if odom stale or no state yet
            if self.state is None or self.last_odom_time is None or (now - self.last_odom_time).to_sec() > self.timeout_sec:
                self.prev_cmd = np.zeros(2, dtype=np.float32)
                self.publish_twist(self.prev_cmd)
                rate.sleep()
                continue

            cmd = self.predict_cmd(self.state)
            if not np.all(np.isfinite(cmd)):
                rospy.logwarn_throttle(2.0, 'Non-finite cmd from model, sending zero')
                cmd = np.zeros(2, dtype=np.float32)

            cmd = self.clamp(cmd)
            cmd = self.lowpass(cmd)
            self.publish_twist(cmd)
            rate.sleep()


def main():
    rospy.init_node('dnn_cmdvel_node')
    node = DNNCmdVelNode()
    node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


# =============================
# File: export_model_and_scalers.py
# Description: Run this on your PC AFTER training to export TorchScript model and scalers.
# =============================

import torch
import joblib

# 1) Assuming you already have: `model`, `input_scaler`, `output_scaler`
#    from your training script (the one you posted). After training finishes:

def export(model, input_scaler, output_scaler, outdir='artifacts'):
    import os
    os.makedirs(outdir, exist_ok=True)

    # a) TorchScript export (preferred for ROS runtime)
    model.eval()
    example = torch.randn(1, 4)
    traced = torch.jit.trace(model, example)
    traced.save(os.path.join(outdir, 'velocity_predictor.pt'))

    # b) Save scalers with joblib
    joblib.dump(input_scaler, os.path.join(outdir, 'input_scaler.pkl'))
    joblib.dump(output_scaler, os.path.join(outdir, 'output_scaler.pkl'))

    print('Exported to:', os.path.abspath(outdir))

# Usage (inside your training script after training):
# export(model, input_scaler, output_scaler, outdir='artifacts')


# =============================
# File: cmdvel_dnn.launch (XML for ROS1 Noetic)
# Description: Example launch file
# =============================
# <launch>
#   <node pkg="your_pkg" type="dnn_cmdvel_node.py" name="dnn_cmdvel" output="screen">
#     <param name="model_path" value="/home/ubuntu/models/velocity_predictor.pt" />
#     <param name="input_scaler_path" value="/home/ubuntu/models/input_scaler.pkl" />
#     <param name="output_scaler_path" value="/home/ubuntu/models/output_scaler.pkl" />
#     <param name="rate" value="50" />
#     <param name="linear_limit" value="0.8" />
#     <param name="angular_limit" value="1.8" />
#     <param name="odom_timeout" value="0.5" />
#     <param name="use_pose_xy" value="true" />
#     <param name="lowpass_alpha" value="0.35" />
#     <param name="odom_topic" value="/odom" />
#     <param name="cmd_topic" value="/cmd_vel" />
#   </node>
# </launch>
