#!/usr/bin/env python3
import rospy
import numpy as np
import casadi as ca
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Path
import tf.transformations as tft
import math


class NMPCFilter:
    def __init__(self):
        # --- Parameters ---
        self.state_topic = rospy.get_param('~state_topic', 'amcl_pose')
        self.cmd_in_topic = rospy.get_param('~cmd_in_topic', 'cmd_vel')       # 원본 입력 cmd
        self.cmd_out_topic = rospy.get_param('~cmd_out_topic', 'cmd_vel_filtered')  # 보정된 cmd
        self.path_topic = rospy.get_param('~path_topic', 'waypoints')

        self.N = int(rospy.get_param('~horizon', 10))
        self.dt = float(rospy.get_param('~dt', 0.1))

        # 물리 제약 (TurtleBot3 Burger 기준)
        self.v_max = float(rospy.get_param('~v_max', 0.22))
        self.w_max = float(rospy.get_param('~w_max', 2.84))
        self.dv_max = float(rospy.get_param('~dv_max', 0.1))
        self.dw_max = float(rospy.get_param('~dw_max', 0.5))

        # Cost Weights
        self.q_cmd = float(rospy.get_param('~q_cmd', 10.0))  # 원래 cmd와의 차이 최소화
        self.q_terminal = float(rospy.get_param('~q_terminal', 1.0))  # 목표점 안정화

        # --- State holders ---
        self.pose = None
        self.path = []
        self.last_U = np.zeros((2, self.N))

        # --- ROS I/O ---
        self.cmd_pub = rospy.Publisher(self.cmd_out_topic, Twist, queue_size=10)
        rospy.Subscriber(self.state_topic, PoseWithCovarianceStamped, self.cb_pose)
        rospy.Subscriber(self.path_topic, Path, self.cb_path)
        rospy.Subscriber(self.cmd_in_topic, Twist, self.cb_cmd_in)

        # --- Build NMPC optimizer ---
        self._build_opt()
        rospy.loginfo("NMPC Filter Node started. Listening on %s → Publishing %s",
                      self.cmd_in_topic, self.cmd_out_topic)

    def cb_pose(self, msg):
        q = msg.pose.pose.orientation
        yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])

    def cb_path(self, msg):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def cb_cmd_in(self, msg: Twist):
        """외부에서 들어온 cmd_vel을 NMPC로 필터링"""
        if self.pose is None:
            return

        v_ref = msg.linear.x
        w_ref = msg.angular.z

        # 초기 상태
        S0 = self.pose.copy()

        # 목표점 (없으면 현위치 유지)
        if self.path:
            gx, gy = self.path[-1]
            gth = math.atan2(gy - self.pose[1], gx - self.pose[0])
        else:
            gx, gy, gth = self.pose
        G = np.array([gx, gy, gth])

        # 레퍼런스 입력 (horizon 동안 동일)
        U_ref = np.tile([v_ref, w_ref], self.N)

        # 제약 bounds
        lbg = np.zeros(self.ng)
        ubg = np.zeros(self.ng)

        # warm start
        U_init = np.hstack([self.last_U[:, 1:], self.last_U[:, -1:]]) if self.last_U.size else np.zeros((2, self.N))
        S_init = np.zeros((3, self.N+1))
        S_init[:, 0] = S0
        w0 = np.concatenate([U_init.flatten(), S_init.flatten()])

        # Solve NMPC
        sol = self.solver(x0=w0,
                          lbx=self.lbx, ubx=self.ubx,
                          lbg=lbg, ubg=ubg,
                          p=np.concatenate([S0, G, U_ref]))
        w_opt = np.array(sol['x']).flatten()
        U_opt = w_opt[:self.nu * self.N].reshape(self.nu, self.N)

        v_cmd = float(U_opt[0, 0])
        w_cmd = float(U_opt[1, 0])
        self.last_U = U_opt

        out = Twist()
        out.linear.x = v_cmd
        out.angular.z = w_cmd
        self.cmd_pub.publish(out)

    def _build_opt(self):
        # States/controls
        x = ca.SX.sym('x'); y = ca.SX.sym('y'); th = ca.SX.sym('th')
        v = ca.SX.sym('v'); w = ca.SX.sym('w')
        s = ca.vertcat(x, y, th)
        u = ca.vertcat(v, w)

        def dyn(s, u):
            return ca.vertcat(
                s[0] + u[0]*ca.cos(s[2])*self.dt,
                s[1] + u[0]*ca.sin(s[2])*self.dt,
                s[2] + u[1]*self.dt
            )

        U = ca.SX.sym('U', 2, self.N)
        S = ca.SX.sym('S', 3, self.N+1)
        S0 = ca.SX.sym('S0', 3)
        G = ca.SX.sym('G', 3)
        U_ref = ca.SX.sym('U_ref', 2*self.N)

        g = [S[:, 0] - S0]
        J = 0

        for k in range(self.N):
            g.append(S[:, k+1] - dyn(S[:, k], U[:, k]))
            J += self.q_cmd*((U[0, k]-U_ref[2*k])**2 + (U[1, k]-U_ref[2*k+1])**2)

        # Terminal cost
        dx = S[0, self.N] - G[0]
        dy = S[1, self.N] - G[1]
        dth = ca.atan2(ca.sin(S[2, self.N]-G[2]), ca.cos(S[2, self.N]-G[2]))
        J += self.q_terminal*(dx*dx + dy*dy + dth*dth)

        wvars = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(S, -1, 1))
        g = ca.vertcat(*g)

        nlp = {'x': wvars, 'f': J, 'g': g, 'p': ca.vertcat(S0, G, U_ref)}
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.nu = 2; self.nx = 3
        self.ng = self.nx*(self.N+1)  # 초기+동역학 제약

        # bounds
        lbU = np.tile([0.0, -self.w_max], self.N)   # 전진만 허용
        ubU = np.tile([self.v_max, self.w_max], self.N)
        lbS = np.full(self.nx*(self.N+1), -np.inf)
        ubS = np.full(self.nx*(self.N+1),  np.inf)
        self.lbx = np.concatenate([lbU, lbS])
        self.ubx = np.concatenate([ubU, ubS])


def main():
    rospy.init_node("nmpc_filter_node")
    NMPCFilter()
    rospy.spin()


if __name__ == "__main__":
    main()
