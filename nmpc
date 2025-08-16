# =============================
# File: nmpc_controller_node.py
# Description: Standalone NMPC controller for TurtleBot3 (ROS1 Noetic)
# - Subscribes: /state (PoseWithCovarianceStamped) or ~state_topic param (default: /amcl_pose or /ukf/pose)
# - Subscribes: /waypoints (nav_msgs/Path)
# - Publishes: /cmd_vel (geometry_msgs/Twist)
# - Model: unicycle (x, y, yaw), controls: v (m/s), w (rad/s)
# - Solver: CasADi + IPOPT
# =============================

#!/usr/bin/env python3
import rospy
import numpy as np
import casadi as ca
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Path
import tf.transformations as tft
import math

class NMPCController:
    def __init__(self):
        # Parameters
        self.frame_map = rospy.get_param('~frame_map', 'map')
        self.state_topic = rospy.get_param('~state_topic', 'amcl_pose')  # e.g., amcl_pose or ukf/pose
        self.cmd_topic = rospy.get_param('~cmd_topic', 'cmd_vel')
        self.path_topic = rospy.get_param('~path_topic', 'waypoints')

        self.N = int(rospy.get_param('~horizon', 15))
        self.dt = float(rospy.get_param('~dt', 0.1))

        # TurtleBot3 Burger limits (adjust as needed)
        self.v_max = float(rospy.get_param('~v_max', 0.22))
        self.w_max = float(rospy.get_param('~w_max', 2.84))
        self.dv_max = float(rospy.get_param('~dv_max', 0.1))   # m/s per step
        self.dw_max = float(rospy.get_param('~dw_max', 0.5))   # rad/s per step

        # Weights
        self.q_xy = float(rospy.get_param('~q_xy', 10.0))
        self.q_yaw = float(rospy.get_param('~q_yaw', 1.0))
        self.r_v = float(rospy.get_param('~r_v', 0.1))
        self.r_w = float(rospy.get_param('~r_w', 0.1))
        self.q_terminal = float(rospy.get_param('~q_terminal', 5.0))

        # Goal handling
        self.goal_tol_xy = float(rospy.get_param('~goal_tolerance_xy', 0.05))
        self.goal_hold_yaw = bool(rospy.get_param('~goal_hold_yaw', True))

        # State holders
        self.pose = None  # np.array([x,y,yaw])
        self.path = []    # list of (x,y)
        self.goal_idx = 0
        self.last_U = np.zeros((2, self.N))  # warm-start controls

        # ROS I/O
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)
        rospy.Subscriber(self.state_topic, PoseWithCovarianceStamped, self.cb_pose)
        rospy.Subscriber(self.path_topic, Path, self.cb_path)

        # Build optimizer
        self._build_opt()
        self.timer = rospy.Timer(rospy.Duration(self.dt), self._on_timer)
        rospy.loginfo("NMPC ready: state=%s path=%s cmd=%s", self.state_topic, self.path_topic, self.cmd_topic)

    def cb_pose(self, msg: PoseWithCovarianceStamped):
        q = msg.pose.pose.orientation
        yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])

    def cb_path(self, msg: Path):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.goal_idx = 0

    def _active_goal(self):
        if not self.path:
            return None
        # advance to next waypoint once close
        while self.goal_idx < len(self.path):
            gx, gy = self.path[self.goal_idx]
            if self.pose is None:
                break
            if np.hypot(gx - self.pose[0], gy - self.pose[1]) < self.goal_tol_xy:
                self.goal_idx += 1
            else:
                break
        # final goal if finished
        if self.goal_idx >= len(self.path):
            return self.path[-1]
        return self.path[self.goal_idx]

    # ----------------- NMPC setup -----------------
    def _build_opt(self):
        # States and controls
        x = ca.SX.sym('x'); y = ca.SX.sym('y'); th = ca.SX.sym('th')
        v = ca.SX.sym('v'); w = ca.SX.sym('w')
        s = ca.vertcat(x,y,th)
        u = ca.vertcat(v,w)

        def dyn(s,u):
            x,y,th = s[0], s[1], s[2]
            v,w = u[0], u[1]
            x_next = x + v*ca.cos(th)*self.dt
            y_next = y + v*ca.sin(th)*self.dt
            th_next = th + w*self.dt
            return ca.vertcat(x_next, y_next, th_next)

        # Decision vars
        U = ca.SX.sym('U', 2, self.N)
        S = ca.SX.sym('S', 3, self.N+1)
        S0 = ca.SX.sym('S0', 3)
        G = ca.SX.sym('G', 3)  # goal [gx, gy, gth]
        U_prev = ca.SX.sym('U_prev', 2)  # previous applied control for slew

        g = []
        g.append(S[:,0]-S0)
        J = 0
        for k in range(self.N):
            # dynamics constraint
            g.append(S[:,k+1]-dyn(S[:,k], U[:,k]))
            # tracking cost
            dx = S[0,k]-G[0]
            dy = S[1,k]-G[1]
            dth = ca.atan2(ca.sin(S[2,k]-G[2]), ca.cos(S[2,k]-G[2]))
            J += self.q_xy*(dx*dx + dy*dy) + self.q_yaw*(dth*dth) \
                 + self.r_v*(U[0,k]*U[0,k]) + self.r_w*(U[1,k]*U[1,k])
            # slew constraints relative to prev step (hard bounds)
            if k == 0:
                g.append( U[:,k] - U_prev )
            else:
                g.append( U[:,k] - U[:,k-1] )
        # terminal cost
        dxT = S[0,self.N]-G[0]
        dyT = S[1,self.N]-G[1]
        dthT = ca.atan2(ca.sin(S[2,self.N]-G[2]), ca.cos(S[2,self.N]-G[2]))
        J += self.q_terminal*self.q_xy*(dxT*dxT + dyT*dyT) + self.q_terminal*self.q_yaw*(dthT*dthT)

        # Variable vector
        wvars = ca.vertcat(ca.reshape(U,-1,1), ca.reshape(S,-1,1))
        g = ca.vertcat(*g)

        nlp = {'x': wvars, 'f': J, 'g': g, 'p': ca.vertcat(S0, G, U_prev)}
        opts = {'ipopt.print_level':0, 'print_time':0}
        self.solver = ca.nlpsol('solver','ipopt', nlp, opts)

        # dims for unpacking
        self.nu = 2; self.nx = 3
        self.nw = self.nu*self.N + self.nx*(self.N+1)
        self.ng_dyn = self.nx*(self.N+1)
        self.ng_slew = self.N*2  # two per step (v,w) because of slew constraints appended per k
        self.ng = self.ng_dyn + self.ng_slew

        # Bounds on decision vars
        lbU = np.tile([-self.v_max, -self.w_max], self.N)
        ubU = np.tile([ self.v_max,  self.w_max], self.N)
        lbS = np.full(self.nx*(self.N+1), -np.inf)
        ubS = np.full(self.nx*(self.N+1),  np.inf)
        self.lbx = np.concatenate([lbU, lbS])
        self.ubx = np.concatenate([ubU, ubS])

    # ----------------- Solve & act -----------------
    def _on_timer(self, _):
        if self.pose is None or not self.path:
            return
        gx, gy = self._active_goal()
        # reference heading: face next goal; keep final yaw if at last point
        if self.goal_idx >= len(self.path)-1 and self.goal_hold_yaw:
            gth = self.pose[2]  # hold current yaw at final goal
        else:
            gth = math.atan2(gy - self.pose[1], gx - self.pose[0])

        S0 = self.pose.copy()
        G = np.array([gx, gy, gth])
        U_prev = self.last_U[:,0] if self.last_U.size else np.zeros(2)

        # Build constraint bounds (dynamics = 0; slew in [-dv_max,dv_max],[-dw_max,dw_max])
        lbg_dyn = np.zeros(self.ng_dyn)
        ubg_dyn = np.zeros(self.ng_dyn)
        lbg_slew = np.tile([-self.dv_max, -self.dw_max], self.N)
        ubg_slew = np.tile([ self.dv_max,  self.dw_max], self.N)
        lbg = np.concatenate([lbg_dyn, lbg_slew])
        ubg = np.concatenate([ubg_dyn, ubg_slew])

        # Warm start: shift previous control sequence forward
        U_init = np.hstack([self.last_U[:,1:], self.last_U[:,-1:]]) if self.last_U.size else np.zeros((2,self.N))
        S_init = np.zeros((3,self.N+1)); S_init[:,0] = S0
        w0 = np.concatenate([U_init.flatten(), S_init.flatten()])

        sol = self.solver(x0=w0, lbx=self.lbx, ubx=self.ubx, lbg=lbg, ubg=ubg,
                          p=np.concatenate([S0, G, U_prev]))
        w_opt = np.array(sol['x']).flatten()
        U_opt = w_opt[:self.nu*self.N].reshape(self.nu, self.N)

        # Apply first control
        v_cmd = float(U_opt[0,0])
        w_cmd = float(U_opt[1,0])
        self.last_U = U_opt

        # Stop if at final goal
        final = np.array(self.path[-1])
        if np.hypot(final[0]-self.pose[0], final[1]-self.pose[1]) < self.goal_tol_xy:
            v_cmd = 0.0; w_cmd = 0.0

        msg = Twist(); msg.linear.x = v_cmd; msg.angular.z = w_cmd
        self.cmd_pub.publish(msg)


def main():
    rospy.init_node('nmpc_controller_node')
    NMPCController()
    rospy.spin()

if __name__ == '__main__':
    main()

# =============================
# File: launch/nmpc_only.launch (example)
#
# <launch>
#   <node pkg="your_pkg" type="nmpc_controller_node.py" name="nmpc" output="screen">
#     <param name="state_topic" value="ukf/pose"/>
#     <param name="path_topic" value="waypoints"/>
#     <param name="horizon" value="15"/>
#     <param name="dt" value="0.1"/>
#     <param name="v_max" value="0.22"/>
#     <param name="w_max" value="2.84"/>
#   </node>
# </launch>
