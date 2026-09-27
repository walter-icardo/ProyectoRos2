#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class TurtlebotCtrl(Node):
    def __init__(self):
        super().__init__('TurtlebotCtrl')
        self.laser = LaserScan()
        self.odom = Odometry()

        # 0 = no recorrible/obstáculo, 1 = no visitado, 2 = visitado
        self.map = np.array([
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0],
            [0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
            [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,0,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
            [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ], dtype=int)

        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.callback_odom, 10)
        self.create_subscription(LaserScan, '/scan', self.callback_laser, 10)
        self.timer = self.create_timer(0.5, self.cmd_vel_pub)

        self.map_resolution = 4.0
        self.last_x = None
        self.last_y = None
        self.stuck_counter = 0
        self.turn_direction = 1
        self.last_percentage = -1
        self.target_cell = None
        self.target_lock = 0
        self.recent_targets = []

        self.get_logger().info('TurtlebotCtrl V2 iniciado')

    def get_yaw(self):
        q = self.odom.pose.pose.orientation
        return math.atan2(2.0*(q.w*q.z + q.x*q.y),
                          1.0 - 2.0*(q.y*q.y + q.z*q.z))

    def sector_distance(self, start_deg, end_deg):
        if not self.laser.ranges or self.laser.angle_increment == 0:
            return 10.0
        a = int((math.radians(start_deg)-self.laser.angle_min)/self.laser.angle_increment)
        b = int((math.radians(end_deg)-self.laser.angle_min)/self.laser.angle_increment)
        if a > b:
            a, b = b, a
        a = max(0, a); b = min(len(self.laser.ranges)-1, b)
        s = np.asarray(self.laser.ranges[a:b+1], dtype=float)
        s = s[np.isfinite(s) & (s > 0.05)]
        return float(np.min(s)) if s.size else 10.0

    def current_cell(self):
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        ix = -int(x*self.map_resolution) + self.map.shape[0]//2
        iy = -int(y*self.map_resolution) + self.map.shape[1]//2
        ix = max(1, min(ix, self.map.shape[0]-2))
        iy = max(1, min(iy, self.map.shape[1]-2))
        return ix, iy

    def find_target(self, cx, cy):
        candidates = []
        for i in range(1, self.map.shape[0]-1):
            for j in range(1, self.map.shape[1]-1):
                if self.map[i, j] != 1 or (i, j) in self.recent_targets:
                    continue
                d = math.hypot(i-cx, j-cy)
                if d > 8:
                    continue
                frontier = False
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i+di, j+dj
                        if 0 <= ni < 20 and 0 <= nj < 20 and self.map[ni, nj] == 2:
                            frontier = True
                if frontier:
                    candidates.append((d, i, j))

        # Si no quedan fronteras recientes, buscar cualquier 1 cercano.
        if not candidates:
            for i in range(1, self.map.shape[0]-1):
                for j in range(1, self.map.shape[1]-1):
                    if self.map[i, j] == 1:
                        d = math.hypot(i-cx, j-cy)
                        if d <= 8:
                            candidates.append((d, i, j))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        target = (candidates[0][1], candidates[0][2])
        self.recent_targets.append(target)
        if len(self.recent_targets) > 10:
            self.recent_targets.pop(0)
        return target

    def update_map(self):
        ix, iy = self.current_cell()
        if self.map[ix, iy] == 1:
            self.map[ix, iy] = 2
            n1 = np.count_nonzero(self.map == 1)
            n2 = np.count_nonzero(self.map == 2)
            total = n1 + n2
            if total:
                pct = 100.0*n2/total
                if int(pct) != self.last_percentage:
                    self.last_percentage = int(pct)
                    self.get_logger().info(f'Porcentaje recorrido: {pct:.2f}%')
                    self.get_logger().info('Mapa discreto:\n' + str(self.map))
        return ix, iy

    def cmd_vel_pub(self):
        cx, cy = self.update_map()

        front = self.sector_distance(-25, 25)
        front_left = self.sector_distance(25, 70)
        left = self.sector_distance(70, 120)
        front_right = self.sector_distance(-70, -25)
        right = self.sector_distance(-120, -70)

        obstacle = front < 0.70
        msg = TwistStamped()

        if obstacle:
            msg.twist.linear.x = 0.0
            left_space = front_left + left
            right_space = front_right + right
            self.turn_direction = 1 if left_space >= right_space else -1
            msg.twist.angular.z = (0.70 if front < 0.45 else 0.50) * self.turn_direction
            self.get_logger().info(
                'Obstáculo -> girando ' + ('izquierda' if self.turn_direction == 1 else 'derecha')
            )
        else:
            if self.target_cell is None or self.target_lock <= 0:
                self.target_cell = self.find_target(cx, cy)
                self.target_lock = 8
                if self.target_cell is not None:
                    self.get_logger().info(f'Nuevo objetivo de exploración: {self.target_cell}')
            else:
                self.target_lock -= 1

            if self.target_cell is not None:
                tx, ty = self.target_cell
                target_x = -(tx - self.map.shape[0]/2) / self.map_resolution
                target_y = -(ty - self.map.shape[1]/2) / self.map_resolution
                rx = self.odom.pose.pose.position.x
                ry = self.odom.pose.pose.position.y
                dx, dy = target_x-rx, target_y-ry
                dist = math.hypot(dx, dy)

                if dist < 0.30:
                    self.target_cell = None
                    self.target_lock = 0
                    msg.twist.linear.x = 0.12
                    msg.twist.angular.z = 0.0
                else:
                    error = math.atan2(dy, dx) - self.get_yaw()
                    while error > math.pi:
                        error -= 2*math.pi
                    while error < -math.pi:
                        error += 2*math.pi
                    msg.twist.angular.z = max(-0.60, min(0.60, error))
                    if abs(error) > 0.70:
                        msg.twist.linear.x = 0.05
                    elif abs(error) > 0.35:
                        msg.twist.linear.x = 0.12
                    else:
                        msg.twist.linear.x = 0.22
            else:
                msg.twist.linear.x = 0.15
                msg.twist.angular.z = 0.0

        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        if self.last_x is not None:
            movement = math.hypot(x-self.last_x, y-self.last_y)
            if not obstacle and msg.twist.linear.x > 0.10 and movement < 0.01:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        self.last_x, self.last_y = x, y

        if self.stuck_counter > 12:
            self.get_logger().warning('Robot posiblemente atascado -> recuperación')
            msg.twist.linear.x = -0.05
            msg.twist.angular.z = 0.70*self.turn_direction
            self.stuck_counter = 0
            self.target_cell = None
            self.target_lock = 0

        self.pub.publish(msg)

    def callback_laser(self, msg):
        self.laser = msg

    def callback_odom(self, msg):
        self.odom = msg


def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotCtrl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = TwistStamped()
        node.pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
