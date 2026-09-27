#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

import numpy as np
import math


class TurtlebotCtrl(Node):

    def __init__(self):
        super().__init__("TurtlebotCtrl")

        # ============================================================
        # SENSORES
        # ============================================================

        self.laser = LaserScan()
        self.odom = Odometry()

        # ============================================================
        # MAPA DISCRETO
        #
        # 0 = obstáculo / zona no recorrible
        # 1 = zona no visitada
        # 2 = zona visitada
        # ============================================================

        self.map = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=int)

        # ============================================================
        # ROS
        # ============================================================

        self.publish_cmd_vel = self.create_publisher(
            TwistStamped,
            "/cmd_vel",
            10
        )

        self.subscriber_odom = self.create_subscription(
            Odometry,
            "/odom",
            self.callback_odom,
            10
        )

        self.subscriber_laser = self.create_subscription(
            LaserScan,
            "/scan",
            self.callback_laser,
            10
        )

        self.timer = self.create_timer(
            0.5,
            self.cmd_vel_pub
        )

        # ============================================================
        # VARIABLES DE NAVEGACIÓN
        # ============================================================

        self.map_resolution = 4.0

        self.last_x = None
        self.last_y = None

        self.stuck_counter = 0

        self.turn_direction = 1

        self.last_percentage = -1

        # Objetivo de exploración actual
        self.target_cell = None

        # Mantener el objetivo durante varios ciclos
        self.target_lock = 0

        self.get_logger().info(
            "TurtlebotCtrl V2 iniciado"
        )

    # ================================================================
    # OBTENER YAW DESDE ODOMETRÍA
    # ================================================================

    def get_yaw(self):

        q = self.odom.pose.pose.orientation

        siny_cosp = 2.0 * (
            q.w * q.z +
            q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y * q.y +
            q.z * q.z
        )

        return math.atan2(
            siny_cosp,
            cosy_cosp
        )

    # ================================================================
    # DISTANCIA DE UN SECTOR DEL LIDAR
    # ================================================================

    def get_sector_distance(
        self,
        start_deg,
        end_deg
    ):

        if len(self.laser.ranges) == 0:
            return 10.0

        if self.laser.angle_increment == 0:
            return 10.0

        ranges = np.array(
            self.laser.ranges
        )

        start_rad = math.radians(
            start_deg
        )

        end_rad = math.radians(
            end_deg
        )

        start_index = int(
            (start_rad - self.laser.angle_min)
            / self.laser.angle_increment
        )

        end_index = int(
            (end_rad - self.laser.angle_min)
            / self.laser.angle_increment
        )

        if start_index > end_index:
            start_index, end_index = (
                end_index,
                start_index
            )

        start_index = max(
            0,
            start_index
        )

        end_index = min(
            len(ranges) - 1,
            end_index
        )

        sector = ranges[
            start_index:end_index + 1
        ]

        sector = sector[
            np.isfinite(sector)
        ]

        sector = sector[
            sector > 0.05
        ]

        if len(sector) == 0:
            return 10.0

        return float(
            np.min(sector)
        )

    # ================================================================
    # CONVERTIR ODOMETRÍA A CELDA DEL MAPA
    # ================================================================

    def get_current_cell(self):

        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y

        index_x = -int(
            x * self.map_resolution
        )

        index_y = -int(
            y * self.map_resolution
        )

        index_x += int(
            self.map.shape[0] / 2
        )

        index_y += int(
            self.map.shape[1] / 2
        )

        index_x = max(
            1,
            min(
                index_x,
                self.map.shape[0] - 2
            )
        )

        index_y = max(
            1,
            min(
                index_y,
                self.map.shape[1] - 2
            )
        )

        return index_x, index_y

    # ================================================================
    # BUSCAR CELDA NO VISITADA
    # ================================================================

    def find_exploration_target(
        self,
        current_x,
        current_y
    ):

        candidates = []

        # Buscar celdas no visitadas
        # alrededor de la zona actual.
        for i in range(1, self.map.shape[0] - 1):

            for j in range(1, self.map.shape[1] - 1):

                if self.map[i][j] != 1:
                    continue

                distance = math.sqrt(
                    (i - current_x) ** 2 +
                    (j - current_y) ** 2
                )

                # No elegir objetivos demasiado lejanos.
                if distance > 8:
                    continue

                # ====================================================
                # FRONTIER
                #
                # Preferimos una celda 1 que esté junto
                # a una celda que ya fue visitada (2).
                # ====================================================

                frontier = False

                for di in [-1, 0, 1]:

                    for dj in [-1, 0, 1]:

                        if di == 0 and dj == 0:
                            continue

                        ni = i + di
                        nj = j + dj

                        if (
                            ni >= 0
                            and ni < self.map.shape[0]
                            and nj >= 0
                            and nj < self.map.shape[1]
                        ):

                            if self.map[ni][nj] == 2:
                                frontier = True

                if not frontier:
                    continue

                # ====================================================
                # PREFERIR OBJETIVOS CERCANOS
                # ====================================================

                score = distance

                candidates.append(
                    (
                        score,
                        i,
                        j
                    )
                )

        if len(candidates) == 0:

            return None

        candidates.sort(
            key=lambda item: item[0]
        )

        return (
            candidates[0][1],
            candidates[0][2]
        )

    # ================================================================
    # ACTUALIZAR MAPA
    # ================================================================

    def update_map(self):

        index_x, index_y = (
            self.get_current_cell()
        )

        if self.map[index_x][index_y] == 1:

            self.map[index_x][index_y] = 2

            total = (
                np.count_nonzero(
                    self.map == 1
                )
                +
                np.count_nonzero(
                    self.map == 2
                )
            )

            visited = np.count_nonzero(
                self.map == 2
            )

            if total > 0:

                percentage = (
                    100.0 *
                    float(visited) /
                    float(total)
                )

                if int(percentage) != self.last_percentage:

                    self.last_percentage = int(
                        percentage
                    )

                    self.get_logger().info(
                        "Porcentaje recorrido: "
                        +
                        str(round(
                            percentage,
                            2
                        ))
                        +
                        "%"
                    )

                    self.get_logger().info(
                        "Mapa discreto:"
                    )

                    self.get_logger().info(
                        "\n" + str(self.map)
                    )

        return index_x, index_y

    # ================================================================
    # CONTROL PRINCIPAL
    # ================================================================

    def cmd_vel_pub(self):

        # ============================================================
        # 1. ACTUALIZAR MAPA
        # ============================================================

        current_x, current_y = (
            self.update_map()
        )

        # ============================================================
        # 2. LIDAR
        # ============================================================

        frente = self.get_sector_distance(
            -25,
            25
        )

        frente_izquierda = self.get_sector_distance(
            25,
            70
        )

        izquierda = self.get_sector_distance(
            70,
            120
        )

        frente_derecha = self.get_sector_distance(
            -70,
            -25
        )

        derecha = self.get_sector_distance(
            -120,
            -70
        )

        # ============================================================
        # 3. DISTANCIAS DE SEGURIDAD
        # ============================================================

        distancia_peligro = 0.45
        distancia_seguridad = 0.70

        obstaculo_frente = (
            frente < distancia_seguridad
        )

        # ============================================================
        # 4. MENSAJE
        # ============================================================

        msg = TwistStamped()

        # ============================================================
        # 5. EVITAR OBSTÁCULOS
        # ============================================================

        if obstaculo_frente:

            msg.twist.linear.x = 0.0

            espacio_izquierda = (
                frente_izquierda +
                izquierda
            )

            espacio_derecha = (
                frente_derecha +
                derecha
            )

            if espacio_izquierda >= espacio_derecha:

                self.turn_direction = 1

            else:

                self.turn_direction = -1

            if frente < distancia_peligro:

                msg.twist.angular.z = (
                    0.70 *
                    self.turn_direction
                )

            else:

                msg.twist.angular.z = (
                    0.50 *
                    self.turn_direction
                )

            # Al detectar un obstáculo,
            # el objetivo puede mantenerse,
            # pero no contamos este giro
            # como atasco.

            self.get_logger().info(
                "Obstáculo -> girando "
                +
                (
                    "izquierda"
                    if self.turn_direction == 1
                    else "derecha"
                )
            )

        else:

            # ========================================================
            # 6. BUSCAR NUEVO OBJETIVO DE EXPLORACIÓN
            # ========================================================

            if (
                self.target_cell is None
                or self.target_lock <= 0
            ):

                self.target_cell = (
                    self.find_exploration_target(
                        current_x,
                        current_y
                    )
                )

                self.target_lock = 8

                if self.target_cell is not None:

                    self.get_logger().info(
                        "Nuevo objetivo de exploracion: "
                        +
                        str(self.target_cell)
                    )

            else:

                self.target_lock -= 1

            # ========================================================
            # 7. SI ENCONTRAMOS UNA CELDA NO VISITADA
            # ========================================================

            if self.target_cell is not None:

                target_x = self.target_cell[0]
                target_y = self.target_cell[1]

                # Convertir celda del mapa a coordenadas
                # aproximadas del mundo.

                world_target_x = -(
                    target_x -
                    self.map.shape[0] / 2
                ) / self.map_resolution

                world_target_y = -(
                    target_y -
                    self.map.shape[1] / 2
                ) / self.map_resolution

                robot_x = (
                    self.odom.pose.pose.position.x
                )

                robot_y = (
                    self.odom.pose.pose.position.y
                )

                dx = (
                    world_target_x -
                    robot_x
                )

                dy = (
                    world_target_y -
                    robot_y
                )

                target_distance = math.sqrt(
                    dx * dx +
                    dy * dy
                )

                # ====================================================
                # 8. SI LLEGAMOS AL OBJETIVO
                # ====================================================

                if target_distance < 0.30:

                    self.target_cell = None

                    self.target_lock = 0

                    msg.twist.linear.x = 0.12
                    msg.twist.angular.z = 0.0

                else:

                    yaw = self.get_yaw()

                    target_angle = math.atan2(
                        dy,
                        dx
                    )

                    angle_error = (
                        target_angle -
                        yaw
                    )

                    # Normalizar a [-pi, pi]

                    while angle_error > math.pi:

                        angle_error -= (
                            2.0 * math.pi
                        )

                    while angle_error < -math.pi:

                        angle_error += (
                            2.0 * math.pi
                        )

                    # =================================================
                    # CONTROL DE ORIENTACIÓN
                    # =================================================

                    angular = (
                        1.0 *
                        angle_error
                    )

                    angular = max(
                        -0.60,
                        min(
                            angular,
                            0.60
                        )
                    )

                    # =================================================
                    # VELOCIDAD
                    # =================================================

                    if abs(angle_error) > 0.70:

                        linear = 0.05

                    elif abs(angle_error) > 0.35:

                        linear = 0.12

                    else:

                        linear = 0.22

                    msg.twist.linear.x = linear

                    msg.twist.angular.z = angular

            else:

                # ====================================================
                # 9. SI NO QUEDAN OBJETIVOS
                # ====================================================

                msg.twist.linear.x = 0.15
                msg.twist.angular.z = 0.0

        # ============================================================
        # 10. DETECCIÓN DE ATASCO
        # ============================================================

        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y

        if self.last_x is not None:

            movimiento = math.sqrt(
                (x - self.last_x) ** 2 +
                (y - self.last_y) ** 2
            )

            # Solo considerar atasco cuando:
            # - no hay obstáculo delante
            # - el robot debería avanzar
            # - prácticamente no cambia de posición

            if (
                not obstaculo_frente
                and msg.twist.linear.x > 0.10
                and movimiento < 0.01
            ):

                self.stuck_counter += 1

            else:

                self.stuck_counter = 0

        self.last_x = x
        self.last_y = y

        # ============================================================
        # 11. RECUPERACIÓN
        # ============================================================

        if self.stuck_counter > 12:

            self.get_logger().warning(
                "Robot posiblemente atascado -> recuperacion"
            )

            msg.twist.linear.x = -0.05

            msg.twist.angular.z = (
                0.70 *
                self.turn_direction
            )

            self.stuck_counter = 0

            self.target_cell = None

            self.target_lock = 0

        # ============================================================
        # 12. PUBLICAR
        # ============================================================

        self.publish_cmd_vel.publish(
            msg
        )

    # ================================================================
    # CALLBACK LIDAR
    # ================================================================

    def callback_laser(self, msg):

        self.laser = msg

    # ================================================================
    # CALLBACK ODOMETRÍA
    # ================================================================

    def callback_odom(self, msg):

        self.odom = msg


# ====================================================================
# MAIN
# ====================================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = TurtlebotCtrl()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        stop_msg = TwistStamped()

        stop_msg.twist.linear.x = 0.0
        stop_msg.twist.angular.z = 0.0

        node.publish_cmd_vel.publish(
            stop_msg
        )

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()
