import csv
import os
from math import cos, sin, atan2

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp_integrate

from dynopy.datahandling.objects import GroundTruth, Input, Measurement, get_noisy_measurement
from dynopy.estimationtools.importance_sampling import SIS, bootstrap

# from channel_filter import ChannelFilter
# from data_objects import InformationEstimate, Measurement
# from estimation_tools import get_noisy_measurement
# import information_filter as IF


class Workspace:
    def __init__(self, boundary_coordinates, obstacle_coordinates, landmarks):
        self.boundary_coordinates = boundary_coordinates
        self.obstacle_coordinates = obstacle_coordinates
        self.landmarks = landmarks

        self.obstacles = list()
        self.robots = list()

        i = 1
        for obstacle in self.obstacle_coordinates:
            self.obstacles.append(Obstacle('WO ' + str(i), obstacle))
            i += 1

        x_coordinates = [i[0] for i in self.boundary_coordinates]
        y_coordinates = [i[1] for i in self.boundary_coordinates]

        self.x_bounds = (min(x_coordinates), max(x_coordinates))
        self.y_bounds = (min(y_coordinates), max(y_coordinates))

    def plot(self):
        """
        Plots the environment boundaries as a black dashed line, the polygon obstacles, and the robot starting position
        and goal.
        :return: none
        """
        x_coordinates = [i[0] for i in self.boundary_coordinates]
        x_coordinates.append(self.boundary_coordinates[0][0])

        y_coordinates = [i[1] for i in self.boundary_coordinates]
        y_coordinates.append(self.boundary_coordinates[0][1])

        plt.plot(x_coordinates, y_coordinates, 'k-')

        for obstacle in self.obstacles:
            obstacle.plot()

        for landmark in self.landmarks:
            landmark.plot()

        for robot in self.robots:
            robot.plot_initial()

        x_min = self.x_bounds[0]
        x_max = self.x_bounds[1] + 1
        y_min = self.y_bounds[0]
        y_max = self.y_bounds[1] + 1

        plt.axis('equal')
        plt.xticks(range(x_min, x_max, 10))
        plt.yticks(range(y_min, y_max, 10))


class Obstacle:
    def __init__(self, the_name, the_coordinates):
        self.name = the_name
        self.vertices = the_coordinates

    def plot(self):
        """
        Plots the edges of the polygon obstacle in a 2-D represented workspace.
        :return: none
        """
        x_coordinates = [i[0] for i in self.vertices]
        x_coordinates.append(self.vertices[0][0])

        y_coordinates = [i[1] for i in self.vertices]
        y_coordinates.append(self.vertices[0][1])

        plt.plot(x_coordinates, y_coordinates)


class Landmark:
    def __init__(self, name, x, y, model):
        self.name = name
        self.vertices = (x, y)
        self.type = model

        self.range_measurements = True
        self.bearing_measurements = True

    def plot(self):
        """
        Plots as a square
        :return: none
        """
        plt.plot(self.vertices[0], self.vertices[1], 's')

    def get_x(self):
        return self.vertices[0]

    def get_y(self):
        return self.vertices[1]

    def return_measurement(self, state):

        x1 = state.x_1
        y1 = state.x_2
        k = state.step

        x2 = self.vertices[0]
        y2 = self.vertices[1]

        if self.range_measurements and self.bearing_measurements:
            r = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1 / 2)
            b = atan2(y2 - y1, x2 - x1)

        elif self.range_measurements:
            r = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1 / 2)
            b = None

        elif self.bearing_measurements:
            r = None
            b = atan2(y2 - y1, x2 - x1)

        else:
            r = None
            b = None

        return r, b

    @staticmethod
    def create_from_dict(settings):

        return Landmark(
            settings['name'],
            float(settings['x']),
            float(settings['y']),
            settings['model'])


class TwoDimensionalRobot:
    def __init__(self, settings: dict, state: dict):
        self.settings = settings
        self.state = state

        self.name = settings.get('name')
        self.color = settings.get('color')
        self.state_names = list(state.keys())

        self.current_measurement_step = 0
        self.input_list = []
        self.measurement_list = []

    def plot_initial(self):
        """
        Plots the position of the robot as an x
        :return: none
        """
        x_i = self.state.get(self.state_names[0])
        y_i = self.state.get(self.state_names[1])
        plt.plot(x_i, y_i, 'x', color=self.color)

    def return_state_array(self):
        """
        converts and returns the robot's state into a numpy array
        :return: n x 1 numpy array of current state variables
        """
        state_list = list(self.state.values())
        return np.array(state_list).reshape((-1, 1))

    def return_state_list(self):
        """
        converts and returns the robot's state into a numpy array
        :return: n x 1 numpy array of current state variables
        """
        return list(self.state.values())

    def read_inputs(self, input_file: str):
        with open(input_file, 'r', encoding='utf8') as fin:
            reader = csv.DictReader(fin, skipinitialspace=True, delimiter=',')

            for row in reader:
                u = Input.create_from_dict(row)
                self.input_list.append(u)


class DifferentialDrive(TwoDimensionalRobot):
    def __init__(self, settings: dict, state: dict, workspace):
        super().__init__(settings, state)

        self.name = settings.get('name')
        self.color = settings.get('color')
        self.axel_length = settings.get('axel_length')
        self.wheel_radius = settings.get('wheel_radius')
        self.state_names = list(state.keys())
        self.workspace = workspace

        self.Q = np.array([[]])
        self.R = np.array([[]])

        self.current_measurement_step = 0
        self.input_list = []
        self.measurement_list = []
        self.ground_truth = [GroundTruth.create_from_list(0, list(state.values()))]
        self.perfect_measurements = []
        self.particle_set = []
        self.particle_set_list = []

    def step_sis(self, z, u, dt):
        self.particle_set = SIS(self.particle_set, z, self, u, dt)
        self.particle_set_list.append(self.particle_set)

    def step_bootstrap(self, z, u, dt):
        self.particle_set = bootstrap(self.particle_set, z, self, u, dt)
        self.particle_set_list.append(self.particle_set)

    def initialize_particle_set(self, particle_set):
        self.particle_set = particle_set
        self.particle_set_list.append(self.particle_set)

    def get_ground_truth(self, Q_true, dt: float):
        """
        Will take a list of input objects and run them through the system dynamics to get
        :param dt:
        :return:
        """
        inputs = self.input_list

        for u in inputs:
            x_k0 = np.squeeze(self.ground_truth[-1].return_data_array())
            k0 = self.ground_truth[-1].step

            if u.step != k0:
                print("Error: ground truth to input step misalignment")

            k1 = k0 + 1
            sol = sp_integrate.solve_ivp(self.dynamics_ode, (k0*dt, k1*dt), x_k0,
                                         args=(u, self.axel_length, self.wheel_radius))

            x_k1 = sol.y[:, -1]
            self.ground_truth.append(GroundTruth.create_from_list(k1, x_k1, Q_true))

    def get_measurement(self, true_measurement: Measurement):
        noisy_measurement = get_noisy_measurement(self.R, true_measurement)
        self.measurement_list.append(noisy_measurement)
        self.current_measurement_step += 1
        return noisy_measurement

    def get_perfect_measurements(self):

        for state in self.ground_truth:
            measurement_values = []
            output_names = []

            for landmark in self.workspace.landmarks:
                r, b = landmark.return_measurement(state)

                if r:
                    measurement_values.append(r)
                    output_names.append(landmark.name + '_range')
                if b:
                    measurement_values.append(b)
                    output_names.append(landmark.name + '_bearing')

            self.perfect_measurements.append(Measurement.create_from_list(state.step, measurement_values, output_names))

    def create_noisy_measurements(self):
        for meas in self.perfect_measurements:
            self.measurement_list.append(self.get_measurement(meas))

    def run_prediction_update(self, x_k0, u, dt: float):
        """
        given an initial state and an input, this function runs the full system dynamics prediction.
        :param x_k0: initial state [StateEstimate object]
        :param u: input [Input object]
        :param dt: time step
        :return: StateEstimate object for the next time step.
        """
        k0 = x_k0.step
        k1 = k0 + 1
        sol = sp_integrate.solve_ivp(self.dynamics_ode, (k0 * dt, k1 * dt), x_k0.return_data_list(),
                                     args=(u, self.axel_length, self.wheel_radius))
        x_k1 = sol.y[:, -1]
        return x_k1.reshape(-1, 1)

    def get_predicted_measurement(self, state_object):
        """

        :param state_object:
        :return:
        """
        state = state_object.return_data_list()
        e = state[0]
        n = state[1]
        theta = state[2]

        measurement_values = []
        output_names = []

        for landmark in self.workspace.landmarks:
            r, b = landmark.return_measurement(state_object)

            if r:
                measurement_values.append(r)
                output_names.append(landmark.name + '_range')
            if b:
                measurement_values.append(b)
                output_names.append(landmark.name + '_bearing')

        return Measurement.create_from_list(state_object.step, measurement_values, output_names)

    @staticmethod
    def dynamics_ode(t, x, u, L, r):
        u_r = u.u_1
        u_l = u.u_2

        x3 = x[2]
        x4 = x[3]
        x5 = x[4]

        x1_dot = r/2*(x4 + x5)*cos(x3)
        x2_dot = r/2*(x4 + x5)*sin(x3)
        x3_dot = r/L * (x4 - x5)
        x4_dot = u_r
        x5_dot = u_l

        x_dot = np.array([x1_dot, x2_dot, x3_dot, x4_dot, x5_dot])
        return x_dot

# class Seeker(TwoDimensionalRobot):
#     def __init__(self, name: str, state: dict, color: str, R: np.ndarray):
#         super().__init__(name, state, color)
#         self.R = R
#
#         # TODO: this is a bit static
#         self.i_init = np.array([
#             [0],
#             [0]
#         ])
#         self.I_init = np.array([
#             [0, 0],
#             [0, 0]
#         ])
#         self.information_list = [InformationEstimate.create_from_array(0, self.i_init, self.I_init)]
#         self.channel_filter_dict = {}
#
#     def plot_measurements(self):
#         x_coordinates = [x.y_1 for x in self.measurement_list]
#         y_coordinates = [y.y_2 for y in self.measurement_list]
#
#         plt.plot(x_coordinates, y_coordinates, 'o', mfc=self.color, markersize=2, mec='None', alpha=0.5)
#
#     def get_measurement(self, true_measurement: Measurement):
#         noisy_measurement = get_noisy_measurement(self.R, true_measurement)
#         self.measurement_list.append(noisy_measurement)
#         self.current_measurement_step += 1
#         return noisy_measurement
#
#     def run_filter(self, target):
#         current_step = self.information_list[-1].step
#         true_measurement = next((x for x in target.truth_model.true_measurements if x.step == current_step + 1), None)
#         y = self.get_measurement(true_measurement)
#         self.information_list.append(IF.run(target.state_space, self.information_list[-1], y, self.R))
#
#     def create_channel_filter(self, robot_j, target):
#         """
#         adds a new channel filter for a single target to the
#         :param robot_j:
#         :param target:
#         :return:
#         """
#         channel_filter = ChannelFilter(self, robot_j, target)
#         self.channel_filter_dict[robot_j.name] = channel_filter
#
#     def send_update(self):
#         for cf in self.channel_filter_dict.values():
#             cf.update_and_send()
#
#     def receive_update(self):
#         for cf in self.channel_filter_dict.values():
#             cf.receive_and_update()
#
#     def fuse_data(self):
#         # TODO: should be a summation in here for more sensors, but works as is
#         y_k1_p = self.information_list[-1].return_data_array()
#         Y_k1_p = self.information_list[-1].return_information_matrix()
#
#         yj_novel_sum = 0
#         Yj_novel_sum = 0
#
#         for cf in self.channel_filter_dict.values():
#             yj_novel_sum += cf.yj_novel
#             Yj_novel_sum += cf.Yj_novel
#
#         y_k1_fused = y_k1_p + yj_novel_sum
#         Y_k1_fused = Y_k1_p + Yj_novel_sum
#         self.information_list[-1].update(y_k1_fused, Y_k1_fused)
#
#
# class Hider(TwoDimensionalRobot):
#     def __init__(self, name: str, state: dict, color: str, state_space):
#         super().__init__(name, state, color)
#         self.state_space = state_space
