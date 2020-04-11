from math import sqrt

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from dynopy.estimationtools.estimation_tools import monte_carlo_sample


class GroundTruth:
    """
    data object to hold all information pertinent to the ground truth at a given time step
    """
    def __init__(self, step: int, state_1: float, state_2: float, state_3: float, state_4: float, state_5: float,
                 Q=None, state_names=None):
        self.step = step
        self.x_1 = state_1
        self.x_2 = state_2
        self.x_3 = state_3
        self.x_4 = state_4
        self.x_5 = state_5
        self.Q = Q
        self.state_names = state_names

        if isinstance(self.Q, np.ndarray):
            noisy_state = monte_carlo_sample(self.return_data_array(), self.Q)
            self.update_values(noisy_state)

    def update_values(self, new_values: np.ndarray):
        new_values = np.squeeze(new_values)

        self.x_1 = new_values[0]
        self.x_2 = new_values[1]
        self.x_3 = new_values[2]
        self.x_4 = new_values[3]
        self.x_5 = new_values[4]

    @staticmethod
    def create_from_array(step: int, state_array: np.ndarray, Q=None):
        """
        fast way to create a StateEstimate object from a numpy array of the state
        :param step: time step associated with the data
        :param state_array: numpy array with ordered state values
        :param Q: process noise to be applied
        :return: StateEstimate object
        """
        print(state_array)
        print(type(state_array))
        if state_array.shape[1]:
            # reduces 2D state array down to a single dimension
            state_array = state_array.squeeze()

        return GroundTruth(
            step,
            state_array[0],
            state_array[1],
            state_array[2],
            state_array[3],
            state_array[4],
            Q
        )

    @staticmethod
    def create_from_list(step: int, state_list: list, Q=None):
        """
        fast way to create a StateEstimate object from a numpy array of the state
        :param step: time step associated with the data
        :param state_list: list with ordered state values
        :param Q: process noise to be applied
        :return: StateEstimate object
        """
        return GroundTruth(
            step,
            state_list[0],
            state_list[1],
            state_list[2],
            state_list[3],
            state_list[4],
            Q
        )

    def return_data_array(self):
        """
        provides intuitive and usefully formatted access to the state estimate data.
        :return: the state estimate data as an order 2D numpy array
        """
        return np.array([
            [self.x_1],
            [self.x_2],
            [self.x_3],
            [self.x_4],
            [self.x_5]
        ])

    def return_data_list(self):
        """
        provides intuitive and usefully formatted access to the state estimate data.
        :return: the state estimate data as an order list
        """
        return [self.x_1, self.x_2, self.x_3, self.x_4, self.x_5]

    def plot(self):
        plt.plot(self.x_1, self.x_2)


class InformationEstimate:
    def __init__(self, step: int, info_1: float, info_2: float, info_matrix: np.ndarray):
        self.step = step
        self.i_1 = info_1
        self.i_2 = info_2
        self.I_matrix = info_matrix

    @staticmethod
    def create_from_array(step: int, info_array: np.ndarray, info_matrix: np.ndarray):
        if info_array.shape[1]:
            # reduces 2D state array down to a single dimension
            info_array = info_array.squeeze()

        return InformationEstimate(
            step,
            info_array[0],
            info_array[1],
            info_matrix
        )

    def return_data_array(self):
        return np.array([
            [self.i_1],
            [self.i_2],
        ])

    def return_information_matrix(self):
        return self.I_matrix

    def get_state_estimate(self):
        covariance = np.linalg.inv(self.I_matrix)
        state_array = covariance @ self.return_data_array()
        return StateEstimate.create_from_array(self.step, state_array, covariance)

    def update(self, info_array: np.ndarray, info_matrix: np.ndarray):
        self.i_1, self.i_2 = info_array[0][0], info_array[1][0]
        self.I_matrix = info_matrix


class StateEstimate:
    """
    data object to hold all information pertinent to the state estimate at a given time step
    """
    def __init__(self, step: int, state_1: float, state_2: float, state_3: float, state_4: float, state_5: float,
                 covariance: np.ndarray, state_names=None):
        self.step = step
        self.x_1 = state_1
        self.x_2 = state_2
        self.x_3 = state_3
        self.x_4 = state_4
        self.x_5 = state_5
        self.state_names = state_names

        self.covariance = covariance
        self.x1_2sigma = 2 * float(covariance[0][0]) ** 0.5
        self.x2_2sigma = 2 * float(covariance[1][1]) ** 0.5
        self.x3_2sigma = 2 * float(covariance[0][0]) ** 0.5
        self.x4_2sigma = 2 * float(covariance[1][1]) ** 0.5
        self.x5_2sigma = 2 * float(covariance[0][0]) ** 0.5


    @staticmethod
    def create_from_array(step: int, state_array: np.ndarray, covariance: np.ndarray):
        """
        fast way to create a StateEstimate object from a numpy array of the state
        :param step: time step associated with the data
        :param state_array: numpy array with ordered state values
        :param covariance: numpy array with the estimate covariance
        :return: StateEstimate object
        """
        if state_array.shape[1]:
            # reduces 2D state array down to a single dimension
            state_array = state_array.squeeze()

        return StateEstimate(
            step,
            state_array[0],
            state_array[1],
            state_array[2],
            state_array[3],
            state_array[4],
            covariance
        )

    @staticmethod
    def create_from_list(step: int, state_list: list, covariance):
        """
        fast way to create a StateEstimate object from a numpy array of the state
        :param step: time step associated with the data
        :param state_list: list with ordered state values
        :param state_list: list with covariance #TODO: decide if this is a np array
        :return: StateEstimate object
        """
        return StateEstimate(
            step,
            state_list[0],
            state_list[1],
            state_list[2],
            state_list[3],
            state_list[4],
            covariance
        )

    def return_data_array(self):
        """
        provides intuitive and usefully formatted access to the state estimate data.
        :return: the state estimate data as an order 2D numpy array
        """
        return np.array([
            [self.x_1],
            [self.x_2],
            [self.x_3],
            [self.x_4],
            [self.x_5],
        ])

    def return_data_list(self):
        """
        provides intuitive and usefully formatted access to the state estimate data.
        :return: the state estimate data as a list
        """
        return [self.x_1, self.x_2, self.x_3, self.x_4, self.x_5]

    def return_covariance_array(self):
        """
        provides intuitive access to the covariance matrix
        :return: the covariance data as a 2D numpy array
        """
        return self.covariance

    def get_two_sigma_value(self, state: str):
        """
        provides intuitive access to the two sigma value
        :param state: name of the state attribute associated with the desired two sigma value
        :return: a float of the two sigma value, or 'None" if the v
        """
        if state == 'x_1':
            return self.x1_2sigma
        elif state == 'x_2':
            return self.x2_2sigma
        elif state == 'x_3':
            return self.x1_2sigma
        elif state == 'x_4':
            return self.x2_2sigma
        elif state == 'x_5':
            return self.x1_2sigma
        else:
            print("ERROR: requested state not found for 'get_two_sigma_value' in data_objects")
            return None

    def plot_state(self, color='r'):
        plt.plot(self.x_1, self.x_2, 'd', color=color, markerfacecolor='none')

    def get_covariance_ellipse(self, color='b'):
        major_axis = 2*self.x1_2sigma * sqrt(5.991)
        minor_axis = 2 * self.x2_2sigma * sqrt(5.991)

        return Ellipse(xy=(self.x_1, self.x_2), width=major_axis, height=minor_axis, edgecolor=color, fc='None',
                       ls='--')


class Input:
    """
    data object to hold all information pertinent to the measurement data at a given time step
    """
    def __init__(self, step, input_1, input_2, input_names=None):
        self.step = step
        self.u_1 = input_1
        self.u_2 = input_2
        self.input_names = input_names

    @staticmethod
    def create_from_dict(lookup):
        """
        Used to construct objects directly from a CSV data file
        :param lookup: dictionary keys
        :return: constructed ground truth object
        """
        return Input(
            int(lookup['step']),
            float(lookup['u_1']),
            float(lookup['u_2']),
        )


class Measurement:
    """
    data object to hold all information pertinent to the measurement data at a given time step
    """
    def __init__(self, step, output_1, output_2, output_3, output_4, output_5, output_6, output_names=None):
        self.step = step
        self.y_1 = output_1
        self.y_2 = output_2
        self.y_3 = output_3
        self.y_4 = output_4
        self.y_5 = output_5
        self.y_6 = output_6

        self.output_names = output_names

    @staticmethod
    def create_from_dict(lookup):
        """
        Used to construct objects directly from a CSV data file
        :param lookup: dictionary keys
        :return: constructed ground truth object
        """
        return Measurement(
            int(lookup['step']),
            float(lookup['y_1']),
            float(lookup['y_2']),
        )

    @staticmethod
    def create_from_array(step: int, output_array: np.ndarray):
        """
        fast way to create a Measurement object from a numpy array of measurements
        :param step: time step associated with the data
        :param output_array: numpy array with ordered measurement values
        :return: Measurement object
        """
        if output_array.shape[1]:
            # reduces 2D measurement array down to a single dimension
            output_array = output_array.squeeze()

        return Measurement(
            step,
            output_array[0],
            output_array[1],
            output_array[2],
            output_array[3],
            output_array[4],
            output_array[5],
        )

    @staticmethod
    def create_from_list(step: int, output_list: list, output_names=None):
        """
        fast way to create a Measurement object from a numpy array of measurements
        :param step: time step associated with the data
        :param output_list: list with ordered measurement values
        :param output_names:
        :return: Measurement object
        """

        return Measurement(
            step,
            output_list[0],
            output_list[1],
            output_list[2],
            output_list[3],
            output_list[4],
            output_list[5],
            output_names
        )

    def return_data_list(self):
        """
        provides intuitive access to the measurement data
        :return: the measurement data as a list
        """
        return [self.y_1, self.y_2, self.y_3, self.y_4, self.y_5, self.y_6]

    def return_data_array(self):
        """
        provides intuitive access to the measurement data
        :return: the measurement data as a 2D numpy array
        """
        return np.array([
            [self.y_1],
            [self.y_2],
            [self.y_3],
            [self.y_4],
            [self.y_5],
            [self.y_6],
        ])


def get_noisy_measurement(R: np.ndarray, true_measurement: Measurement):
    step = true_measurement.step
    sample = monte_carlo_sample(true_measurement.return_data_array(), R, 1)
    noisy_measurement = Measurement.create_from_array(step, sample)
    return noisy_measurement


def get_noisy_measurements(R: np.ndarray, true_measurements: list):
    noisy_measurements = list()
    for measurement in true_measurements:
        noisy_measurement = get_noisy_measurement(R, measurement)
        noisy_measurements.append(noisy_measurement)

    return noisy_measurements
