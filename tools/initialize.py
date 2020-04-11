import csv
import os
from dynopy.workspace.workspace import Workspace, Landmark, DifferentialDrive


def get_data_files(filename, base_folder):
    file = os.path.join(base_folder, 'settings', filename)

    with open(file, 'r', encoding='utf8') as fin:
        reader = csv.reader(fin, skipinitialspace=True, delimiter=',')
        map_data_filename = next(reader)[0]
        landmark_data_filename = next(reader)[0]

    map_file = os.path.join(base_folder, 'settings', 'maps', map_data_filename)
    print(map_file)
    landmark_file = os.path.join(base_folder, 'settings', 'landmarks', landmark_data_filename)
    settings_files_dict = {'map': map_file, 'landmarks': landmark_file}
    return settings_files_dict


def initialize_workspace(files: dict):
    bounds, obstacles, landmarks = load_environment_data(files.get('map'), files.get('landmarks'))
    return Workspace(bounds, obstacles, landmarks)


def initialize_robot(workspace, settings_filename, pose_filename, inputs_filename, base_folder):
    settings_file = os.path.join(base_folder, 'settings/robots', settings_filename)
    pose_file = os.path.join(base_folder, 'settings/poses', pose_filename)
    inputs_file = os.path.join(base_folder, 'settings/robots', inputs_filename)
    robot_settings = load_robot_settings(settings_file)
    state = load_pose(pose_file)
    robot = DifferentialDrive(robot_settings, state, workspace)
    robot.read_inputs(inputs_file)
    workspace.robots.append(robot)
    return robot


def load_environment_data(map_filename, landmark_filename=None):
    """
    Loads the environment boundaries and obstacles from a text file
    :param map_filename: path and name to the file with robot state information
    :param landmark_filename: path and name to the file with robot state information
    :return: lists of tuples of coordinates for the environment boundaries and obstacles, and a list of landmark objects
    """
    environment_bounds = list()
    obstacles = list()

    with open(map_filename, 'r', encoding='utf8') as fin:

        reader = csv.reader(fin, skipinitialspace=True, delimiter=',')

        raw_bounds = next(reader)
        while raw_bounds:
            x_coordinate = int(raw_bounds.pop(0))
            y_coordinate = int(raw_bounds.pop(0))
            coordinate = (x_coordinate, y_coordinate)
            environment_bounds.append(coordinate)

        for raw_obstacle in reader:
            temporary_obstacle = list()

            while raw_obstacle:
                x_coordinate = float(raw_obstacle.pop(0))
                y_coordinate = float(raw_obstacle.pop(0))
                coordinate = (x_coordinate, y_coordinate)
                temporary_obstacle.append(coordinate)

            obstacles.append(temporary_obstacle)

    with open(landmark_filename, 'r', encoding='utf8') as fin:
        reader = csv.DictReader(fin, skipinitialspace=True, delimiter=',')

        landmarks = []
        for settings in reader:
            landmarks.append(Landmark.create_from_dict(settings))

    return environment_bounds, obstacles, landmarks


def load_robot_settings(file):
    with open(file, 'r', encoding='utf8') as fin:
        reader = csv.reader(fin, skipinitialspace=True, delimiter=',')

        keys = next(reader)
        values = next(reader)

        settings = {}
        for key, val in zip(keys, values):
            try:
                val = float(val)
            except ValueError:
                print("    Error: {} cannot be converted to a float".format(val))

            settings.update({key: val})

        return settings


def load_pose(filename):
    """
    Loads initial and goal state information for a robot from a text file
    :param filename: path and name to the file with robot state information
    :return: dictionaries for the initial state and goal state
    """

    with open(filename, 'r', encoding='utf8') as fin:

        reader = csv.DictReader(fin, skipinitialspace=True, delimiter=',')

        raw_states = []
        for state in reader:

            temporary_state = {}
            for key, value in state.items():
                try:
                    temporary_state[key] = float(value)
                except ValueError:
                    temporary_state[key] = None

            raw_states.append(temporary_state)

    initial_state = raw_states[0]

    return initial_state
