from math import pi, cos, sin
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from tools import initialize as init


def main():
    pass


def run():
    # TODO: create the workspace with a map and an agent that drives around
    setup()

    # TODO: have the robot drive around

    # add landmarks
    # add obstacles?
    # print range and bearing measurements

    pass


def setup():
    settings_filename = "settings.txt"
    base_folder = os.path.dirname(__file__)
    data_files = init.get_data_files(settings_filename, base_folder)
    workspace = init.initialize_workspace(data_files)

    robot_settings_filename = 'firebrick.txt'
    robot_pose_filename = 'pose_36_9_0.txt'
    robot_input_list_filename = 'inputs2.csv'

    robot = init.initialize_robot(workspace, robot_settings_filename, robot_pose_filename, robot_input_list_filename,
                                  base_folder)
    robot.get_ground_truth(.5)

    # print([x.x_3 for x in robot.ground_truth])
    run_animation(workspace, robot.ground_truth)
    # x_ords = [x.x_1 for x in robot.ground_truth]
    # y_ords = [y.x_2 for y in robot.ground_truth]


    # plt.plot(x_ords, y_ords, '.')
    # workspace.plot()
    # plt.show()


def run_animation(workspace, states_list):
    fig, ax = plt.subplots()
    ax.axis('scaled')
    workspace.plot()

    pos1, = ax.plot([], [], 'x')
    lines = [pos1]

    robot1 = workspace.robots[0]
    patch1 = Rectangle(xy=(0, 0), width=robot1.axel_length, height=robot1.axel_length, angle=robot1.state.get('x3'),
                       edgecolor=robot1.color, fc='None', ls='-')
    patches = [patch1]
    for patch in patches:
        ax.add_patch(patch)

    count_text = ax.text(15, -45, "Current Step: ")
    count_text.set_bbox(dict(facecolor='white'))

    anim = FuncAnimation(fig, animate, frames=len(states_list), fargs=[lines, patches, states_list, count_text],
                         interval=100, blit=True, repeat_delay=5000)

    plt.show()


def animate(i, lines, patches, states_list, text):
    text.set_text("Current Step: {}".format(i))
    for lnum, line in enumerate(lines):
        state = states_list[i].return_data_list()
        x = state[0]
        y = state[1]
        line.set_data(x, y)

    for pnum, patch in enumerate(patches):
        state = states_list[i].return_data_list()
        x = state[0]
        y = state[1]
        theta = state[2]

        w = patch.get_width()
        h = patch.get_height()

        x_correction = w/2*cos(theta) - h/2*sin(theta)
        y_correction = w/2*sin(theta) + h/2*cos(theta)

        patch.set_xy((x - x_correction, y-y_correction))
        patch.angle = theta*180/pi

    return lines + patches + [text]


if __name__ == '__main__':
    main()
