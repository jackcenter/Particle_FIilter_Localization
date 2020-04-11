from math import pi, cos, sin
import os
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Rectangle
import numpy as np
import scipy.stats as stats

from tools import initialize as init
from dynopy.estimationtools.importance_sampling import construct_initial_particles, SIS


def main():
    pass


def run():
    dt = 0.5
    workspace = setup(dt)
    robot = workspace.robots[0]
    inputs = robot.input_list
    measurements = robot.measurement_list

    for u, z in zip(inputs, measurements):
        robot.step_bootstrap(z, u, dt)
        # robot.particle_set.sort(key=myFunc, reverse=True)

    run_animation(workspace, robot)


def myFunc(vals):
    return vals[1]


def setup(dt):
    settings_filename = "settings.txt"
    base_folder = os.path.dirname(__file__)
    data_files = init.get_data_files(settings_filename, base_folder)
    workspace = init.initialize_workspace(data_files)

    robot_settings_filename = 'firebrick.txt'
    robot_pose_filename = 'pose_36_9_0.txt'
    robot_input_list_filename = 'inputs2.csv'

    robot = init.initialize_robot(workspace, robot_settings_filename, robot_pose_filename, robot_input_list_filename,
                                  base_folder)
    Q_true = np.diag([0.000000001, 0.000000001, 0.000000001, 0.000001, 0.000001])
    robot.Q = np.diag([0.01, 0.01, 0.01, 0.001, 0.001])
    robot.R = np.diag([5, .1, 5, .1, 5, .1])
    robot.get_ground_truth(Q_true, dt)
    robot.get_perfect_measurements()
    robot.create_noisy_measurements()

    initial_distros = [
        stats.uniform(loc=24, scale=48),
        stats.uniform(loc=0, scale=18),
        stats.uniform(loc=-.5, scale=1),
        stats.uniform(0, 0),
        stats.uniform(0, 0)
    ]

    initial_particle_set = construct_initial_particles(initial_distros, 100, robot.Q)
    robot.initialize_particle_set(initial_particle_set)

    return workspace

    # plt.plot(x_ords, y_ords, '.')
    # workspace.plot()
    # plt.show()


def run_animation(workspace, robot):
    fig, ax = plt.subplots()
    ax.axis('scaled')
    workspace.plot()



    particle_plotter = []
    estimates = []
    for particle_set in robot.particle_set_list:
        x_ords = []
        y_ords = []
        weights = []
        for particle in particle_set:
            state = particle[0].return_data_array()
            x = state[0]
            y = state[1]
            w = particle[1]

            x_ords.append(x)
            y_ords.append(y)
            weights.append(w)

        particle_plotter.append((x_ords, y_ords, weights))

    estimates = ax.plot([], [], 'b.', ms=2)

    states_list = robot.ground_truth
    pos1, = ax.plot([], [], 'x')
    lines = [pos1]

    patch1 = Rectangle(xy=(0, 0), width=robot.axel_length, height=robot.axel_length, angle=robot.state.get('x3'),
                       edgecolor=robot.color, fc='None', ls='-')
    patches = [patch1]
    for patch in patches:
        ax.add_patch(patch)

    # est1, = ax.plot([], [], marker='d', mec='r', mfc='none')
    # est2, = ax.plot([], [], 'bd', mfc='none')
    # est3, = ax.plot([], [], 'gd', mfc='none')
    # est4, = ax.plot([], [], 'kd', mfc='none')
    # estimates = [est1, est2, est3, est4]

    count_text = ax.text(0, -6, "Current Step: ")
    count_text.set_bbox(dict(facecolor='white'))

    # anim = FuncAnimation(fig, animate, frames=len(states_list),
    #                      fargs=[lines, patches, states_list, estimates, particle_plotter, count_text],
    #                      interval=50, blit=True, repeat_delay=5000)

    anim = FuncAnimation(fig, animate, frames=len(states_list),
                         fargs=[lines, patches, states_list, estimates, particle_plotter, count_text],
                         interval=50, blit=True, repeat=True)

    # base_folder = os.path.dirname(__file__)
    # print(base_folder)
    # movie_file = os.path.join(base_folder, "mymovie.gif")
    # writer = PillowWriter(fps=30)
    #
    # anim.save(movie_file, writer=writer)
    plt.show()


def animate(i, lines, patches, states_list, estimates, particle_plotter, text):
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

    for lnum, line in enumerate(estimates):
        x = particle_plotter[i][0]
        y = particle_plotter[i][1]
        line.set_data(x, y)

    return lines + patches + estimates + [text]


if __name__ == '__main__':
    main()
