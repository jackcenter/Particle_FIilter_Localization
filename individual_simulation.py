import os
import simulation


def main():
    working = True

    while working:
        print_header()
        cmd = get_user_input()
        working = interpret_command(cmd)

    return 0


def print_header():
    print('---------------------------------------------------')
    print('              Individual Simulation ')
    print('---------------------------------------------------')
    print()


def get_user_input():
    print('This program runs the following simulations:')
    print(' [1]: Grid Approximation')
    print(' [2]: Importance Sampling')
    print(' [3]: Sequential Importance Sampling (Bootstrap) Particle Filter')
    print(' [4]: Gaussian Mixture')
    print()
    print(' NOTE: parameters for landmarks and dynamics models can be changed in settings.')
    print()

    cmd = input(' Select an exercise would you like to run: ')
    cmd = cmd.strip().lower()
    return cmd


def interpret_command(cmd):
    if cmd == '1':      # path planning
        simulation.run()

    elif cmd == '2':
        print(" Sorry, this section is not functional at this time")

    elif cmd == '3':
        print(" Sorry, this section is not functional at this time")

    elif cmd == '4':
        print(" Sorry, this section is not functional at this time")

    else:
        print(' ERROR: unexpected command...')

    print()
    run_again = input(' Would you like to run another exercise?[y/n]: ')

    if run_again != 'y':
        return False

    return True


if __name__ == '__main__':
    main()
