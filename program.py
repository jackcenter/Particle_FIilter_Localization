import os
from tools import initialize


def main():
    working = True
    import tools

    while working:
        print_header()
        cmd = get_user_input()
        working = interpret_command(cmd)


def print_header():
    print('---------------------------------------------------')
    print('                    COHRINT')
    print('           Particle Filter Localization ')
    print('                  Jack Center ')
    print('---------------------------------------------------')
    print()


def get_user_input():
    print('Select from the following programs:')
    print(' [1]: Individual Simulation')
    # print(' [2]: Benchmarking')
    # print(' [3]: Decentralized Data Fusion')
    # print(' [4]: Target Search')
    print(' [q]: Quit')
    print()
    print(' NOTE: parameters for landmarks and dynamics models can be changed in settings.')
    print()

    cmd = input(' Select an exercise would you like to run: ')
    print()

    cmd = cmd.strip().lower()

    return cmd


def interpret_command(cmd):
    if cmd == '1':      # path planning
        status = os.system("python individual_simulation.py")

    elif cmd == '2':
        print(" Sorry, this section is not functional at this time")
        # status = os.system("python benchmarking/individual_simulation.py")

    elif cmd == '3':
        print(" Sorry, this section is not functional at this time")
        # status = os.system("python decentralized_data_fusion/individual_simulation.py")

    elif cmd == '4':
        print(" Sorry, this section is not functional at this time")
        # status = os.system("python target_search/individual_simulation.py")

    elif cmd == 'q':
        print(" closing program ... goodbye!")
        return False

    else:
        print(' ERROR: unexpected command...')
        run_again = input(' Would you like to run another program?[y/n]: ')
        print()

        if run_again != 'y':
            print(" closing program ... goodbye!")
            return False

    return True


if __name__ == '__main__':
    main()
