import os


def main():
    working = True

    while working:
        print_header()
        cmd = get_user_input()
        working = interpret_command(cmd)


def print_header():
    print('---------------------------------------------------')
    print('                  Benchmarking ')
    print('---------------------------------------------------')
    print()


def get_user_input():
    print('This program runs the following exercises:')
    print(' [1]: ')
    print(' [2]: ')
    print(' [3]: ')
    print(' [4]: ')
    print()
    print(' NOTE: parameters for landmarks and dynamics models can be changed in settings.')
    print()

    cmd = input(' Select an exercise would you like to run: ')
    cmd = cmd.strip().lower()
    return cmd


def interpret_command(cmd):
    if cmd == '1':      # path planning
        print(cmd)

    elif cmd == '2':
        print(cmd)

    elif cmd == '3':
        print(cmd)

    elif cmd == '4':
        print(cmd)

    else:
        print(' ERROR: unexpected command...')

    print()
    run_again = input(' Would you like to run another exercise?[y/n]: ')

    if run_again != 'y':
        return False

    return True


if __name__ == '__main__':
    main()
