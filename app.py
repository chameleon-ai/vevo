
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', type=str, default='vevosing', choices=['vevo', '1', 'vevosing', '1.5'], help='Vevo version to launch.')
    args = parser.parse_args()
    if args.version == 'vevosing' or args.version == '1':
        print('Launching vevosing (vevo 1.5)')
        from vevosing_gui import vevosing_gui
        vevosing_gui()
    elif args.version == 'vevo' or args.version == '1.5':
        print('Launching vevo (vevo 1.0)')
        from vevo_gui import vevo_gui
        vevo_gui()