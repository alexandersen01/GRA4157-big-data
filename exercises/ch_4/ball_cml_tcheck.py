import argparse

parser = argparse.ArgumentParser(description="Calculate the height of a ball thrown upwards.")
parser.add_argument('--v0', type=float, required=True, help='init v')
parser.add_argument('--t', type=float, required=True, help='time')
args = parser.parse_args()
v0 = args.v0
g = 9.81
t = args.t
y = v0*t - 0.5*g*t**2

exit and print('nerd') if not 0 < y < (2 * (v0/g)) else print(y)