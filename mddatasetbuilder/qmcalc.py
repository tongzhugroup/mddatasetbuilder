"""QM calculation with Gaussian."""

import argparse
import os

from gaussianrunner import GaussianRunner


def qmcalc(gjfdir, command='g16'):
    """QM Calculation."""
    gjflist = [os.path.join(gjfdir, filename) for filename in os.listdir(
        gjfdir) if filename.endswith('.gjf')]
    GaussianRunner(command=command).runGaussianInParallel(
        'GJF', gjflist)


def _commandline():
    parser = argparse.ArgumentParser(description='QM Calculator')
    parser.add_argument('-d', '--dir',
                        help='Dataset dirs', required=True)
    parser.add_argument('-c', '--command',
                        help='Gaussian command, default is g16', default="g16")
    args = parser.parse_args()
    qmcalc(gjfdir=args.dir, command=args.command)
