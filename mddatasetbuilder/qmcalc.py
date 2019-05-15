"""QM calculation with Gaussian."""

import argparse
import os

from gaussianrunner import GaussianRunner


def qmcalc(gjfdir, command='g16', nologs=False):
    """QM Calculation."""
    gjflist = [os.path.join(gjfdir, filename) for filename in os.listdir(
        gjfdir) if filename.endswith('.gjf')]
    properties = ['energy', 'atomic_number',
                  'coordinate', 'force'] if nologs else None
    GaussianRunner(command=command).runGaussianInParallel(
        'GJF', gjflist, properties=properties, savelog=not nologs)


def _commandline():
    parser = argparse.ArgumentParser(description='QM Calculator')
    parser.add_argument('-d', '--dir',
                        help='Dataset dirs', required=True)
    parser.add_argument('-c', '--command',
                        help='Gaussian command, default is g16', default="g16")
    parser.add_argument(
        '--nologs', help='Store out files instead of logs', action="store_true")
    args = parser.parse_args()
    qmcalc(gjfdir=args.dir, command=args.command, nologs=args.nologs)
