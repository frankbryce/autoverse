#!/usr/bin/python3

from datetime import datetime
import math
import numpy as np
import os
from pathlib import Path
import scipy
import scipy.ndimage
import sty
import sys
from time import sleep
from tqdm import tqdm

class Autoverse2D:
    def __init__(self, infile='diamond.csv'):
        self.kernel = np.array([[1<<(3*x+y) for y in range(3)] for x in range(3)])
        goal = np.loadtxt(infile, delimiter=',')
        self.universe = np.zeros((64,128))
        rules = np.zeros((512,))
        ncnt = lambda n: sum([int(x) for x in str(bin(n&495))[2:]])
        is_alive = lambda n: n != (n&495)
        gol_rules = [1 if (is_alive(n) and 2<=ncnt(n)<=3 or
                            not is_alive(n) and    ncnt(n)==3)
                            else 0 for n in range(512)]

        # gol walker
        self.rules = gol_rules
        self.universe[[28,29,30,30,30], [30,31,29,30,31]] = 1

    def step(self):
        tmp = scipy.ndimage.convolve(
                self.universe,
                self.kernel,
                output=np.uint64,
                mode='wrap')
    
        for i, x in np.ndenumerate(tmp):
            if self.rules[x]:
                self.universe[i] = 1
            else:
                self.universe[i] = 0

    def print_universe(self):
        print('\033[0;0H', end='')
        # tmp = scipy.ndimage.convolve(
        #         self.universe,
        #         self.kernel,
        #         output=np.uint64,
        #         mode='wrap')
        # for i, xt in enumerate(tmp):
        for i, xt in enumerate(self.universe):
            for j, x in enumerate(xt):
                # print(sum([int(d) for d in str(bin(x))[2:]]), end='')
                if x == 0:
                    print(' ', end='')
                if x == 1:
                    print('X', end='')
            print()


def main():
    grid = Autoverse2D()
    grid.print_universe()
    for _ in tqdm(range(1000)):
        grid.step()
        grid.print_universe()
    grid.print_universe()
    

if __name__ == "__main__":
    main()
