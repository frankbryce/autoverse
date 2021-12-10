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
        self.universe = np.zeros(goal.shape)
        self.universe[np.random.randint(goal.shape[0]),np.random.randint(goal.shape[1])] = 1
        rules = np.zeros(goal.shape+(512,))
        for i, x in np.ndenumerate(goal):
            if x == 1:
                rules[i] = [0] + [1]*511
            else:
                rules[i] = [0] * 512
        self.alive_rules = rules.copy()
        self.dead_rules = rules.copy()

    def step(self):
        tmp = scipy.ndimage.convolve(
                self.universe,
                self.kernel,
                output=np.uint8, mode='wrap')
    
        for i, x in np.ndenumerate(tmp):
            if self.universe[i] == 1:
                if self.alive_rules[i][x]:
                    self.universe[i] = 1
                else:
                    self.universe[i] = 0
            else:
                if self.dead_rules[i][x]:
                    self.universe[i] = 1
                else:
                    self.universe[i] = 0

    def print_universe(self):
        print('\033[0;0H', end='')
        for i, xt in enumerate(self.universe):
            for j, x in enumerate(xt):
                if x == 0:
                    print(' ', end='')
                if x == 1:
                    print('X', end='')
            print()


def main():
    grid = Autoverse2D()
    grid.print_universe()
    input()
    for _ in tqdm(range(1000)):
        grid.step()
        grid.print_universe()
        # sleep(0.1)
        input()
    

if __name__ == "__main__":
    main()
