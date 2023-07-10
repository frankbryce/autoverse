#!/usr/bin/python3

from itertools import combinations
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

r, c, cnt = 64, 128, 4096

class Autoverse2D:
    def __init__(self):
        self.kernel = np.array([[1<<(3*x+y) for y in range(3)] for x in range(3)])
        self.universe = np.zeros((r,c))
        self.rules = [0] * 512

        # helpers
        ncnt = lambda n: sum([int(x) for x in str(bin(n&495))[2:]])
        is_alive = lambda n: n != (n&495)

        # gol rules
        self.gol_rules = [1 if (is_alive(n) and 2<=ncnt(n)<=3 or
                            not is_alive(n) and    ncnt(n)==3)
                            else 0 for n in range(512)]

        # ~1024 random spots
        self.universe[np.random.randint(r, size=cnt),
                      np.random.randint(c, size=cnt)] = 1
        self.rules = list(self.gol_rules)

        # gol walker
        # self.rules = gol_rules
        # self.universe[[28,29,30,30,30], [30,31,29,30,31]] += 1


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
        for i, x in enumerate(self.rules):
            if x:
                print(i, end=",")
        print()
        for i, xt in enumerate(self.universe):
            for j, x in enumerate(xt):
                if x == 0:
                    print(' ', end='')
                if x == 1:
                    print('X', end='')
            print()

def runloop():
    while True:
        yield

def main():
    os.system('clear')
    hist = []
    grid = Autoverse2D()
    grid.print_universe()
    for _ in tqdm(runloop()):
        grid.print_universe()
        p = input()
        if p != "":
            print(p)
            if "r" in p or "R" in p:
                grid.universe[:,:] = 0
                grid.universe[np.random.randint(r, size=cnt),
                              np.random.randint(c, size=cnt)] = 1
                continue
            if "a" in p or "A" in p:
                idx = np.random.randint(512)
                grid.rules[idx] = 1 - grid.rules[idx]
                hist.append(idx)
                continue
            if "u" in p or "U" in p:
                idx = hist.pop(len(hist)-1)
                grid.rules[idx] = 1 - grid.rules[idx]
            if "d" in p or "D" in p:
                while True:
                    idx = np.random.randint(len(grid.rules))
                    if grid.rules[idx] == 1:
                        break
                grid.rules[idx] = 0
                hist.append(idx)
            if "g" in p or "G" in p:
                grid.rules = list(grid.gol_rules)

            try:
                grid.rules[int(p)] = 1 - grid.rules[int(p)]
            except ValueError:
                pass
        grid.step()
    grid.print_universe()
    

if __name__ == "__main__":
    main()
