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

r, c, cnt = 32, 64, 1024

class Autoverse2D:
    def __init__(self):
        self.kernel = np.array([[1<<(3*x+y) for y in range(3)] for x in range(3)])
        self.universe = np.zeros((r,c))
        self.rules = [0] * 512

        # helpers
        ncnt = lambda n: sum([int(x) for x in str(bin(n&495))[2:]])
        is_alive = lambda n: n != (n&495)

        # gol rules
        def symmetrize(rules):
            for i, ru in enumerate(rules):
                if not ru:
                    continue
                new_bin = int(f'{int(bin(i)[2:]):09}'[::-1],2)
                rules[new_bin] = 1
            return rules
        def add(rules):
            for i, ru in enumerate(rules):
                if ru:
                    self.rules[i] = 1
        def gol_rules():
            return [1 if (is_alive(n) and 2<=ncnt(n)<=3 or
                      not is_alive(n) and    ncnt(n)==3)
                      else 0 for n in range(512)]
        def blinker_rules():
            rules = [0] * 512
            for x in [7,56,73,146,292,448]:
                rules[x] = 1
            return rules
        def glider_rules():
            rules = [0] * 512
            for x in [13,22,35,52,67,88,120,137,150,176,180,208,240,278,292,344,416,448]:
                rules[x] = 1
            return rules
        def space_ship_rules():
            rules = [0] * 512
            for x in [7,11,14,26,30,35,38,49,53,54,56,60,73,82,89,112,145,147,152,164,200,210,224,290,312,368,392,408,416,432,448]:
                rules[x] = 1
            return rules
        def glider_factory_rules():
            rules = [0] * 512
            for x in [7,11,13,14,19,21,22,23,25,26,27,28,29,30,35,37,38,41,42,44,49,50,51,52,53,54,56,57,67,69,73,74,76,81,82,83,84,85,86,88,89,90,92,97,100,104,112,113,114,116,120,134,137,138,145,146,147,148,149,150,152,153,154,161,162,164,168,176,178,180,193,194,200,208,209,210,212,216,224,240,259,261,262,265,268,273,274,275,276,277,278,280,281,284,289,290,292,296,304,305,306,308,312,321,324,328,336,337,338,340,344,352,368,386,392,400,401,402,404,416,432,448,464]:
                rules[x] = 1
            return rules

        # Blinker
        # self.universe[[10,10,10], [15,16,17]] = 1

        # Light Weight Space Ship (LWSS)
        # self.universe[[18,18,19,20,21,21,21,21,20], [45,48,44,44,44,45,46,47,48]] = 1

        # Glider
        # self.universe[[28,29,30,30,30], [30,31,29,30,31]] = 1

        # Glider Factory
        self.universe[[4,4,5,5],[0,1,0,1]] = 1
        self.universe[[2,2,3,3,4,4,5,5,5,5,6,6,7,7,8,8],[12,13,11,15,10,16,10,14,16,17,10,16,11,15,12,13]] = 1
        self.universe[[0,1,1,2,2,3,3,4,4,5,5,6],[24,22,24,20,21,20,21,20,21,22,24,24]] = 1
        self.universe[[2,2,3,3],[34,35,34,35]] = 1
        
        # ~cnt random spots
        # self.universe[np.random.randint(r, size=cnt),
        #               np.random.randint(c, size=cnt)] = 1

        # add(gol_rules())
        # add(blinker_rules())
        # add(glider_rules())
        # add(space_ship_rules())
        add(glider_factory_rules())
        self.used_rules = [0] * 512

    def step(self):
        tmp = scipy.ndimage.convolve(
                self.universe,
                self.kernel,
                output=np.uint64,
                mode='wrap')
    
        for i, x in np.ndenumerate(tmp):
            if self.rules[x]:
                self.used_rules[x] = 1
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
        for i, x in enumerate(self.used_rules):
            if x:
                print(i, end=",")
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
