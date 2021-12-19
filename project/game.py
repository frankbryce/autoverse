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


def logistic(p, x):
    # p is the desired y-intercept
    # x is the input to the logistic function (x axis value)
    a = (1-p)/p
    ret = 1.0/(1.0+a*math.exp(-x))
    assert ret >= 0.0 and ret <= 1.0
    return ret

# one_cnt = np.array([0.05,0.35,0.35,0.65,0.35,0.65,0.65,0.95])
one_cnt = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])

class Autoverse:

    def __init__(self, width, dissipate, entropy, jitter, pull, ksz, cperr, rule30):
        self.width = width
        self.universe = np.random.randint(2, size=(width,))
        self.kernel = np.array([1<<(x) for x in range(ksz)])
        self.dissipate = dissipate
        self.entropy = entropy
        self.rules = np.random.rand(width, 2**ksz)
        self.ksz = ksz
        self.cperr = cperr
        self.diff = np.zeros((self.width,))
        self.rand = np.zeros((self.width,))
        self.jitter = jitter
        self.pull = pull
        self.energies = np.zeros((self.width,))

        if rule30:
            self.kernel = np.array([1<<(x) for x in range(ksz)])
            self.ksz = 3
            for i in range(width):
                self.rules[i] = np.array([0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0])

    def step(self):
        tmp = scipy.ndimage.convolve(
                self.universe,
                self.kernel,
                output=np.int8, mode='wrap')
        
        # randomness to create stability?
        if self.jitter > 0.0:
            r = self.jitter
            self.rules += np.random.rand(*self.rules.shape) * r - (r / 2)
            self.rules = np.clip(self.rules, 0.0, 1.0)

        # pull to middle instead of random and rules txfr
        if self.pull > 0.0:
            self.rules -= self.pull*(self.rules - 0.5*np.ones_like(self.rules))

        # the goal with this should be to converge on some status quo of energy kernels.
        # that was any perturbations in the surrounding area will be sorted out.
        self.diff = np.zeros((self.width,))
        self.rand = np.zeros((self.width,))
        self.energies += self.dissipate * (self.universe-0.5)
        self.energies = np.clip(self.energies, -100, 100)
        self.rules = np.clip(self.rules, 1e-10, 1.0-1e-10)
        for i, x in enumerate(tmp):
            self.rand[i] = min(self.rules[i][x], 1.0 - self.rules[i][x])
            if np.random.rand() < logistic(self.rules[i][x], -self.energies[i]):
                self.universe[i] = 1
            else:
                self.universe[i] = 0

            # cperr controls copy error percent. Flip universe bit if error occurs.
            if self.cperr > 0.0:
                if np.random.rand() < self.cperr:
                    self.universe[i] = 1 - self.universe[i]

            # if there is entropy, need to adjust rules for this cell.
            if self.entropy > 0.0:
                orig = self.rules[i].copy()
                # move energy towards 0.25 of universe (TODO: not working)
                # d = (1.0-sum(self.rules[i])/len(self.rules[i]))/(1.0-0.15)
                d = 0.5
                if self.universe[i] == 1:
                    dlt = (1 - self.rules[i][x]) * self.entropy * one_cnt[x] * (1.0-d)
                else:
                    dlt = -self.rules[i][x] * self.entropy * (1.0-one_cnt[x]) * d
                # self.rules[i] -= dlt/((2**ksz)-1)
                self.rules[i][x] = orig[x] + dlt
                self.diff[i] = sum(np.absolute(self.rules[i] - orig))

    def print_universe(self, print_diff, print_rules, print_energies):
        if ((print_diff and print_rules) or
            (print_diff and print_energies) or
            (print_rules and print_energies)):
            raise("only one of print_* may be specified.")
        for i in range(self.width):
            c = ' ' if self.universe[i] == 0 else 'X'
            r,g,b = 0,0,0
            if print_rules:
                for e in range(len(self.rules[i])):
                    is_r = e & 0b001
                    is_g = e & 0b010
                    is_b = e & 0b100
                    if is_r:
                        r += self.rules[i][e]
                    if is_g:
                        g += self.rules[i][e]
                    if is_b:
                        b += self.rules[i][e]
                norm = 196.0/(2**(self.ksz-1))
                r,g,b = int(r*norm), int(g*norm), int(b*norm)
            if print_diff:
                b = int(128*self.diff[i]/self.entropy + 64)
            if print_energies:
                b = int(128*logistic(0.5, self.energies[i])+64)
            r = int(min(max(r,0),196))
            g = int(min(max(g,0),196))
            b = int(min(max(b,0),196))
            print(sty.bg(r,g,b) + c, end='')
        
        print(sty.bg.rs, end='')
        if print_diff:
            print(f' {sum(self.diff):.1f}', end='')
        if print_rules:
            print(f' {sum(self.rand):.1f}', end='')
        if print_energies:
            print(f' {sum(self.universe):03d}', end='')
        print()

def run(width,
        steps,
        dissipate,
        entropy,
        jitter,
        pull,
        ksz,
        cperr,
        rule30,
        print_universe,
        print_diff,
        print_rules,
        print_energies,
        prompt_period,
        speed,
        save_location,
        save_period,
        load_location,
        load_iteration,
        load_metadata):
    if load_location:
        if load_iteration is None:
            raise("Please specify load_iteration")
        autoverse = Autoverse(width, dissipate, entropy, jitter, pull, ksz, cperr, rule30)
        universe = np.load(f'{load_location}/universe_{load_iteration:09d}.npy')
        rules = np.load(f'{load_location}/rules_{load_iteration:09d}.npy')
        autoverse.universe = universe
        autoverse.width = len(universe)
        autoverse.rules = rules
        if load_metadata:
            metadata = np.load(f'{load_location}/metadata.npy')
            autoverse.entropy = metadata[0]
            autoverse.ksz = metadata[1]
            autoverse.jitter = metadata[2]
            if len(metadata) >= 4:
                autoverse.pull = metadata[3]
            else:
                autoverse.pull = 0.0
            if len(metadata) >= 5:
                autoverse.cperr = metadata[4]
            else:
                autoverse.cperr = 0.0
            if len(metadata) >= 6:
                autoverse.dissipate = metadata[5]
            else:
                autoverse.dissipate = 0.0
            print(autoverse.dissipate,
                  autoverse.entropy,
                  autoverse.ksz,
                  autoverse.jitter, 
                  autoverse.pull,
                  sep='\n')
    else:
        autoverse = Autoverse(width, dissipate, entropy, jitter, pull, ksz, cperr, rule30)

    Path(save_location).mkdir(parents=True, exist_ok=True)
    np.save(f'{save_location}/metadata.npy',
            np.array([
                autoverse.dissipate,
                autoverse.entropy,
                autoverse.ksz,
                autoverse.jitter,
                autoverse.pull,
                autoverse.cperr]))

    if print_universe:
        autoverse.print_universe(print_diff, print_rules, print_energies)
    step_cnt = 0
    def iterate():
        nonlocal step_cnt
        step_cnt += 1
        autoverse.step()
        if print_universe:
            autoverse.print_universe(print_diff, print_rules, print_energies)
        sleep(speed)
        if save_period > 0 and step_cnt % save_period == 0:
            np.save(f'{save_location}/universe_{step_cnt:09d}.npy', autoverse.universe)
            np.save(f'{save_location}/rules_{step_cnt:09d}.npy', autoverse.rules)
        if step_cnt % prompt_period == 0 and prompt_period > 0:
            autoverse.print_universe(print_diff=print_diff, print_rules=print_rules)
            input("")

    if steps >= 0:
        for _ in range(steps):
            iterate()
    else:
        while True:
            iterate()


if __name__ == "__main__":
    # decode flags?
    w = os.get_terminal_size().columns
    steps = -1
    dissipate = 0.0
    entropy = 0.80
    jitter = 0.000
    pull = 0.0
    cperr = 0.00
    ksz = 3
    print_universe = True
    print_diff = False
    print_rules = True
    print_energies = False
    if print_diff or print_rules or print_energies:
        w -= 5
    rule30 = False
    prompt_period = -1
    speed = 0.00
    save_location = 'save/' + datetime.now().strftime("%Y%m%d_%H%M%S")
    save_period = 1000000
    load_location = None # 'save/20211125_094017'
    load_iteration = 55000000
    load_metadata = False
    run(w,
        steps,
        dissipate,
        entropy,
        jitter,
        pull,
        ksz,
        cperr,
        rule30,
        print_universe,
        print_diff,
        print_rules,
        print_energies,
        prompt_period,
        speed,
        save_location,
        save_period,
        load_location,
        load_iteration,
        load_metadata)
