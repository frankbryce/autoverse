#!/usr/bin/python3

from datetime import datetime
import numpy as np
import os
from pathlib import Path
import scipy
import scipy.ndimage
import sty
import sys
from time import sleep
from tqdm import tqdm

class Autoverse:

    def __init__(self, width, entropy, jitter, pull, ksz, rule30=False):
        self.width = width
        self.universe = np.random.randint(2, size=(width,))
        self.kernel = np.array([1<<(x) for x in range(ksz)])
        self.entropy = entropy
        self.energies = np.random.rand(width, 2**ksz)
        self.ksz = ksz
        self.diff = np.zeros((self.width,))
        self.jitter = jitter
        self.pull = pull

        if rule30:
            self.kernel = np.array([1<<(x) for x in range(ksz)])
            self.ksz = 3
            for i in range(width):
                self.energies[i] = np.array([0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0])

    def step(self):
        tmp = scipy.ndimage.convolve(
                self.universe,
                self.kernel,
                output=np.int8, mode='wrap')
        cnt = dict(zip(*np.unique(tmp, return_counts=True)))
        
        # randomness to create stability?
        if self.jitter > 0.0:
            r = self.jitter
            self.energies += np.random.rand(*self.energies.shape) * r - (r / 2)
            self.energies = np.clip(self.energies, 0.0, 1.0)

        # pull to middle instead of random and energies txfr
        if self.pull > 0.0:
            self.energies -= self.pull*(self.energies - 0.5*np.ones_like(self.energies))

        # the goal with this should be to converge on some status quo of energy kernels.
        # that was any perturbations in the surrounding area will be sorted out.
        self.diff = np.zeros((self.width,))
        for i, x in enumerate(tmp):
            if np.random.rand() < self.energies[i][x]:
                self.universe[i] = 1
            else:
                self.universe[i] = 0
            if self.entropy > 0.0:
                orig = self.energies[i].copy()
                tot = sum(self.energies[i]) - self.energies[i][x]
                if self.universe[i] == 1:
                    dlt = (1 - self.energies[i][x]) * self.entropy
                else:
                    dlt = -self.energies[i][x] * self.entropy
                if self.pull == 0.0:
                    self.energies[i] -= dlt/((2**ksz)-1)
                self.energies[i][x] = orig[x] + dlt
                self.diff[i] = sum(np.absolute(self.energies[i] - orig))

    def print_universe(self, print_diff=False, print_energies=False):
        if print_diff and print_energies:
            raise("only one of print_diff and print_energies may be specified.")
        for i in range(self.width):
            c = ' ' if self.universe[i] == 0 else 'X'
            r,g,b = 0,0,0
            if print_energies:
                for e in range(len(self.energies[i])):
                    is_r = e & 0b001
                    is_g = e & 0b010
                    is_b = e & 0b100
                    if is_r:
                        r += self.energies[i][e]
                    if is_g:
                        g += self.energies[i][e]
                    if is_b:
                        b += self.energies[i][e]
                norm = 255.0/(2**(self.ksz-1))
                r,g,b = int(r*norm), int(g*norm), int(b*norm)
            if print_diff:
                b = int(128*self.diff[i]/self.entropy + 64)
            print(sty.bg(r,g,b) + c, end='')
        
        print(sty.bg.rs + f' {sum(self.diff):.2f}')

def run(width,
        steps,
        entropy,
        jitter,
        pull,
        ksz,
        rule30,
        print_universe,
        print_diff,
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
        autoverse = Autoverse(width, entropy, jitter, pull, ksz, rule30)
        universe = np.load(f'{load_location}/universe_{load_iteration:09d}.npy')
        energies = np.load(f'{load_location}/energies_{load_iteration:09d}.npy')
        autoverse.universe = universe
        autoverse.width = len(universe)
        autoverse.energies = energies
        if load_metadata:
            metadata = np.load(f'{load_location}/metadata.npy')
            autoverse.entropy = metadata[0]
            autoverse.ksz = metadata[1]
            autoverse.jitter = metadata[2]
            if len(metadata) >= 4:
                autoverse.pull = metadata[3]
            else:
                autoverse.pull = 0.0
    else:
        autoverse = Autoverse(width, entropy, jitter, pull, ksz, rule30)

    Path(save_location).mkdir(parents=True, exist_ok=True)
    np.save(f'{save_location}/metadata.npy',
            np.array([autoverse.entropy, autoverse.ksz, autoverse.jitter, autoverse.pull]))

    if print_universe:
        autoverse.print_universe(print_diff=print_diff, print_energies=print_energies)
    step_cnt = 0
    def iterate():
        nonlocal step_cnt
        step_cnt += 1
        autoverse.step()
        if print_universe:
            autoverse.print_universe(print_diff=print_diff, print_energies=print_energies)
        sleep(speed)
        if save_period > 0 and step_cnt % save_period == 0:
            np.save(f'{save_location}/universe_{step_cnt:09d}.npy', autoverse.universe)
            np.save(f'{save_location}/energies_{step_cnt:09d}.npy', autoverse.energies)
        if step_cnt % prompt_period == 0 and prompt_period > 0:
            autoverse.print_universe(print_diff=print_diff, print_energies=print_energies)
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
    entropy = 0.02
    jitter = 0.0
    pull = 0.00001
    ksz = 3
    print_universe = False
    print_diff = False
    print_energies = False
    if print_diff or print_energies:
        w -= 5
    rule30 = True
    prompt_period = -1
    speed = 0.00
    save_location = 'save/' + datetime.now().strftime("%Y%m%d_%H%M%S")
    save_period = 1000000
    load_location = 'save/20211124_222307'
    load_iteration = 250000
    load_metadata = False
    run(w,
        steps,
        entropy,
        jitter,
        pull,
        ksz,
        rule30,
        print_universe,
        print_diff,
        print_energies,
        prompt_period,
        speed,
        save_location,
        save_period,
        load_location,
        load_iteration,
        load_metadata)
