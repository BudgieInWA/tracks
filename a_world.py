import sys
import random
import time


from collections import namedtuple 

from opensimplex import OpenSimplex
from clint.textui import colored, puts, min_width

EPS = 1e-4

class RandomShape:
    def __init__(self, seed=0):
        self.simplex_fn = OpenSimplex(seed=seed)

    def simplex_pos(self, hex):
        return self.simplex(hex) / 2.0 + 0.5

    def simplex(self, hex):
        return self.simplex_fn.noise2d(*hex) 


def close_enough(a, b):
    """Floating point rough eq fn"""
    return abs(a - b) < EPS

class Hex(namedtuple('Hex', ['r', 'q'])):

    def add(self, other):
        return Hex(self.r + other.r, self.q + other.q)

    def subtract(self, other):
        return Hex(self.r - other.r, self.q - other.q)

    def scaled(self, k):
        return Hex(self.r * k, self.q * k)
    
    def neighbours(self):
        return (self.add(d) for d in Hex.directions)


    @staticmethod
    def range(radius):
        for r in range(2*radius + 1): 
            # q < 0
            for q in range(max(-r, -radius), 0):
                yield Hex(r - radius, q)
            # q >= 0
            for q in range(min(radius, 2 * radius - r) + 1):
                yield Hex(r - radius, q)

    @staticmethod
    def print_range(hexes, item=lambda h: h, hex_width=2, indent=lambda: 0):
        # Assume scanline ordering
        last_r = None
        for h in hexes: 
            if h.r != last_r:
                print()
                print(" " * int(hex_width * indent(h.r)), end="")
                last_r = h.r

            print("{:^{}s}".format(str(item(h)), hex_width), end=" ")

    def __str__(self):
        return "{},{}".format(self.r, self.q)

Hex.directions = [Hex(1, 0), Hex(-1, 1), Hex(0, -1), Hex(-1, 0), Hex(1, -1), Hex(0, +1)]


class LandHex:
    def __init__(self, hex):
        self.hex = hex
        self.neighbours = None # needs to be filled in for us

        self.tracks = set() 


    def __str__(self):
        return "{},{}".format(self.hex.r, self.hex.q)

class TrackSegment:
    def __init__(self, land, dir):
        self.land = land
        self.dir = dir

class TrainCar:
    def __init__(self, type):
        self.type = type


class Landscape:
    def __init__(self, seed=0, radius=3):
        self.seed = seed
        self.shape = RandomShape(seed)

        self.radius = radius
        self.hexes = None
        self.land = None

        self.init()

    def init(self):
        """Generate a clump of land."""
        self.hexes = list(Hex.range(self.radius))
        self.land = dict((h, LandHex(h)) for h in self.hexes)

        # Link land hexes via neighbours.
        for l in self.land.values():
            l.neighbours = [self.land[n] for n in l.hex.neighbours() if n in self.land]

        # Generate some random tracks starting in the middle.
        rand = random.Random()
        rand.seed(self.seed)
        land = self.land[Hex(0, 0)]
        dir = None
        while land:
            # Add track back to previous hex
            if dir:
                land.tracks.add(TrackSegment(land, dir.scaled(-1)))

            # Choose new direction and add tracks forward
            dir = rand.choice(Hex.directions) # TODO make sure the turns aren't too tight.
            land.tracks.add(TrackSegment(land, dir))

            land = self.land.get(land.hex.add(dir))


    def do_step(self):
        pass

    def scan_land(self):
        """Return lands in scanline order."""
        return (self.land[h] for h in self.hexes)

    def print(self):
        Hex.print_range(
                self.hexes,
                item=lambda h: self.land[h],
                hex_width=5,
                indent=lambda r: abs(r)/2)

