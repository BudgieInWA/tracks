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

    def distance(self, h):
        return (abs(self.q - h.q) 
              + abs(self.q + self.r - h.q - h.r)
              + abs(self.r - h.r)) / 2


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

print(Hex(1, 0).distance(Hex(-1,1)))
print(Hex(0, 1).distance(Hex(-1,1)))

class LandHex:
    def __init__(self, hex):
        self.hex = hex
        self.neighbours = None # needs to be filled in for us

        self.tracks = set() 


    def __str__(self):
        return "{},{}".format(self.hex.r, self.hex.q)


class Track:
    """A track on some land that goes between edges."""

    def __init__(self, land, length):
        self.land = land
        self.length = length

    def leave(self, car):
        pass

    def enter(self, car):
        pass

class Track2(Track):
    """A Track with two ends."""

    def __init__(self, land, length, start_dir, end_dir):
        super().__init__(land, length)

        self.start = start_dir
        self.end = end_dir

        self.start_neighbours = set()
        self.end_neighbours = set()

        self.cars = set()

    @staticmethod
    def make(land, dir1, dir2):
        if dir1 == dir2.scaled(-1):
            return StraightTrack(land, dir1)
        elif dir1.distance(dir2) == 2:
            return CurvedTrack(land, dir1, dir2)
        else:
            raise ValueError("cannot create 2track between given directions")

    def enter(self, car, dir, dist):
        self.cars.add(car)
        car.track = self
        if dir == self.start:
            car.track_facing = 1
            car.track_pos = 0
        elif dir == self.end:
            car.track_facing = -1
            car.track_pos = self.length
        else:
            raise ValueError("could not add car to track because track doesn't go to given direction")

        if dist > 0:
            self.move_car(car, dist)

    def move_car(self, car, dist):
        """Move a car along the track."""

        car.track_pos += dist * car.track_facing

        # Move to the next track segment if needed.
        if car.track_pos > self.length:
            next_segment = car.choose_next_track(self.end_neighbours)
            next_segment.enter(car, self.end.scaled(-1), car.track_pos - self.length)
            self.leave(car)

        if car.track_pos < 0:
            next_segment = car.choose_next_track(self.start_neighbours)
            next_segment.enter(car, self.end.scaled(-1), -car.track_pos)
            self.leave(car)


class StraightTrack(Track2):
    """Track that goes from one edge to the oposite edge.
    
    Length is 1 by definition."""
    def __init__(self, land, dir1):
        super().__init__(land, 1.0, dir1, dir1.scaled(-1))

class CurvedTrack(Track2):
    """Track that goes from one edge to an edge two spots away."""

    def __init__(self, land, dir1, dir2):
        super().__init__(land, 0.90689968211, dir1, dir2)

        self.arc_center_dir = None
        for d in Hex.directions:
            if close_enough(d.distance(self.start), 1) and close_enough(d.distance(self.end), 1):
                self.arc_center_dir = d
                break

        print("start: {}\narc: {}\nend: {}".format(self.start, self.arc_center_dir, self.end))

        if not self.arc_center_dir:
            raise ValueError("dir1 and dir2 are not two hexes from oneanother")


class TrainCar:
    def __init__(self):
        self.track = None
        self.speed = 0.1

    def do_step(self):
        if self.track:
            self.track.move_car(self, self.speed)

    def choose_next_track(self, tracks):
        return next(iter(tracks))


class Landscape:
    def __init__(self, seed=0, radius=3):
        self.seed = seed
        self.shape = RandomShape(seed)

        self.radius = radius
        self.hexes = None
        self.land = None

        self.trains = []

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
        start_dir = Hex.directions[0]
        last_track = None
        while land:
            print("Track through {}".format(land.hex))

            end_dir = rand.choice([d for d in Hex.directions if start_dir.distance(d) > 1])
            track = Track2.make(land, start_dir, end_dir)
            land.tracks.add(track)

            if last_track:
                last_track.end_neighbours.add(track)
                track.start_neighbours.add(last_track)

            last_track = track
            land = self.land.get(land.hex.add(end_dir))
            start_dir = end_dir.scaled(-1)

        # Put a train on the track
        track = next(iter(self.land[(0, 0)].tracks))
        self.trains.append(TrainCar())
        track.enter(self.trains[0], track.start, 0)

    def do_step(self):
        for car in self.trains:
            car.do_step()

    def scan_land(self):
        """Return lands in scanline order."""
        return (self.land[h] for h in self.hexes)

    def print(self):
        Hex.print_range(
                self.hexes,
                item=lambda h: self.land[h],
                hex_width=5,
                indent=lambda r: abs(r)/2)

