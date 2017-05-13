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
        r = self.r
        q = self.q
        return (Hex(*h) for h in ((r+1, q), (r-1, q+1), (r, q-1), 
                (r-1, q), (r+1, q-1), (r, q+1)))


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


class Effect:
    def do_default(self):
        pass


class FlowWaterIn(Effect):
    def __init__(self, amount):
        self.amount = amount

    def do_default(self, land):
        land.water += self.amount
        if close_enough(land.water, 0):
            land.water = 0

class LandHex:
    def __init__(self, shape, hex):
        self.hex = hex
        self.neighbours = None # needs to be filled in for us

        self.changes = []

        self.height = None
        self.water = 0

        self.things = set() 

        self.shape = shape
        self.init()

    def init(self):
        """Populate state from random shape."""
        mix = [ # (amp, freq)
                (0.2, 0.7),
                (0.1, 0.3),
                (1, 0.08),
                (0.8, 0.01)]
        num = sum(m[0] * self.shape.simplex(self.hex.scaled(m[1])) for m in mix)
        num = num + 0.2
        num = num * abs(num) - 0.2
        self.height = 14 * num

        waterline = -3
        if waterline > self.height:
            self.water = waterline - self.height

    def calculate_effects(self):
        # Flow water around.
        if (self.water) > 0:
            # binary search to find the height at which the water on this hex would settle if it
            # flowed on top of hexes in the neighbourhood.
            low = self.height
            high = self.surface_height()
            for n in self.neighbours:
                if n.surface_height() < low:
                    low = n.surface_height()
                # The mean will never be higher than this hexes water hight because we don't alow
                # flowing up.

            while not close_enough(low, high):
                mid = (low + high) / 2
                water_required = sum(max(0, mid - n.surface_height()) for n in self.neighbours) + max(0, mid - self.height)
                if water_required > self.water:
                    high = mid
                else:
                    low = mid

            for n in self.neighbours:
                amount = low - n.surface_height() # TODO? max
                if (amount > 0):
                    n.add_change(FlowWaterIn(amount))
                    self.add_change(FlowWaterIn(-amount))

    def add_change(self, change):
        self.changes.append(change)

    def do_changes(self):
        for c in self.changes:
            #TODO find if any things want to handle this change, else
            # else do the default
            c.do_default(self)
        self.changes = []

    def surface_height(self):
        return self.height + self.water

    def __str__(self):
        #return "{},{}".format(self.hex.r, self.hex.q)

        h = "{:03.1f}".format(self.surface_height())
        if self.water > 0:
            h = colored.blue(h) 
        elif self.water < 0:
            h = colored.red(h)
            
        return "{}".format(
                #self.hex.r, 
                #self.hex.q,
                h)


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
        self.land = dict((h, LandHex(self.shape, h)) for h in self.hexes)

        # Link land hexes via neighbours.
        for l in self.land.values():
            l.neighbours = [self.land[n] for n in l.hex.neighbours() if n in self.land]

    def do_step(self):
        for l in self.land.values():
            l.calculate_effects()
        for l in self.land.values():
            l.do_changes()

    def scan_land(self):
        return (self.land[h] for h in self.hexes)

    def print(self):
        Hex.print_range(
                self.hexes,
                item=lambda h: self.land[h],
                hex_width=5,
                indent=lambda r: abs(r)/2)

class AntFarm:
    """A 2d slice of land for easier visualisation while debugging."""
    def __init__(self, seed=0, offset=0, width=5):
        self.seed = seed
        self.shape = RandomShape(seed)

        self.offset = offset
        self.width = width

        self.hexes = None
        self.land = None

        self.generate_a()

    def generate_a(self):
        """Generate a straight line of adjacent hexes."""
        self.hexes = []
        self.land = dict()

        last = None
        for q in range(self.width):
            hex = Hex(0, q)
            l = LandHex(self.shape, hex)

            # Do neighbours cos we're here.
            l.neighbours = []
            if last is not None:
                l.neighbours.append(last)
                last.neighbours.append(l)

            # Save the new landhex.
            self.hexes.append(hex)
            self.land[hex] = l
            last = l


    def do_step(self):
        for l in self.land.values():
            l.calculate_effects()
        for l in self.land.values():
            l.do_changes()

    def scan_land(self):
        return (self.land[h] for h in self.hexes)

    def print(self):
        print("Total water: {}".format(sum(l.surface_height() - l.height for l in self.land.values())))
        for l in self.scan_land():
            numbers = list()#(number, char)
            numbers.append((l.height, "|"))
            if l.water > 0:
                numbers.append((l.water, colored.blue("$")))
            elif l.water < 0:
                    numbers.append((l.water, colored.red("-")))

            #TODO do processing to deal with backtracking.
            print(
                min_width(
                    ("{}"*len(numbers)).format(*[char*int(num) for num, char in numbers]),
                    20
                )
                + " {}".format(l.surface_height())
            )



if __name__ == "__main__":
    """Demonstrate and debug some stuff."""

    """
    l = Landscape(radius=2)
    l.land[(0,2)].water = 20
    l.print()
    """
    l = AntFarm(width=30)
    l.land[(0,0)].water = 20
    l.land[(0,10)].water = 20
    l.print()

    timelimit = 500
    dt = 0.5

    flow_until = 200

    for x in range(timelimit):
        time.sleep(dt)
        print("\n\ndoing step")
        if flow_until and x < flow_until:
            l.land[(0,0)].add_change(FlowWaterIn(1))
        l.do_step()
        l.print()


