import sys
import random
import time
import itertools


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

def strs(ss, depth=0, kids=None):
    if depth == 0 or kids is None:
        return [ss]
    else:
        return [ss] + ["\t" + s for s in itertools.chain.from_iterable(k.strs(depth=depth-1) for k in kids)]

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

Hex.directions = [Hex(1, 0), Hex(1, -1), Hex(0, -1), Hex(-1, 0), Hex(-1, 1), Hex(0, 1)]
Hex.direction_id = dict((Hex.directions[i], i) for i in range(len(Hex.directions)))


class Tile:
    def __init__(self, hex):
        self.hex = hex

        self.tracks = set() #TODO? list instead?
        self.buildings = [] #TODO set instead?

        self.highlighted = 0



    def do_step(self):
        for b in self.buildings:
            b.do_step()

    def strs(self, depth=0):
        return strs(self.__str__(), depth=depth, kids=itertools.chain(self.tracks, self.buildings))

    def __str__(self):
        return "Tile at {}".format(self.hex)



class Track:
    """A track on some tile that goes between edges."""

    def __init__(self, tile, length):
        self.tile = tile
        self.length = length

    def leave(self, car):
        pass

    def enter(self, car):
        pass

class Track2(Track):
    """A Track with two ends."""

    def __init__(self, tile, length, start_dir, end_dir):
        super().__init__(tile, length)

        self.start = start_dir
        self.end = end_dir

        self.start_neighbours = set()
        self.end_neighbours = set()

        self.cars = set()

    @staticmethod
    def make(tile, dir1, dir2=None):
        if dir2 is None:
            return Station(tile, dir1)
        if dir1 == dir2.scaled(-1):
            return StraightTrack(tile, dir1)
        elif close_enough(dir1.distance(dir2), 2):
            return CurvedTrack(tile, dir1, dir2)
        else:
            raise ValueError("Cannot create 2track on tile {} between directions {} and {}".format(tile, dir1, dir2))

    def connecting_dirs(self):
        return [self.start, self.end]

    @staticmethod
    def connect(track1, track2):
        """Connect two tracks to oneanother."""
        track1.add_neighbour(track2);
        track2.add_neighbour(track1);
        #TODO Done?

    def add_neighbour(self, other, dir):
        if dir == self.start:
            self.start_neighbours.add(other)
            return True
        elif dir == self.end:
            self.end_neighbours.add(other)
            return True
        else:
            return False

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
            raise ValueError("could not add {car} to track because track doesn't go to direction {dir}"
                    .format(car=car, dir=dir))

        if dist > 0:
            self.move_car(car, dist)

    def leave(self, car):
        self.cars.remove(car)

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
            next_segment.enter(car, self.start.scaled(-1), -car.track_pos)
            self.leave(car)

    def strs(self, depth=0):
        return [self.__str__()]

    def __str__(self):
        return "2Track from {} to {}".format(self.start, self.end)


class StraightTrack(Track2):
    """Track that goes from one edge to the oposite edge.
    
    Length is 1 by definition."""
    def __init__(self, tile, dir1):
        super().__init__(tile, 1.0, dir1, dir1.scaled(-1))


class CurvedTrack(Track2):
    """Track that goes from one edge to an edge two spots away."""

    def __init__(self, tile, start_dir, end_dir):
        super().__init__(tile, 0.90689968211, start_dir, end_dir)

        self.arc_center_dir = None
        start_dir_id = Hex.direction_id[start_dir]
        end_dir_id = Hex.direction_id[end_dir]
        if (start_dir_id + 2) % len(Hex.directions) == end_dir_id:
            self.angle_dir = 1
            self.arc_center_dir = Hex.directions[(start_dir_id + 1) % len(Hex.directions)]
        elif (end_dir_id + 2) % len(Hex.directions) == start_dir_id:
            self.angle_dir = -1
            self.arc_center_dir = Hex.directions[(end_dir_id + 1) % len(Hex.directions)]
        else:
            raise ValueError("start_dir and end_dir are not two hexes from oneanother")



class Station(StraightTrack):
    def get_productions(self):
        nested_productions = (b.get_productions() for b in self.tile.buildings if isinstance(b, Producer))
        return itertools.chain.from_iterable(nested_productions)

    def get_consumers(self):
        return (b for b in self.tile.buildings if isinstance(b, Consumer))

class ResourceProduction:
    def __init__(self, id, max=1.0, rate=0.1):
        self.id = id
        self.max = max
        self.rate = rate
        self.current = 0.0

    def produce(self):
        self.current += self.rate
        self.current = min(self.max, self.current)

    def do_step(self):
        self.produce()

    def full(self):
        return self.current >= self.max

    def collect(self, max=None):
        """Collect the resource from the production."""

        amount = min(self.current // 1, max or self.max)
        self.current -= amount
        return int(amount)

    def strs(self, depth=0):
        return [self.__str__()]

    def __str__(self):
        return "{} ({}/{} @ {})".format(self.id, self.current, self.max, self.rate)

class Producer:
    def __init__(self, resources):
        self.resources = { r.id: r for r in resources }

    def resource_ids(self):
        return self.resources.keys()

    def get_productions(self):
        return self.resources.values()

    def do_step(self):
        for r in self.resources.values():
            r.do_step()

    def strs(self, depth=0):
        return strs(self.__str__(), depth=0, kids=self.resources)

    def __str__(self):
        return "Producer of {}".format(", ".join("{} ({:.1f})".format(r.id, r.current) for r in self.resources.values()))
        

class Consumer:
    def consume(self, resource, amount):
        pass

class AllPurposeShop(Consumer):
    def __init__(self):
        self.prices = {
                "wood": 2.0,
                }

    def accepts(self, resource):
        return resource in self.prices

    def consume(self, resource, amount):
        if resource in self.prices:
            print("Sold {} {} for {}.".format(amount, resource, amount * self.prices[resource]))
            return amount
        else:
            print("Cannot sell {}.".format(resource))
            return 0.0

    def do_step(self):
        pass

        
    def __str__(self):
        return "All Purpose Shop"

class Forest(Producer):
    def __init__(self):
        super().__init__([ResourceProduction('wood', max=10.0, rate=0.01)])


class TrainCar:
    def __init__(self):
        self.track = None
        self.track_pos = None
        self.track_facing = None
        self.speed = 0.05

        self.last_station = None
        self.cargo_type = None
        self.cargo_max = 1.0
        self.cargo_amount = 0.0

    def do_step(self):
        if False:
            print("{}: {} {} {}".format(self, self.track, self.track_pos, self.track_facing))

        if self.track:
            if (isinstance(self.track, Station) and self.track is not self.last_station and
                    self.track_pos > self.track.length * 0.3 and
                    self.track_pos < self.track.length * 0.7):
                # Arrived at a station.
                station = self.track

                if self.cargo_amount > 0.0:
                    consumer = next((c for c in station.get_consumers() if c.accepts(self.cargo_type)), None)
                    if consumer:
                        self.cargo_amount -= consumer.consume(self.cargo_type, self.cargo_amount)
                        if self.cargo_amount < 0.0:
                            cargo_amount = 0.0
                        if self.cargo_amount == 0.0:
                            self.cargo_type = None

                if self.cargo_amount == 0.0:
                    self.collect_from_station(station)

                self.track_facing *= -1;
                self.last_station = station
            else:
                self.track.move_car(self, self.speed)

    def choose_next_track(self, tracks):
        return random.choice(list(tracks))

    def collect_from_station(self, station):
        for production in station.get_productions():
            amount = production.collect(self.cargo_max)
            if amount > 0.0:
                self.cargo_type = production.id
                self.cargo_amount = amount
                return

class Landscape:
    def __init__(self, seed=0, radius=3):
        self.seed = seed
        self.shape = RandomShape(seed)

        self.radius = radius
        self.hexes = list(Hex.range(radius))

        # Create linked Tiles for the region.
        self.tile = dict((h, Tile(h)) for h in self.hexes)

        self.trains = []

        self.build_path = None

        """
        # Generate some random tracks starting in the middle.
        rand = random.Random()
        rand.seed(self.seed)
        tile = self.tile[Hex(0, 0)]
        start_dir = Hex.directions[0]
        last_track = None
        while tile:
            print("Track through {}".format(tile.hex))

            dir_id = Hex.direction_id[start_dir.scaled(-1)]
            end_dir = rand.choice([Hex.directions[dir_id], Hex.directions[(dir_id+1)%6], Hex.directions[(dir_id-1)%6]])
            track = Track2.make(tile, start_dir, end_dir)
            tile.tracks.add(track)

            if last_track:
                last_track.end_neighbours.add(track)
                track.start_neighbours.add(last_track)

            last_track = track
            tile = self.tile.get(tile.hex.add(end_dir))
            start_dir = end_dir.scaled(-1)

        # Put a train on the track
        track = next(iter(self.tile[(0, 0)].tracks))
        self.trains.append(TrainCar())
        track.enter(self.trains[0], track.start, 0)
        """

        self.build(Hex(0, 0), AllPurposeShop())
        self.build(Hex(2, 0), Forest())

    def do_step(self):
        for tile in self.tile.values():
            tile.do_step()
        for car in self.trains:
            car.do_step()


    def highlight(self, hex):
        self.tile[hex].highlighted += 1
    def dehighlight(self, hex):
        self.tile[hex].highlighted -= 1

    def build_track_start(self):
        print("starting build track")
        self.build_path = []

    def build_track_select_hex(self, hex):
        p = self.build_path

        if not self.tile[hex]:
            return

        elif len(p) == 0:
            p.append(hex)
            self.highlight(hex)

        # Check if the latest hex is still selected
        elif hex == p[len(p) - 1]:
            pass

        # Check for backtracking to shorten the path.
        elif len(p) > 1 and hex == p[len(p) - 2]:
            self.dehighlight(p.pop())

        # Simple adding of one hex.
        elif close_enough(hex.distance(p[len(p) - 1]), 1):
            # Check that the turn is not too sharp.
            while len(p) > 1:
                if close_enough(hex.distance(p[len(p) - 2]), 1):
                    self.dehighlight(p.pop())
                else:
                    break
            p.append(hex)
            self.highlight(hex)

        else:
            raise ValueError("Cannot select hexes far away while building")

    def build_track_end(self):
        if len(self.build_path) == 1:
            return

        from_dir = None
        for i in range(len(self.build_path)):
            hex = self.build_path[i]
            tile = self.tile[hex]

            self.dehighlight(hex)

            to_dir = None
            if i + 1 < len(self.build_path):
                to_dir = self.build_path[i + 1].subtract(hex)
            
            if from_dir and to_dir:
                track = Track2.make(tile, from_dir, to_dir)
            elif from_dir:
                track = Track2.make(tile, from_dir)
            else:
                track = Track2.make(tile, to_dir)

            self.build(hex, track)

            from_dir = to_dir.scaled(-1) if to_dir else None

        train = TrainCar()
        track.enter(train, track.start, 0)
        self.trains.append(train)

        self.build_path = None


    def build(self, hex, building):
        if isinstance(building, Track):
            self.tile[hex].tracks.add(building)

            # Connect the track to adjacent tracks.
            for d in building.connecting_dirs():
                n = self.tile.get(hex.add(d))
                if n:
                    for t in n.tracks:
                        if t.add_neighbour(building, d.scaled(-1)):
                            building.add_neighbour(t, d)


            # TODO fix collisions with other tracks

        
        else:
            self.tile[hex].buildings.append(building)


    def scan_land(self):
        """Return lands in scanline order."""
        return (self.tile[h] for h in self.hexes)

    def print(self):
        Hex.print_range(
                self.hexes,
                item=lambda h: self.tile[h],
                hex_width=5,
                indent=lambda r: abs(r)/2)

