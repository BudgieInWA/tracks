import logging
import random
import itertools

import string
from collections import namedtuple, defaultdict

from opensimplex import OpenSimplex
from clint.textui import colored, puts, min_width

log = logging.getLogger(__name__)


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


def labels_of_length(length):
    if length == 0:
        yield ""

    else:
        for letter in string.ascii_uppercase:
            for suffix in labels_of_length(length - 1):
                yield letter + suffix


def all_labels():
    length = 1
    while True:
        for label in labels_of_length(length):
            yield label
        length += 1

def label_for(obj):
    if hasattr(obj, "label"):
        return obj.label

    cls = obj.__class__
    if not hasattr(cls, "label_generator"):
        cls.label_generator = all_labels()

    return next(cls.label_generator)


def strs(ss, depth_limit=10, kids=None):
    """Render a tree of `strs` implementing objects."""
    if depth_limit == 0 or kids is None:
        return [ss]
    else:
        return [ss] + ["\t" + s for s in
                       itertools.chain.from_iterable(k.strs(depth_limit=depth_limit-1) for k in kids)]


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
    def print_range(hexes, item=lambda h: h, hex_width=2, indent=lambda r: 0):
        # Assume scanline ordering.
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

        self.tracks = set()  #TODO? list instead?
        self.buildings = []  #TODO set instead?

        self.highlighted = 0

    def do_step(self):
        for b in self.buildings:
            b.do_step()

    def add_track(self, track):
        """Add a track to the set, merging it with the others."""

        if track in self.tracks:
            return

        # Ensure new track isn't a duplicate.
        for other in self.tracks:
            if track.dirs == other.dirs:
                raise ValueError("Track is a duplicate of an existing track.")

        self.tracks.add(track)

    def strs(self, depth_limit=10):
        return strs(str(self), depth_limit=depth_limit, kids=itertools.chain(self.tracks, self.buildings))

    def __str__(self):
        return "<Tile at {}>".format(self.hex)


class Track:
    """A track on some tile that goes between edges."""

    def __init__(self, tile, length=1, dirs=set()):
        self.label = label_for(self)

        self.tile = tile
        self.length = length
        self.dirs = dirs

        self.cars = set()
        self.neighbours_at = defaultdict(lambda: set())

    def add_neighbour(self, dir, other):
        if dir not in self.dirs:
            return False

        self.neighbours_at[dir].add(other)
        return True

    @staticmethod
    def try_connect(track1, dir, track2):
        """Connect two tracks to one another if they meet."""
        log.debug("Trying to connect {} going {} to {}...".format(track1, dir, track2))

        other_dir = dir.scaled(-1)
        if track2.add_neighbour(other_dir, track1):
            if track1.add_neighbour(dir, track2):
                log.debug("\tyes.")
                return
            else:
                track2.remove_neightbour(other_dir, track1)
        log.debug("\tno.")
                
    def add_car(self, car):
        self.cars.add(car)
        car.track = self

    def enter(self, car, dir, dist):
        raise NotImplementedError("Track subclass must implement the enter method.")

    def remove_car(self, car):
        car.track = None
        self.cars.remove(car)

    def leave(self, car):
        self.remove_car(car)


    def strs(self, depth_limit=10):
        return strs(str(self), depth_limit=depth_limit, kids=self.cars)

    def __str__(self, name="Track"):
        return "<{} {} between dirs {}>".format(name, self.label, ", ".join(map(str, self.dirs)))


class Track2(Track):
    """A Track with two ends."""

    def __init__(self, tile, length, start_dir, end_dir):
        super().__init__(tile, length, dirs={start_dir, end_dir})

        self.start = start_dir
        self.end = end_dir

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

    def enter(self, car, dir, dist):
        super().add_car(car)

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

    def move_car(self, car, dist):
        """Move a car along the track."""

        car.track_pos += dist * car.track_facing

        # Is the car off the end?
        if car.track_pos > self.length:
            self.leave(car)
            if self.neighbours_at[self.end]:
                next_segment = car.choose_next_track(self.neighbours_at[self.end])
                next_segment.enter(car, self.end.scaled(-1), car.track_pos - self.length)
            else:
                car.speed = 0
                log.warning("Car {} crashed off the {} end of {}.".format(car, self.end, self))

        # Is the car off the start?
        if car.track_pos < 0:
            self.leave(car)
            if self.neighbours_at[self.start]:
                next_segment = car.choose_next_track(self.neighbours_at[self.start])
                next_segment.enter(car, self.start.scaled(-1), -car.track_pos)
            else:
                car.speed = 0
                print("Car {} crashed off the {} end of {}.".format(car, self.start, self))

    def __str__(self):
        return super().__str__("2Track")


class StraightTrack(Track2):
    """Track that goes from one edge to the opposite edge.
    
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

    def __str__(self):
        return Track.__str__(self, "CurvedTrack")


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

    def strs(self, depth_limit=10):
        return [self.__str__()]

    def __str__(self):
        return "{} ({}/{} @ {})".format(self.id, self.current, self.max, self.rate)


class Producer:
    def __init__(self, resources):
        self.resources = {r.id: r for r in resources}

    def resource_ids(self):
        return self.resources.keys()

    def get_productions(self):
        return self.resources.values()

    def do_step(self):
        for r in self.resources.values():
            r.do_step()

    def strs(self, depth_limit=10):
        return strs(self.__str__(), depth_limit=depth_limit, kids=self.resources)

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

    def strs(self, **kwargs):
        return strs(str(self), **kwargs)

    def __str__(self):
        return "All Purpose Shop"


class Forest(Producer):
    def __init__(self):
        super().__init__([ResourceProduction('wood', max=10.0, rate=0.01)])

    def strs(self, **kwargs):
        return strs(str(self), **kwargs)


class TrainCar:
    def __init__(self):
        self.label = label_for(self)

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
                            self.cargo_amount = 0.0
                        if self.cargo_amount == 0.0:
                            self.cargo_type = None

                if self.cargo_amount == 0.0:
                    self.collect_from_station(station)

                self.track_facing *= -1
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

    def strs(self, **kwargs):
        return strs(str(self), **kwargs)

    def __str__(self):
        return "<TrainCar carrying {}>".format(self.cargo_type)


class Landscape:
    def __init__(self, seed=0, radius=3):
        self.seed = seed
        self.shape = RandomShape(seed)

        self.radius = radius
        self.hexes = list(Hex.range(radius))

        # Create linked Tiles for the region.
        self.tiles = dict((h, Tile(h)) for h in self.hexes)

        self.trains = []

        self.build_path = None

        """
        # Generate some random tracks starting in the middle.
        rand = random.Random()
        rand.seed(self.seed)
        tile = self.tiles[Hex(0, 0)]
        start_dir = Hex.directions[0]
        last_track = None
        while tile:
            print("Track through {}".format(tile.hex))

            dir_id = Hex.direction_id[start_dir.scaled(-1)]
            end_dir = rand.choice([Hex.directions[dir_id], Hex.directions[(dir_id+1)%6], Hex.directions[(dir_id-1)%6]])
            track = Track2.make(tile, start_dir, end_dir)
            tile.tracks.add(track)

            if last_track:
                last_track.neighbours_at[self.end].add(track)
                track.neighbours_at[self.start].add(last_track)

            last_track = track
            tile = self.tiles.get(tile.hex.add(end_dir))
            start_dir = end_dir.scaled(-1)

        # Put a train on the track
        track = next(iter(self.tiles[(0, 0)].tracks))
        self.trains.append(TrainCar())
        track.enter(self.trains[0], track.start, 0)
        """

        self.build(Hex(0, 0), AllPurposeShop())
        self.build(Hex(2, 0), Forest())

    def do_step(self):
        for tile in self.tiles.values():
            tile.do_step()
        for car in self.trains:
            car.do_step()


    def highlight(self, hex):
        self.tiles[hex].highlighted += 1
    def dehighlight(self, hex):
        self.tiles[hex].highlighted -= 1

    def build_track_start(self):
        print("starting build track")
        self.build_path = []

    def build_track_select_hex(self, hex):
        p = self.build_path

        if hex not in self.tiles:
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
        if len(self.build_path) <= 1:
            return

        from_dir = None
        for i in range(len(self.build_path)):
            hex = self.build_path[i]
            tile = self.tiles[hex]

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

            log.debug("Building {} at {}".format(track, hex))
            try:
                self.build(hex, track)
                log.debug("\tyep.")
            except ValueError:
                log.debug("\tnope.")

            from_dir = to_dir.scaled(-1) if to_dir else None

        train = TrainCar()
        track.enter(train, track.start, 0)
        self.trains.append(train)

        self.build_path = None

    def build(self, hex, building):
        if isinstance(building, Track):
            track = building
            self.tiles[hex].add_track(track)

            # Connect the track to adjacent tracks.
            for dir in track.dirs:
                for other in self.tiles[hex.add(dir)].tracks:
                    Track.try_connect(track, dir, other)
        
        else:
            self.tiles[hex].buildings.append(building)


    def scan_land(self):
        """Return lands in scanline order."""
        return (self.tiles[h] for h in self.hexes)

    def print(self):
        Hex.print_range(
                self.hexes,
                item=lambda h: self.tiles[h],
                hex_width=5,
                indent=lambda r: abs(r)/2)

