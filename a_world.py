import string
from collections import namedtuple, defaultdict, deque
from enum import Enum

import random
import itertools

from opensimplex import OpenSimplex
from clint.textui import colored, puts, min_width

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)  # FIXME
log.info = log.critical
log.debug("DEBUG output on.")
log.info("INFO output on.")


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
    #TODO all the operators
    def __sub__(self, other):
        raise NotImplementedError()

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

    def __str__(self):
        name = self.__class__.__name__
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

    def enter(self, car, from_dir, dist):
        super().add_car(car)

        car.track_pos = 0
        if from_dir == self.start:
            car.track_facing = self.end
        elif from_dir == self.end:
            car.track_facing = self.start
        else:
            raise ValueError("could not add {car} to track because track doesn't have an end at {dir}"
                             .format(car=car, dir=from_dir))

        if dist > 0:
            self.move_car(car, dist)

    def move_car(self, car, dist):
        """Move a car along the track."""

        dist_fraction = dist / 10  # FIXME New length

        car.track_pos += dist_fraction

        # Is the car off the end?
        if car.track_pos > self.length:
            self.leave(car)
            facing = car.track_facing
            if self.neighbours_at[facing]:
                next_segment = car.choose_next_track(self.neighbours_at[facing])
                next_segment.enter(car, facing.scaled(-1), car.track_pos - self.length)
            else:
                car.speed = 0
                log.warning("Car {} crashed off the {} end of {}.".format(car, self.end, self))

        if car.track_pos < 0:
            log.warning("Car {} has a pos less than 0.".format(car))

    def __str__(self):
        return super().__str__()


class StraightTrack(Track2):
    """Track that goes from one edge to the opposite edge.
    
    Length is 1 by definition."""
    def __init__(self, tile, dir1):
        super().__init__(tile, 1.0, dir1, dir1.scaled(-1))

    def __str__(self):
        return super().__str__()

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
        return Track.__str__(self)


class Station(StraightTrack):
    def __str__(self):
        return super().__str__()


class Resource(Enum):
    Currency = 'currency'

    Wood = 'wood'
    Stone = 'stone'


class Inventory(dict):
    def __init__(self, resource_capacity):
        """
        :param resource_capacity: maps resource ids to capacity, or -1 for unlimited.
        """
        kv = None
        default = 0
        if resource_capacity is None:
            default = -1
        elif isinstance(resource_capacity, dict):
            kv = resource_capacity
        elif isinstance(resource_capacity, int):
            default = resource_capacity

        if kv is not None:
            self.caps = defaultdict(lambda: default, mapping=kv)  #TODO
        else:
            self.caps = defaultdict(lambda: default)

    # TODO
    # += and -= with bounds checks

    @staticmethod
    def trade(giver_inv, receiver_inv, resource, count, price):
        # TODO
        if giver_inv[resource] < count or receiver_inv.cap[resource] < count:
            return False
        if receiver_inv[Resource.Currency] < price:
            return False

        giver_inv[resource] -= count
        receiver_inv[resource] += count

        giver_inv[Resource.Currency] += price
        receiver_inv[Resource.Currency] += price

    # def strs(self, depth_limit=10):
    #     return strs(self.__str__(), depth_limit=depth_limit, kids=self.resources)

    def __str__(self):
        return ", ".join("{} ({:.1f})".format(r, c) for r, c in self.values() if c > 0) or "nothing"


class Home:
    def __init__(self):
        self.inventory = Inventory(resource_capacity=None)
        self.prices = {
                Resource.Wood: 2.0,
                }

    def do_step(self):
        pass

    def strs(self, **kwargs):
        return strs(str(self), kids=[self.inventory], **kwargs)

    def __str__(self):
        return "Home"


class Forest:
    def __init__(self, rate=2):
        self.inventory = Inventory({
            Resource.Wood: 100,
        })
        self.rate = rate

    def do_step(self):
        try:
            self.inventory[Resource.Wood] += self.rate
        except:
            pass

    def strs(self, **kwargs):
        return strs(str(self), **kwargs)


class TrackMovementPlan:
    """

    """
    def __init__(self, path=None):
        self.path = path

        self.current_path_index = -1

    def done(self):
        return self.current_path_index + 1 >= len(self.path)

    def get_next_track(self):
        if self.done():
            return None
        return self.path[self.current_path_index + 1]

    def advance_to(self, track):
        expected_track = self.get_next_track()
        if track != expected_track:
            raise ValueError("Cannot advance to {} as it is not in the plan: {}".format(track, self))

        self.current_path_index += 1

    def __str__(self):
        return "<TrackMovementPlan at {} through {}>".format(self.current_path_index, self.path)

class TrainCar:
    def __init__(self):
        self.label = label_for(self)

        self.inventory = Inventory(resource_capacity=10)  # TODO restrict to any one stack

        self.track = None
        self.track_pos = None
        self.track_facing = None
        self.speed = 0

        self.plan = None

        self.enact_some_plan()

    def enact_some_plan(self):
        """"""
        # TODO Look for a job to do, basically
        #self.enact_plan(some_plan)
        pass

    def enact_plan(self, plan):
        try:
            plan.advance_to(self.track)
            self.plan = plan
        except ValueError:
            log.warning("Plan doesn't start where I am ({}): {}".format(self.track.tile.hex, plan))



    def pathfind_to(self, target, length_limit=100, time_budget=1000):
        """
        Find a path through the track network to target hex(or maybe as close as possible).

        Flood fill to see if you can find the target.
        """
        origin = self.track
        if origin is None:
            log.warning("{} is trying to pathfind, but isn't on a track.".format(self))
            return None
        destination = None

        # Set up initial state.
        queue = deque()
        dists = defaultdict(lambda: float('inf'), {origin: 0})
        parents = {origin: None}
        for dir in origin.dirs:
            queue.append((origin, dir))

        # Explore one more node while we have more to explore.
        for t in range(time_budget):
            try:
                (track, from_dir) = queue.popleft()
                if track.tile.hex == target:
                    destination = track
                    break

                if dists[track] >= length_limit:
                    log.debug("Finishing due to length limit ({})".format(length_limit))
                    break

                # Explore down all connections to other dirs.
                for dir in track.dirs:
                    if dir != from_dir:
                        # Take note of the places we can go next.
                        for next_track in track.neighbours_at[dir]:
                            new_dist = dists[track] + 1
                            existing_dist = dists[next_track]

                            if new_dist < existing_dist:
                                dists[next_track] = new_dist
                                parents[next_track] = track
                                queue.append((next_track, dir))

            except IndexError as empty_queue:
                log.debug("Finishing due to empty queue.")
                break

        if destination is None:
            log.info("No path to {}.".format(target))
            return None

        # Walk the parents tree, building the path.
        path = deque()
        track = destination
        while track is not None:
            path.appendleft(track)
            track = parents[track]

        log.info("Found path: {}".format(path))
        return path


    def choose_next_track(self, tracks):
        if self.plan is not None:
            planned_track = self.plan.get_next_track()
            if planned_track in tracks:
                self.plan.advance_to(planned_track)
                return planned_track
            else:
                self.plan = None

        return random.choice(list(tracks))


    def do_step(self):
        if self.track is None:
            pass
        else:
            # Decide what movements to make: accelerate, coast, decelerate, turn around

            if self.plan is not None:
                if self.plan.done():
                    self.speed = 0  # TODO deceleration limit
                    self.plan = None  # TODO plan?
                else:
                    required_dir = self.plan.get_next_track().tile.hex.subtract(self.track.tile.hex)
                    if self.track_facing != required_dir:
                        log.info("{} turning around to follow path.".format(self))
                        self.track_facing = required_dir


                    self.speed = 1  # TODO acceleration over time

            if self.speed == 0:
                # Do trades and such.
                pass  # TODO
            elif self.speed > 0:
                # Actually cause the car to move.
                self.track.move_car(self, self.speed)
            else:
                log.warning("{} has -ve speed: {}".format(self, self.speed))


    def strs(self, **kwargs):
        return strs(str(self), kids=[self.inventory], **kwargs)

    def __str__(self):
        types = (r for r, c in self.inventory if c > 0)
        return "<TrainCar carrying {}>".format(", ".join(types))


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

        # TODO Use the random shape to add features to the map.
        self.build(Hex(0, 0), Home())
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
        log.debug("starting build track")
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
            raise ValueError("Cannot select hexes far away while building.")

    def build_track_commit(self):
        if len(self.build_path) > 1:
            from_dir = None
            for i in range(len(self.build_path)):
                hex = self.build_path[i]
                tile = self.tiles[hex]

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
                    # DEBUG Build a train car for free:
                    if i == 0:
                        self.build_car(track)
                except ValueError:
                    log.debug("\tnope.")

                from_dir = to_dir.scaled(-1) if to_dir else None
        elif len(self.build_path) == 1:
            # Try to send all trains here.
            target = self.build_path[0]
            log.info("Trying to send all trains to {}".format(target))
            for train in self.trains:
                path = train.pathfind_to(target)
                if path is not None:
                    train.enact_plan(TrackMovementPlan(path))

        self.build_track_cancel()

    def build_track_cancel(self):
        for hex in self.build_path:
            self.dehighlight(hex)
        self.build_path = None

    def build_car(self, track):
        train = TrainCar()
        track.enter(train, track.start, 0)
        self.trains.append(train)

    def build(self, hex, building):
        if isinstance(building, Track):
            track = building
            self.tiles[hex].add_track(track)

            # Connect the track to adjacent tracks.
            for dir in track.dirs:
                try:
                    for other in self.tiles[hex.add(dir)].tracks:
                        Track.try_connect(track, dir, other)
                except KeyError:
                    pass
        
        else:
            self.tiles[hex].buildings.append(building)

    def scan_land(self, bounding_rectangle=None):
        """Return lands in scanline order."""
        # TODO bounding_rectangle
        return (self.tiles[h] for h in self.hexes)

    def print(self):
        Hex.print_range(
                self.hexes,
                item=lambda h: self.tiles[h],
                hex_width=5,
                indent=lambda r: abs(r)/2)

