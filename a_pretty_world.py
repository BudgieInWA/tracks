#!/usr/bin/env python3

import sys
import collections
import math
import pygame
from pygame.locals import *
from Geometry import Point

import hexgrid

from a_world import *

import logging
log = logging.getLogger(__name__)


class RGB(collections.namedtuple("Colour", "r g b")):
    @staticmethod
    def clamped(val):
        return min(255, max(0, val))

    def __new__(cls, *args, **kwargs):
        args = (RGB.clamped(a) for a in args)
        kwargs = {k: RGB.clamped(v) for k, v in kwargs.items()}
        return super().__new__(cls, *args, **kwargs)

    def w(self, **kwargs):
        return RGB(
                r=kwargs.get("r") or self.r,
                g=kwargs.get("g") or self.g,
                b=kwargs.get("b") or self.b)

    def scaled(self, k):
        return RGB(r=k*self.r, g=k*self.g, b=k*self.b)


def pgp(point):
    """Format a Point for pygame.draw.* pos arguments."""
    return [round(point.x), round(point.y)]


FPS = 60
STEPS_PER_TURN = 15

# Constants representing colours
# TODO namespace
BLACK = RGB(  0,   0,   0)
WHITE = RGB(255, 255, 255)
BROWN = RGB(153,  76,   0)
RED   = RGB(255,   0,   0)
GREEN = RGB(  0, 255,   0)
BLUE  = RGB(  0,   0, 255)

pygame.font.init()
FONT = pygame.font.SysFont("sans-serif", 24)

# Useful game dimensions
WORLD_RADIUS = 3
HEX_BIG_RADIUS = 100
HEX_SMALL_RADIUS = HEX_BIG_RADIUS * 0.86602540378

SCREEN_WIDTH  = (4 * WORLD_RADIUS + 1) * HEX_BIG_RADIUS
SCREEN_HEIGHT = (4 * WORLD_RADIUS)     * HEX_BIG_RADIUS

layout = hexgrid.Layout(orientation=hexgrid.layout_pointy, size=Point(HEX_BIG_RADIUS, HEX_BIG_RADIUS),
                        origin=Point(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))

TRACK_WIDTH = int(HEX_BIG_RADIUS / 5)

# Set up the display
pygame.init()
DISPLAYSURF = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Set up hexworld
landscape = Landscape(radius=WORLD_RADIUS, seed=420)

clock = pygame.time.Clock()
step = 0

debug_gui = False
currently_building = False

# Event Loop
while True:
    clock.tick(FPS)

    log.debug("\nNew loop at time {}.".format(clock))

    # Find out what the mouse is pointing at.
    # Calculate where in the world the mouse is pointing.
    mouse_xy_pos = pygame.mouse.get_pos()
    mouse_hex_pos = hexgrid.pixel_to_hex(layout, Point(*mouse_xy_pos))
    mouse_hex_cubic = hexgrid.hex_round(mouse_hex_pos)
    mouse_hex = Hex(mouse_hex_cubic.r, mouse_hex_cubic.q)
    # Calculate which segment of the hex the mouse is in.
    off = hexgrid.Hex(mouse_hex_pos.r - mouse_hex_cubic.r, mouse_hex_pos.q - mouse_hex_cubic.q, mouse_hex_pos.s - mouse_hex_cubic.s)
    diff = (off.r - off.q, off.q - off.s, off.s - off.r)
    m = -1.0
    mi = None
    for i in range(len(diff)):
        if abs(diff[i]) > m:
            m = abs(diff[i])
            mi = i
    d = [0, 0, 0]
    sign = 1 if diff[mi] > 0 else -1
    d[mi] = sign
    d[(mi+1) % 3] = -sign
    mouse_dir = hexgrid.Hex(*d)

    log.debug("Mouse at {}".format(mouse_hex))

    # Handle the new events.
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == KEYDOWN:
            # Toggle debug GUI.
            if event.key == K_BACKQUOTE:
                debug_gui = not debug_gui

            # Speed up / Slow down the game
            if event.key == K_MINUS:
                log.info("Speed down.")
                STEPS_PER_TURN += 1
            if event.key == K_EQUALS:
                log.info("Speed up.")
                STEPS_PER_TURN -= 1

        # Activate current tool.
        # Currently build track.
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            # Building tracks.
            landscape.build_track_start()
            currently_building = True
        elif event.type == MOUSEBUTTONUP and event.button == 1:
            landscape.build_track_commit()
            currently_building = False

        # Print detailed info about thing under mouse.
        if event.type == MOUSEBUTTONDOWN and event.button == 2:
            print("Mouse at {}".format(mouse_hex))
            tile = landscape.tiles.get(mouse_hex)
            if not tile:
                print("Outside of the map.")
            else:
                print("\n".join(tile.strs()))

    if currently_building:
        landscape.build_track_select_hex(mouse_hex)

    #
    # Advance the game state.
    # TODO Advance time in turns, when the user gives input, or every 1 sec or something.

    # TODO Only advance the game state if we've passed a tick threshold.
    if step % STEPS_PER_TURN == 0:
        landscape.do_step()

    #
    # Draw everything, starting with a blank canvas.
    DISPLAYSURF.fill((20, 20, 20))
    # TODO Make the viewport move and zoom around the world.

    # Draw Landscape.
    for tile in landscape.scan_land():
        selected = False
        if tile.hex.r == mouse_hex_cubic.r and tile.hex.q == mouse_hex_cubic.q:
            selected = True
        colour = BROWN.scaled(1.5) if selected else BROWN

        center = hexgrid.hex_to_pixel(layout, tile.hex)

        # Draw tile background.
        pygame.draw.circle(
                DISPLAYSURF,
                colour,
                pgp(center),
                int(HEX_SMALL_RADIUS),
                0)

        # Draw tracks.
        for track in tile.tracks:
            if isinstance(track, StraightTrack):
                start_target_pos = hexgrid.hex_to_pixel(layout, tile.hex.add(track.start))
                start_pos = Point((center.x + start_target_pos.x) / 2, (center.y + start_target_pos.y) / 2)
                end_target_pos = hexgrid.hex_to_pixel(layout, tile.hex.add(track.end))
                end_pos = Point((center.x + end_target_pos.x) / 2, (center.y + end_target_pos.y) / 2)

                track.xy_start = start_pos
                track.xy_vector = end_pos - start_pos

                pygame.draw.line(
                        DISPLAYSURF,
                        WHITE,
                        pgp(start_pos),
                        pgp(end_pos),
                        TRACK_WIDTH)

            elif isinstance(track, CurvedTrack):
                arc_center = hexgrid.hex_to_pixel(layout, tile.hex.add(track.arc_center_dir))
                track.xy_arc_center = arc_center
                angle_to_hex_center = -pygame.math.Vector2(1, 0).angle_to(pygame.math.Vector2(center) - pygame.math.Vector2(arc_center))
                track.xy_start_angle = (angle_to_hex_center - 30) * math.pi / 180.0 # TODO figure out order
                track.xy_end_angle = (angle_to_hex_center + 31) * math.pi / 180.0
                R = HEX_BIG_RADIUS * 1.5

                pygame.draw.arc(
                        DISPLAYSURF,
                        BLUE if selected else WHITE,
                        [round(arc_center.x - R - TRACK_WIDTH/2), round(arc_center.y - R -
                            TRACK_WIDTH/2), R * 2 + TRACK_WIDTH, R * 2 + TRACK_WIDTH],
                        track.xy_start_angle,
                        track.xy_end_angle,
                        TRACK_WIDTH)

            if isinstance(track, Station):
                R = HEX_BIG_RADIUS * 0.3
                pygame.draw.rect(
                        DISPLAYSURF,
                        WHITE,
                        [round(center.x - R), round(center.y - R), 2 * R, 2 * R],
                        0)

        # Draw buildings.
        label = FONT.render("\n".join(str(b) for b in tile.buildings), False, BLACK)
        DISPLAYSURF.blit(label, pygame.math.Vector2(center) - (label.get_width()/2, label.get_height()/2))

        # Draw highlights.
        if tile.highlighted:
            pygame.draw.circle(
                    DISPLAYSURF,
                    GREEN,
                    pgp(center),
                    int(HEX_BIG_RADIUS * 0.5),
                    tile.highlighted)

    # Draw train cars.
    for car in landscape.trains:
        if not hasattr(car, 'colour'):
            car.colour = RGB(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        if car.track:
            car_xy = None
            if isinstance(car.track, StraightTrack):
                if car.track_facing == car.track.end:
                    car_xy = car.track.xy_start + car.track.xy_vector * car.track_pos
                elif car.track_facing == car.track.start:
                    car_xy = car.track.xy_start + car.track.xy_vector * (car.track.length - car.track_pos)
                else:
                    log.warning("{} is facing a direction that the track it is on does not connect to.".format(car))

            elif isinstance(car.track, CurvedTrack):
                R = HEX_BIG_RADIUS * 1.5
                angle_diff = car.track_pos
                if car.track_facing == car.track.end:
                    angle_diff = car.track.length - car.track_pos
                facing_sign = 1 if car.track_facing == car.track.end else -1
                if car.track.angle_dir * facing_sign < 1:
                    angle = car.track.xy_end_angle - angle_diff
                else:
                    angle = car.track.xy_start_angle + angle_diff
                car_x = car.track.xy_arc_center.x + math.cos(-angle) * R
                car_y = car.track.xy_arc_center.y + math.sin(-angle) * R
                car_xy = Point(car_x, car_y)

            if car_xy is not None:
                pygame.draw.circle(
                        DISPLAYSURF,
                        car.colour,
                        pgp(car_xy),
                        int(HEX_BIG_RADIUS / 5),
                        0)

                label = FONT.render(str(car.inventory), False, BLACK)
                xy = tuple(map(int, round(car_xy - (label.get_width()/2, label.get_height()/2)).xy))
                DISPLAYSURF.blit(label, xy)
            else:
                log.debug("not car_xy: {}".format(car_xy))

    # Draw debug GUI.
    if debug_gui:
        # Draw mouse coords

        tile = landscape.tiles.get(mouse_hex)
        if not tile:
            print()
        else:
            print()
        debug_sting = "({}, {}) in ({}, {}, {})\n{}".format(
            *mouse_xy_pos,
            *(round(x, 2) for x in mouse_hex_cubic),
            "\n".join(tile.strs()) if tile else "Outside of the map."
        )

        label = FONT.render(debug_sting, False, WHITE)
        DISPLAYSURF.blit(label, (0, 0))

        selected_hex_pos = pgp(Point(*hexgrid.hex_to_pixel(layout, mouse_hex_cubic)))
        pygame.draw.line(
                DISPLAYSURF,
                BLACK,
                selected_hex_pos,
                pgp(Point(*hexgrid.hex_to_pixel(layout, hexgrid.hex_add(mouse_hex_cubic, mouse_dir)))),
                3)
        pygame.draw.circle(
                DISPLAYSURF,
                BLACK,
                selected_hex_pos,
                round(HEX_BIG_RADIUS/5),
                0)

    # Update the display.
    pygame.display.update()

    step += 1
