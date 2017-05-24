#!/usr/bin/env python3

import pygame, sys
from pygame.locals import *
import collections
import math

import hexgrid
Point = hexgrid.Point

from a_world import *

class RGB(collections.namedtuple("Colour", "r g b")):

    @staticmethod
    def clamped(val):
        return min(255, max(0, val))

    def __new__(self, *args, **kwargs):
        args = (RGB.clamped(a) for a in args)
        kwargs = {k: RGB.clamped(v) for k, v in kwargs.items()}
        return super().__new__(self, *args, **kwargs)

    def w(self, **kwargs):
        return RGB(
                r = kwargs.get("r") or self.r,
                g = kwargs.get("g") or self.g,
                b = kwargs.get("b") or self.b)

    def scaled(self, k):
        return RGB(r=k*self.r, g=k*self.g, b=k*self.b)

FPS = 15

#constants representing colours
BLACK = RGB(  0,   0,   0)
WHITE = RGB(255, 255, 255)
BROWN = RGB(153,  76,   0)
RED   = RGB(255,   0,   0)
GREEN = RGB(  0, 255,   0)
BLUE  = RGB(  0,   0, 255)

pygame.font.init()
FONT = pygame.font.SysFont("sans-serif", 24)

#useful game dimensions

RADIUS = 3 
TILESIZE  = 100 # outer hex radius
WIDTH_HEIGHT_RATIO = 0.86602540378 # ratio between outer radius and inner radius
MAPWIDTH  = (4 * RADIUS + 1) * TILESIZE
MAPHEIGHT = (4 * RADIUS)     * TILESIZE

layout = hexgrid.Layout(orientation=hexgrid.layout_pointy, size=Point(TILESIZE, TILESIZE),
        origin=Point(MAPWIDTH/2, MAPHEIGHT/2))

TRACK_WIDTH = int(TILESIZE / 5)

#set up the display
pygame.init()
DISPLAYSURF = pygame.display.set_mode((MAPWIDTH, MAPHEIGHT))

# setup hexworld
landscape = Landscape(radius=RADIUS, seed=420)

clock = pygame.time.Clock()
step = 0

currently_building = False

vomit = False

# Event Loop
while True:
    clock.tick(FPS)

    if vomit:
        print("=============================")

    # Find out what the mouse is pointing at.
    mouse_xy_pos = pygame.mouse.get_pos()
    mouse_hex_pos = hexgrid.pixel_to_hex(layout, Point(*mouse_xy_pos))
    mouse_hex_cubic = hexgrid.hex_round(mouse_hex_pos)
    mouse_hex = Hex(mouse_hex_cubic.r, mouse_hex_cubic.q)
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

    # Get all the user events
    for event in pygame.event.get():
        #if the user wants to quit
        if event.type == QUIT:
            #and the game and close the window
            pygame.quit()
            sys.exit()

        # left click
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            if currently_building:
                landscape.build_track_end()
                currently_building = False
            else:
                landscape.build_track_start()
                currently_building = True

        if event.type == KEYDOWN and event.key == K_i:
            print("Mouse at {}".format(mouse_hex))
            land = landscape.land.get(mouse_hex)
            if not land: continue
            print(land.str(depth=1))


    if currently_building:
        landscape.build_track_select_hex(mouse_hex)
    

    # Advance the game state.
    # TODO only advance the game state if we've passed a tick threshold
    try:
        landscape.do_step()
    except BaseException as e:
        print(e)


    # Refresh the canvas
    DISPLAYSURF.fill((20, 20, 20))
    
    # Draw Landscape.
    for land in landscape.scan_land():
        selected = False
        if land.hex.r == mouse_hex_cubic.r and land.hex.q == mouse_hex_cubic.q:
            selected = True
        colour = BROWN.scaled(1.5) if selected else BROWN

        center = hexgrid.hex_to_pixel(layout, land.hex).rounded()

        # draw bg
        pygame.draw.circle(
                DISPLAYSURF,
                colour,
                center,
                int(TILESIZE * WIDTH_HEIGHT_RATIO),
                0)

        # draw tracks
        for track in land.tracks:
            if isinstance(track, StraightTrack):
                start_target_pos = hexgrid.hex_to_pixel(layout, land.hex.add(track.start))
                start_pos = Point((center.x + start_target_pos.x) / 2, (center.y + start_target_pos.y) / 2).rounded()
                end_target_pos = hexgrid.hex_to_pixel(layout, land.hex.add(track.end))
                end_pos = Point((center.x + end_target_pos.x) / 2, (center.y + end_target_pos.y) / 2).rounded()

                track.xy_start = pygame.math.Vector2(start_pos)
                track.xy_vector = pygame.math.Vector2(end_pos) - pygame.math.Vector2(start_pos)

                pygame.draw.line(
                        DISPLAYSURF,
                        WHITE,
                        start_pos,
                        end_pos,
                        TRACK_WIDTH)
                if vomit:
                    pygame.draw.circle(
                            DISPLAYSURF,
                            WHITE,
                            start_pos,
                            TRACK_WIDTH)

            elif isinstance(track, CurvedTrack):
                arc_center = hexgrid.hex_to_pixel(layout, land.hex.add(track.arc_center_dir)).rounded()
                track.xy_arc_center = arc_center
                angle_to_hex_center = -pygame.math.Vector2(1, 0).angle_to(pygame.math.Vector2(center) - pygame.math.Vector2(arc_center))
                track.xy_start_angle = (angle_to_hex_center - 30) * math.pi / 180.0 # TODO figure out order
                track.xy_end_angle = (angle_to_hex_center + 31) * math.pi / 180.0
                R = TILESIZE * 1.5

                pygame.draw.arc(
                        DISPLAYSURF,
                        BLUE if selected else WHITE,
                        [arc_center.x - R - TRACK_WIDTH/2, arc_center.y - R - TRACK_WIDTH/2, R * 2 + TRACK_WIDTH, R * 2 + TRACK_WIDTH],
                        track.xy_start_angle,
                        track.xy_end_angle,
                        TRACK_WIDTH)

            if isinstance(track, Station):
                R = TILESIZE * 0.3
                pygame.draw.rect(
                        DISPLAYSURF,
                        WHITE,
                        [center.x - R, center.y - R, 2 * R, 2 * R],
                        0)


        # draw buildings
        label = FONT.render("\n".join(str(b) for b in land.buildings), False, BLACK)
        DISPLAYSURF.blit(label, pygame.math.Vector2(center) - (label.get_width()/2, label.get_height()/2))


        # draw highlights
        if land.highlighted:
            pygame.draw.circle(
                    DISPLAYSURF,
                    GREEN,
                    center,
                    int(TILESIZE * 0.5),
                    land.highlighted)

        if False:
            label = FONT.render(str(land.hex), False, WHITE)
            DISPLAYSURF.blit(label, pygame.math.Vector2(p) - (label.get_width()/2, label.get_height()/2))

    # draw train cars
    for car in landscape.trains:
        if not hasattr(car, 'colour'):
            car.colour = RGB(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        if car.track:
            car_xy = None
            if isinstance(car.track, StraightTrack):
                car_xy = car.track.xy_start + car.track.xy_vector * car.track_pos

            elif isinstance(car.track, CurvedTrack):
                R = TILESIZE * 1.5
                angle_diff = car.track_pos
                if car.track_facing < 0:
                    angle_diff = car.track.length - angle_diff
                if car.track.angle_dir * car.track_facing < 1:
                    angle = car.track.xy_end_angle - angle_diff
                else:
                    angle = car.track.xy_start_angle + angle_diff
                car_x = car.track.xy_arc_center.x + math.cos(-angle) * R
                car_y = car.track.xy_arc_center.y + math.sin(-angle) * R
                car_xy = pygame.math.Vector2(car_x, car_y)

        if car_xy:
            pygame.draw.circle(
                    DISPLAYSURF,
                    car.colour,
                    Point(*car_xy).rounded(),
                    int(TILESIZE / 5),
                    0)
            if car.cargo_type:
                label = FONT.render("{} {}".format(car.cargo_type, car.cargo_amount), False, BLACK)
                DISPLAYSURF.blit(label, car_xy - (label.get_width()/2, label.get_height()/2))



    label = FONT.render("{}, {}, {}".format(*(round(x, 2) for x in mouse_hex_cubic)), False, WHITE)
    DISPLAYSURF.blit(label, (0, 0))

    selected_hex_pos = Point(*hexgrid.hex_to_pixel(layout, mouse_hex_cubic)).rounded()
    pygame.draw.line(
            DISPLAYSURF,
            BLACK,
            selected_hex_pos,
            Point(*hexgrid.hex_to_pixel(layout, hexgrid.hex_add(mouse_hex_cubic, mouse_dir))).rounded(),
            3)
    pygame.draw.circle(
            DISPLAYSURF,
            BLACK,
            selected_hex_pos,
            round(TILESIZE/5),
            0)


    # Update the display.
    pygame.display.update()

    step += 1
