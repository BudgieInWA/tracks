#!/usr/bin/env python3

from a_world import *

import pygame, sys
from pygame.locals import *

from hexgrid import *

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

RADIUS = 2 
TILESIZE  = 128
MAPWIDTH  = (4 * RADIUS + 1) * TILESIZE
MAPHEIGHT = (4 * RADIUS)     * TILESIZE

layout = Layout(orientation=layout_pointy, size=Point(TILESIZE, TILESIZE),
        origin=Point(MAPWIDTH/2, MAPHEIGHT/2))

#set up the display
pygame.init()
DISPLAYSURF = pygame.display.set_mode((MAPWIDTH, MAPHEIGHT))

# setup hexworld
landscape = Landscape(radius=RADIUS, seed=420)
#landscape.land[(0,0)].water = 20
train_cars = [TrainCar(next(iter(landscape.land[(0, 0)].tracks)))]


clock = pygame.time.Clock()
step = 0

# Event Loop
while True:
    clock.tick(FPS)

    # Get all the user events
    for event in pygame.event.get():
        #if the user wants to quit
        if event.type == QUIT:
            #and the game and close the window
            pygame.quit()
            sys.exit()


    # Advance the game state.
    # TODO only advance the game state if we've passed a tick threshold
    landscape.do_step()
    for t in train_cars:
        t.do_step()


    # Debug: mark tracks that have cars on them
    for car in train_cars:
        if car.track_segment:
            car.track_segment.had_car = True

    # Draw Landscape.
    for land in landscape.scan_land():
        colour = BROWN

        center = hex_to_pixel(layout, land.hex).rounded()

        # draw bg
        pygame.draw.circle(
                DISPLAYSURF,
                colour,
                center,
                int(TILESIZE/1.2),
                0)

        # draw tracks
        for track in land.tracks:
            target_pos = hex_to_pixel(layout, land.hex.add(track.dir))
            line_end = Point((center.x + target_pos.x) / 2, (center.y + target_pos.y) / 2).rounded()

            track.xy_center = pygame.math.Vector2(center)
            track.xy_vector = pygame.math.Vector2(line_end) - track.xy_center

            pygame.draw.line(
                    DISPLAYSURF,
                    BLUE if track.had_car else WHITE,
                    center,
                    line_end,
                    10)

        if False:
            label = FONT.render(str(land.hex), False, WHITE)
            DISPLAYSURF.blit(label, pygame.math.Vector2(p) - (label.get_width()/2, label.get_height()/2))

    # draw train cars
    for car in train_cars:
        if car.track_segment:
            car_xy = car.track_segment.xy_center + car.track_segment.xy_vector * car.track_pos
            pygame.draw.circle(
                    DISPLAYSURF,
                    RED,
                    Point(*car_xy).rounded(),
                    int(TILESIZE / 5),
                    0)

    # Update the display.
    pygame.display.update()

    step += 1
