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

FPS = 1

#constants representing colours
BLACK = RGB(0,   0,   0  )
WHITE = RGB(255, 255, 255)
BROWN = RGB(153, 76,  0  )
GREEN = RGB(0,   255, 0  )
BLUE  = RGB(0,   0,   255)

pygame.font.init()
FONT = pygame.font.SysFont("sans-serif", 24)

#useful game dimensions

RADIUS = 10
TILESIZE  = 32
MAPWIDTH  = (4 * RADIUS + 1) * TILESIZE
MAPHEIGHT = (4 * RADIUS + 1) * TILESIZE

layout = Layout(orientation=layout_pointy, size=Point(TILESIZE, TILESIZE),
        origin=Point(MAPWIDTH/2, MAPHEIGHT/2))

#set up the display
pygame.init()
DISPLAYSURF = pygame.display.set_mode((MAPWIDTH, MAPHEIGHT))

# setup hexworld
landscape = Landscape(radius=RADIUS, seed=1332)
landscape.print()
landscape.land[(0,0)].water = 20

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
    landscape.land[(0,0)].add_change(FlowWaterIn(1))
    landscape.do_step()
    #landscape = Landscape(radius=RADIUS, seed = step)

    # Draw Landscape.
    for land in landscape.scan_land():
        colour = BLUE if land.water > 0 else GREEN
        # Use red as height indication.
        height_fraction = land.height / 30 + 0.5
        colour = colour.scaled(height_fraction)

        p = hex_to_pixel(layout, land.hex).rounded()

        pygame.draw.circle(
                DISPLAYSURF,
                colour,
                p,
                int(TILESIZE/1.2),
                0)

        if False:
            label = FONT.render(str(land.hex), False, WHITE)
            DISPLAYSURF.blit(label, pygame.math.Vector2(p) - (label.get_width()/2, label.get_height()/2))


    # Update the display.
    pygame.display.update()

    step += 1
