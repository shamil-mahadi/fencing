import time
import pygame as pg
import numpy as np

from math import floor, sin, cos, pi, log10, sqrt

from linalg import Vector, Wireframe
from config import PRECISION



"""
Notes from ChatGPT suggestions
------------------------------

User Interface: The robot should have a user interface that allows the fencer
to adjust the difficulty level and select specific training scenarios.
The interface could include a touchscreen or voice commands.

Safety: It is important to ensure that the robot is safe to use and does not cause injury to the fencer.
The robot should be programmed to stop immediately if the fencer's movements indicate that they are in danger.

Ideas
-----
Make a mobile app to sync with the robot, allowing two players to fence against each other through their
robots over the internet. Could be used for fencing training classes. Would also create a fun, accessible
gamified physical activity. Will need extremely realistic robotic arm movements for this.
"""

# Below is a poorly written demo of a spinning cube. This is just for testing purposes.
# The actual code is in the linalg module.


pg.init()
screen = pg.display.set_mode((700, 600))
clock = pg.time.Clock()
running = True
dt = 0
screen_height = screen.get_height()
screen_width = screen.get_width()

s = 100
test = Wireframe([(0,4), (0,3), (0,1), (1,5), (1,2), (2, 6), (2,3), (3, 7), (4,5), (4,7), (5,6), (6,7)],

                 Vector(-s, s, -s), Vector(s, s, -s), Vector(s, -s, -s), Vector(-s, -s, -s),
                 Vector(-s, s, s), Vector(s, s, s), Vector(s, -s, s), Vector(-s, -s, s))


while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.KEYDOWN:
            test.apply_angular_velocity(Vector(1/sqrt(3), 1/sqrt(3), 1/sqrt(3)), pi/2, 10, dt)
            
    
    screen.fill("white")

    lines = test.render_to_pg(screen_height, Vector(screen_width/2, screen_height/2))
    
    for line in lines:
        pg.draw.line(screen, "black", *line, width=4)

    pg.draw.circle(screen, "green", lines[0][0], 5)
    pg.draw.circle(screen, "blue", lines[2][1], 5)
    pg.draw.circle(screen, "red", lines[4][1], 5)
    pg.draw.circle(screen, "yellow", lines[1][1], 5)
    pg.draw.circle(screen, "purple", lines[0][1], 5)
    pg.draw.circle(screen, "black", lines[3][1], 5)
    pg.draw.circle(screen, "orange", lines[5][1], 5)
    pg.draw.circle(screen, "pink", lines[7][1], 5)
    test.update(dt)
    
    pg.display.flip()
    dt = 1/120
    clock.tick(120)

pg.quit()
