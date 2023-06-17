# Introduction
I realized while making the AI that we couldn't simply train it to take
raw image data as input, or even just normalized data that contained the endpoints
of the opponent's sword. This happens because we don't have a way to get raw image data.
We would have to gather a shitton of footage of it fighting against an IRL opponent. Otherwise,
it's impossible to get environmental reactions to the AI's actions. The solution
to this is to build a physics engine. The Vector engine already covers the base,
so what's left to do is to model the physics.

This document exists to explain the "Arena", which means the virtual world the AI
will train inside. The Arena's physics will be reasonably similar to the real world.


# Forces
Forces will be modeled in the traditional way, as Vectors. Every Force will induce an
acceleration and a torque on the target object. Forces can act upon one point, and one point
only. 

