# Third-Party Licenses

This repository is an independent implementation of a fixed-city
Traveling Salesman Problem (TSP) simulator.

The project may draw architectural or implementation-level inspiration
from open-source software listed below. Third-party software is not
included unless explicitly stated.

## EvoGym

**Project:** EvoGym  
**Repository:** https://github.com/EvolutionGym/evogym  
**License:** MIT License  
**Copyright:** Evolution Gym contributors

EvoGym is used as a reference for general simulator/environment
organization and API design patterns.

The fixed-city TSP simulator in this repository is independently
implemented and does not include EvoGym's physics engine, robot
simulation, voxel representation, actuator system, or robot-specific
environment code.

### EvoGym License

EvoGym is distributed under the MIT License:

> MIT License

> Copyright (c) 2021 Evolution Gym contributors

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:

> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.

> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Current Third-Party Code Usage

At the initial fixed-city simulator stage:

- No EvoGym source files are copied directly.
- No EvoGym physics or robot simulation code is included.
- No EvoGym C++ simulator code is included.
- No EvoGym robot/environment implementation is included.
- The TSP implementation is independently written for this repository.

If third-party source code is incorporated in a future version,
the specific files, modifications, and applicable licenses will be
documented here.
