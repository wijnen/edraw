# Edraw

This is a Python module for generating pretty electronic diagrams and animations of the charges flowing through it.

# How to use

The recommended way to use this module is as follows:

  1. Make a sketch of the diagram on paper.
  1. Draw axes around it (with the origin in the lower left corner).
  1. Write down numbers at the axes for every position where a component or wire node is.
  1. Copy the diagram into a Python program.
  1. Add the lines to generate output.
  1. Run the program.

# Writing the diagram with Python

The program must import edraw and create an edraw.Schematic() object. This document assumes it is called 's'.

Then the components must be defined. They have the form: name = s.Add(edraw.Type, (x, y), ...). This command has the following properties:

  * name: the name of the variable holding the component. This can be chosen freely.
  * Type: the type of component that is created. This must be one of the defined types from the module.
  * (x, y): the location of the component in the diagram. This is read from the sketch.
  * ...: Any extra arguments that the module supports. See below.

When components are defined, they must be connected with wires. Most components have two terminals. The component's terminals can be accessed similar to the items in a list.

To connect two components, the function s.Wire() is used. It takes two or more
arguments. The first and last argument must be an existing terminal. Any
arguments in between must be tuples of x and y coordinates. The wire will go
from the first terminal, through all the specified locations, to the final
terminal.

A new terminal is created at every specified location. The return value of
edraw.Wire is a list of newly created terminals. This does not include the
first and last terminal that were passed to the function.

The direction of a wire has no effect on the direction of the current. The only
difference is how the debugging output (potential and current) are drawn beside
it.

# Common arguments

The following arguments are supported (on all components, unless otherwise specified):

  * U: voltage drop over the component. In the case of a battery, the current is expected to go in the reverse direction, so from terminal 1 to terminal 0.
  * I: current through the component.
  * V: potential of terminal 1 of the component. The first component that is defined has a default value of 0 for this. This should usually be the battery.
  * R: resistance of the component.
  * angle: rotation of the component in the diagram. A positive value is counter clockwise. The units are degrees.
  * open: (Switch only) Bool value describing whether the switch is open or closed.
  * draw\_I: Bool value describing whether the current should be drawn above the component. Default is False.
  * draw\_U: Bool value describing whether the voltage should be drawn below the component. Default is False.
  * draw\_R: (Resistor only) Bool value describing whether its value should be drawn in the box. Default is True.
  * digits: Number of digits to display for any drawn values. Default is 3.

# Components

The following components have been implemented:

  * Battery
  * Lamp
  * Switch
  * Resistor
  * Diode
  * LED
  * Ameter
  * Vmeter

# SVG Output

To generate an SVG image, first the compute() member function of the Schemetic
object should be called. Then the svg() member function will return a string
that can be written to a file.

# 2-D animated output

To generate a 2-D animation, the animate() member function of the Schematic
object should be called. It will call compute() repeatedly and output a new
image for every frame. When all frames are generated, it will combine them into
a video file. animate takes the total time in seconds and the video filename as
arguments, and optionally also the number of frames per second.

# Computations

The module can compute the values of all the properties if enough information is provided.

# Animating changes

If the schematic should change during the animation, a function can be passed
as an argument instead of a value. For example, to open a switch after 5
seconds, the argument that should be passed is open = lambda t: t > 5.
