# NumPy JPEG

This is the solution to a part of a final project for [Stanford's CS 168](http://web.archive.org/web/20230620163410/https://web.stanford.edu/class/cs168/index.html). This repository contains a rather minimal implementation of a JPEG encoder written in NumPy. Unlike other Python implementations of a JPEG encoder, this implementation isn't a translation of a JPEG encoder written in C into Python. It is rewritten from the ground up in order to most closely follow common NumPy practices, like taking advantage of broadcasting and slicing whenever possible instead of using explicit for loops. This implementation is purely educational and is very slow, but we hope that this implementation is more concise and easier to understand for a Python programmer than alternatives.

Take a look at our entire project [here](https://www.overleaf.com/read/vrdrwhfkmsjg). The solution to the project can be found [here](https://drive.google.com/file/d/1NKW5pi6OiUkcE_E6d8l8C6oWGJ8Ry3U9/view?usp=sharing).