This software is released under the MIT License, see LICENSE.txt.

# Fast Conway's Game of Life
 - This is a CPP project for Conway's Game of Life.
 - I use OpenGL, OpenMP and CUDA for high speed performance.

<img src=00_doc/capture01.jpg width=60% height=60%>
<img src=00_doc/Result.jpg width=60% height=60%>
<img src=00_doc/Diagram_Class.jpg width=60% height=60%>

* Link to Youtube

<a href="https://www.youtube.com/watch?v=XFuT3gJwPT0"><img src="http://img.youtube.com/vi/XFuT3gJwPT0/0.jpg" alt="Link to YouTube Video"></a>

## Environment
 - Visual Studio 2017 Community (need to install)
 - OpenGL (no need to install on Windows)
 - freeglut (contained in this project and setting is done)
 - AntTweakBar (contained in this project and setting is done)
 - CUDA Toolkit (need to install)
	- I use cuda_9.0.176


## How to begin from the beginning (if needed)
### Create a new project on Visual Studio
 - File -> New -> Project
 - Visual C++ -> Windows Desktop -> Windows Console Application

### Settings for OpenGL (freeglut)
#### Prepare freeglut library
- Build or retrieve freeglut library
	- freeglut/bin/(x64)/freeglut.dll
	- freeglut/lib/(x64)/freeglut.lib
	- freeglut/include/GL/freeglut.h,  freeglut_ext.h,  freeglut_std.h,  glut.h
- Copy the "freeglut" folder into the same folder as the project

#### Settings for freeglut library
- Include Path
	- Right Click on the project name (not the solution name) in solution explorer -> Properties
	- C/C++ -> General -> Additional Include Directories
	- ..\freeglut\include
- Linker Dependency
	- Linker -> Input -> Additional Dependenies
  	- ..\freeglut\lib\freeglut.lib
- Do the above two steps for x64 platform as well if needed
	- don't forget to modify library path adding "x64"
- Build
- Copy "freeglut.dll" into the same folder as the generated executable file (e.g. Debug folder)

#### Note
It may be better and general way to copy dll and lib files into your windows system folder ans visual studio folder.
I just don't like to messy my system folders.