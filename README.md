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
This project uses cmake for multi-platform.
I confirmed on the following environments:

- Windows 10, 64-bit
	- Intel Core i7-6700@3.4GHz x 8 Core
	- NVIDIA GeForce GTX1070 (CUDA10.0)
	- Visual Studio 2017 64-bit
- Ubuntu 16.04 on VirtualBox
- Jetson Nano


## How to start the project
The following library/app are needed:

- cmake-gui (for Windows), cmake (for Linux)
- CUDA (not mandatory)

### Windows (Visual Studio)
1. Start cmake-gui 
2. Set the same path of top CMakeLists.txt (and README.md) to `Where is the sourcecode`
3. Set the path you want to create project files to `Where to build the binaries`
4. Click `Configure` and choose the project type which you want to generate (e.g. Visual Studio 15 2017 Win64)
5. Click `Generate`, then the project files are created
6. Open `GameOfLife.sln` and set `GameOfLife` project as a startup project, then build and run!!

Note:
In Visual Studio, you might need to `build` Cuda library (AlgorithmCudaNormal, AlgorithmCudaGroup projects) individually, for some reasons. Otherwise, the libray might not be updated.

### Linux
```
cd LifeGameFast
mkdir build && cd build
cmake ..
make -j4
cd GameOfLife
./GameOfLife &
```

## Acknowledge
The following libraries are used in this projects. Thank you!!
I made some modifications so that these libraries can be built with CMake.

- AntTweakBar
- Freeglut
