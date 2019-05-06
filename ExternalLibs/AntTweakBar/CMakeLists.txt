cmake_minimum_required(VERSION 2.8)
project(AntTweakBar)


include_directories(include)


set(SOURCES
	src/TwColors.cpp
	src/TwFonts.cpp
	src/TwOpenGL.cpp
	src/TwOpenGLCore.cpp
	src/TwBar.cpp
	src/TwMgr.cpp
	src/TwPrecomp.cpp
	src/LoadOGL.cpp
	src/LoadOGLCore.cpp
	src/TwEventGLFW.c
	src/TwEventGLUT.c
	src/TwEventSDL.c
	src/TwEventSDL12.c
	src/TwEventSDL13.c
	src/TwEventSFML.cpp
	include/AntTweakBar.h
)

if(WIN32)
	set(SOURCES
		${SOURCES}
		src/TwEventWin.c
	)
else()
set(SOURCES
	${SOURCES}
		src/TwEventX11.c
	)
endif()

add_definitions(-DTW_STATIC)
add_definitions(-DTW_NO_LIB_PRAGMA)

add_library(AntTweakBar ${SOURCES})


