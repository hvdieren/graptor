cmake_minimum_required(VERSION 3.10)

# set project name and version
project(Graptor VERSION 0.1 LANGUAGES C CXX)

# specify C++ standard 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# where to install external projects
set_directory_properties(PROPERTIES EP_BASE "./external")

# init to minimal necessary arguments
set(graptor_cflags "-O4 -Wno-ignored-attributes -g -DLONG -UNDEBUG")
set(graptor_cxxflags "-O4 -Wno-ignored-attributes -ftemplate-backtrace-limit=0 -g -DLONG -UNDEBUG")

# architectures
set(var GRAPTOR_ARCH "native" CACHE STRING "target processor architecture")
string(APPEND graptor_cflags " -march=${GRAPTOR_ARCH}")
string(APPEND graptor_cxxflags " -march=${GRAPTOR_ARCH}")

# set configuration file
configure_file(include/graptor/config.h.in config.h)

# set path to header files to include binary directory so that config.h is found
# also add ROOT/include directory
include_directories("${PROJECT_BINARY_DIR}" "include")

#set(graptor_cflags "-Wno-ignored-attributes -ftemplate-backtrace-limit=0 -O4 -g -march=skylake-avx512 -DLONG -DMAX_VL=16")
#set(graptor_cflags "-Wno-ignored-attributes -ftemplate-backtrace-limit=0 -O4 -g -march=skylake -DLONG -DMAX_VL=16")
set(graptor_cflags "-Wno-ignored-attributes -ftemplate-backtrace-limit=0 -O4 -g -march=znver2 -DLONG -DMAX_VL=8")
#set(graptor_cflags "-Wno-ignored-attributes -ftemplate-backtrace-limit=0 -O0 -g -march=native -DLONG -DMAX_VL=1")
set(EXTRA_LIBS)

# Cilk-enabled C/C++ compiler
# only used to build Cilk runtime; it is ok if this is an older compiler
# C compiler to build Cilk runtime
set(var GRAPTOR_CILK_C_COMPILER "${CMAKE_C_COMPILER}" CACHE FILEPATH "C compiler")

# C++ compiler to build Cilk runtime
set(var GRAPTOR_CILK_CXX_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "C++ compiler")

## choice of parallel backend
set(GRAPTOR_PARALLEL cilk_numa CACHE STRING "Parallel backend chosen by user at configure time")
set_property(CACHE GRAPTOR_PARALLEL PROPERTY STRINGS seq cilk cilk_numa openmp openmp_numa parlay parlay_numa)

# Dependent projects for Cilk; built-in compiler support OpenMP
include(ExternalProject)

if(GRAPTOR_PARALLEL STREQUAL "cilk")
    ExternalProject_Add(
	    cilkrts_project
	    GIT_REPOSITORY "https://github.com/OpenCilk/cilkrts"
	    GIT_TAG "main"
	    CMAKE_ARGS -DCMAKE_C_COMPILER=${GRAPTOR_CILK_C_COMPILER}
	    	       -DCMAKE_CXX_COMPILER=${GRAPTOR_CILK_CXX_COMPILER}
		       -DCMAKE_INSTALL_PREFIX=.
    )

    add_library(GRAPTOR_BACKEND SHARED IMPORTED GLOBAL)
    ExternalProject_Get_Property(cilkrts_project BINARY_DIR)
    list(APPEND EXTRA_LIBS ${BINARY_DIR}/lib/libcilkrts.so.5)
    set(graptor_ld_preload ${BINARY_DIR}/lib/libcilkrts.so.5)
    set(graptor_parallel_variable CILK_NWORKERS)
    target_include_directories(GRAPTOR_BACKEND INTERFACE ${BINARY_DIR}/include)
    add_dependencies(GRAPTOR_BACKEND cilkrts_project)

    find_library(LIBDL NAMES dl)
    if(LIBDL)
	list(APPEND EXTRA_LIBS ${LIBDL})
    endif()
elseif(GRAPTOR_PARALLEL STREQUAL "cilk_numa")
    ExternalProject_Add(
	    cilkrts_numa_project
	    GIT_REPOSITORY "git@hpdc-gitlab.eeecs.qub.ac.uk:hvandierendonck/intel-cilk-runtime.git"
	    GIT_TAG "static"
	    CMAKE_ARGS -DCMAKE_C_COMPILER=${GRAPTOR_CILK_C_COMPILER}
	    	       -DCMAKE_CXX_COMPILER=${GRAPTOR_CILK_CXX_COMPILER}
		       -DCMAKE_INSTALL_PREFIX=.
    )

    add_library(GRAPTOR_BACKEND SHARED IMPORTED GLOBAL)
    ExternalProject_Get_Property(cilkrts_numa_project BINARY_DIR)
    list(APPEND EXTRA_LIBS ${BINARY_DIR}/lib/libcilkrts.so.5)
    set(graptor_ld_preload ${BINARY_DIR}/lib/libcilkrts.so.5)
    target_include_directories(GRAPTOR_BACKEND INTERFACE ${BINARY_DIR}/include)
    add_dependencies(GRAPTOR_BACKEND cilkrts_numa_project)

    # libraries
    set(graptor_cilkrts $ENV{HOME}/graptor/lib/libcilkrts.so)
    set(graptor_ld_preload LD_PRELOAD=${graptor_cilkrts})
    if(LIBNUMA)
	list(APPEND EXTRA_LIBS ${LIBNUMA})
    endif()

    find_library(LIBDL NAMES dl)
    if(LIBDL)
	list(APPEND EXTRA_LIBS ${LIBDL})
    endif()
elseif(GRAPTOR_PARALLEL STREQUAL "parlay")
    ExternalProject_Add(
	    parlay_project
	    GIT_REPOSITORY "https://github.com/cmuparlay/parlaylib"
	    GIT_TAG "master"
	    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=.
    )

    add_library(GRAPTOR_BACKEND SHARED IMPORTED GLOBAL)
    ExternalProject_Get_Property(parlay_project BINARY_DIR)
    target_include_directories(GRAPTOR_BACKEND INTERFACE ${BINARY_DIR}/include)
    add_dependencies(GRAPTOR_BACKEND parlay_project)
    set(graptor_parallel_variable PARLAY_NUM_THREADS)

    string(APPEND graptor_cxxflags " -pthread")
    string(APPEND graptor_cflags " -pthread")
    list(APPEND EXTRA_LIBS "-lpthread")
elseif(GRAPTOR_PARALLEL STREQUAL "parlay_numa")
    ExternalProject_Add(
	    parlay_project
	    GIT_REPOSITORY "https://github.com/cmuparlay/parlaylib"
	    GIT_TAG "master"
	    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=.
    )

    add_library(GRAPTOR_BACKEND SHARED IMPORTED GLOBAL)
    ExternalProject_Get_Property(parlay_project BINARY_DIR)
    target_include_directories(GRAPTOR_BACKEND INTERFACE ${BINARY_DIR}/include)
    add_dependencies(GRAPTOR_BACKEND parlay_project)
    set(graptor_parallel_variable PARLAY_NUM_THREADS)

    string(APPEND graptor_cxxflags " -pthread")
    string(APPEND graptor_cflags " -pthread")
    list(APPEND EXTRA_LIBS "-lpthread")

    # libraries
    find_library(LIBNUMA NAMES numa)
    if(LIBNUMA)
	list(APPEND EXTRA_LIBS ${LIBNUMA})
    endif()
elseif(GRAPTOR_PARALLEL STREQUAL "openmp" OR GRAPTOR_PARALLEL STREQUAL "openmp_numa")
    find_package(OpenMP REQUIRED)
    string(APPEND graptor_cflags " -fopenmp")
    list(APPEND EXTRA_LIBS "-fopenmp")

    if(GRAPTOR_PARALLEL STREQUAL "openmp_numa")
	find_library(LIBNUMA NAMES numa)
	if(LIBNUMA)
	    list(APPEND EXTRA_LIBS ${LIBNUMA})
	endif()
    endif()
    # just need to mention some target to make this work
    add_custom_target(GRAPTOR_BACKEND)
    set(graptor_parallel_variable OMP_NUM_THREADS)
elseif(GRAPTOR_PARALLEL STREQUAL "seq")
    # just need to mention some target to make this work
    add_custom_target(GRAPTOR_BACKEND)
    set(graptor_parallel_variable NOTHREADS)
endif()

# Building documentation?
option(GRAPTOR_BUILD_DOC "Build documentation" ON)
if(GRAPTOR_BUILD_DOC)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
	# set input and output files
	set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
	set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

	# request to configure the file
	execute_process(COMMAND bash "-c" "find ${CMAKE_CURRENT_SOURCE_DIR}/include -name '*.h' | sed -e 's/^/INPUT += /g'" OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/header-list)
	configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

	# note the option ALL which allows to build the docs together
	# with the application
	add_custom_target(doc_doxygen
	    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
	    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	    COMMENT "Generating API documentation with Doxygen"
	    VERBATIM)
    else(DOXYGEN_FOUND)
      message("Doxygen need to be installed to generate the doxygen documentation")
    endif(DOXYGEN_FOUND)
endif(GRAPTOR_BUILD_DOC)

get_target_property(parallel_include GRAPTOR_BACKEND INTERFACE_INCLUDE_DIRECTORIES)
string(APPEND graptor_cflags " -I${parallel_include}")
string(APPEND graptor_cxxflags " -I${parallel_include}")

# our object file(s)
add_library(graptorlib STATIC src/constants.C)
set_target_properties(graptorlib PROPERTIES COMPILE_FLAGS "${graptor_cxxflags}")
add_dependencies(graptorlib GRAPTOR_BACKEND)

# testing
option(GRAPTOR_TEST "Build and run test cases" OFF)
if(GRAPTOR_TEST)
    enable_testing()
    add_subdirectory(test)
endif(GRAPTOR_TEST)

# bench
add_subdirectory(bench)

# tools
add_subdirectory(tools)

# comb
add_subdirectory(comb)
