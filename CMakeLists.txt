cmake_minimum_required(VERSION 3.10)

# set project name and version
project(Graptor VERSION 0.1)

# specify C++ standard 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# set configuration file
configure_file(include/graptor/config.h.in config.h)

# set path to header files to include binary directory so that config.h is found
# also add ROOT/include directory
include_directories("${PROJECT_BINARY_DIR}" "include")

set(graptor_cflags "-Wno-ignored-attributes -ftemplate-backtrace-limit=0 -O4 -g -march=skylake-avx512 -DLONG -DMAX_VL=16")
#set(graptor_cflags "-Wno-ignored-attributes -ftemplate-backtrace-limit=0 -O4 -g -march=znver2 -DLONG -DMAX_VL=8")
#set(graptor_cflags "-Wno-ignored-attributes -ftemplate-backtrace-limit=0 -O0 -g -march=native -DLONG -DMAX_VL=1")
set(EXTRA_LIBS)

## choice of parallel backend
set(GRAPTOR_PARALLEL cilk_numa CACHE STRING "Parallel backend chosen by user at configure time")
set_property(CACHE GRAPTOR_PARALLEL PROPERTY STRINGS seq cilk cilk_numa openmp openmp_numa)
if(GRAPTOR_PARALLEL STREQUAL "cilk" OR GRAPTOR_PARALLEL STREQUAL "cilk_numa")
    add_library(CILKRTS SHARED IMPORTED)
    target_include_directories(CILKRTS
	      INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
	      )
    string(APPEND graptor_cflags " -fcilkplus -DCILK=1 -I/home/hvandierendonck/research/asap/stable/jacob/include")
    if(GRAPTOR_PARALLEL STREQUAL "cilk_numa")
	string(APPEND graptor_cflags " -DNUMA=1")
    endif()

    # dependencies
    find_library(LIBNUMA NAMES numa)
    find_library(LIBDL NAMES dl)

    # libraries
    set(graptor_cilkrts ${CMAKE_CURRENT_SOURCE_DIR}/reproduce/lib/libcilkrts.so)
    set(graptor_ld_preload LD_PRELOAD=${graptor_cilkrts})
    if(LIBNUMA)
	list(APPEND EXTRA_LIBS ${graptor_cilkrts} ${LIBNUMA} ${LIBDL})
    endif()
elseif(GRAPTOR_PARALLEL STREQUAL "openmp" OR GRAPTOR_PARALLEL STREQUAL "openmp_numa")
    find_package(OpenMP REQUIRED)
    string(APPEND graptor_cflags " -fopenmp")
    list(APPEND EXTRA_LIBS "-fopenmp")
    if(GRAPTOR_PARALLEL STREQUAL "openmp_numa")
	string(APPEND graptor_cflags " -DNUMA=1")
    endif()
    find_library(LIBNUMA NAMES numa)
    if(LIBNUMA)
	list(APPEND EXTRA_LIBS ${LIBNUMA})
    endif()
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

	# note the option ALL which allows to build the docs together with the application
	add_custom_target(doc_doxygen ALL
	    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
	    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	    COMMENT "Generating API documentation with Doxygen"
	    VERBATIM)
    else(DOXYGEN_FOUND)
      message("Doxygen need to be installed to generate the doxygen documentation")
    endif(DOXYGEN_FOUND)
endif(GRAPTOR_BUILD_DOC)

# our object file(s)
add_library(graptorlib STATIC src/constants.C)
set_target_properties(graptorlib PROPERTIES COMPILE_FLAGS "${graptor_cflags}")

# testing
option(GRAPTOR_TEST "Build and run test cases" OFF)
if(GRAPTOR_TEST)
    enable_testing()
    add_subdirectory(test)
endif(GRAPTOR_TEST)

# bench
add_subdirectory(bench)

# tools
#add_subdirectory(tools)
