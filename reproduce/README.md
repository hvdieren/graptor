# Overview

Shell scripts in this directory show how to build Graptor to reproduce some of the results in publications.

* ICS22.sh: builds relevant executables for the ICS'22 paper
    > H. Vandierendonck. 2022. Software-Defined Floating-Point Number Formats and Their Application to Graph Processing. *In 2022 International Conference on Supercomputing (ICS '22), June 28--30, 2022, Virtual Event, USA*. ACM, New York, NY, USA, 17 pages. https://doi.org/10.1145/3524059.3532360

# Instructions

1. Set up the build system using cmake and the instructions in the top-level directory. At the very least, build graptorlib.a (the very first build target)
2. Check and edit the script, in particular for the compiler name/version, and the backend. The defaults are set to compile a CilkPlus program.
3. Run the script
    $ ./ICS22.sh
