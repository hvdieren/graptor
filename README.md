# Graptor

Graptor is an auto-vectorizing and optimizing compiler and runtime for efficient single-node graph analytics.

## Configuration and compilation

Recommended environment

* C++ compiler with C++20 and OpenMP support
* Cilkplus runtime clone with NUMA-awareness extension (borrowed from GraphGrind)
* cmake &gt;= 3.10

Cmake configuration flags are provided in the following way:

```
$ cmake -D FLAG=value
```

Relevant flags:

* CMAKE_CXX_COMPILER (or define CXX in the shell environment): the C++ compiler of your choice
* GRAPTOR_PARALLEL={cilk,cilk_numa,openmp,openmp_numa,seq} sets the way thread-level parallelism is handled
* GRAPTOR_TEST={ON,OFF} determines whether unit tests should be run
* GRAPTOR_BUILD_DOC={ON,OFF} determines whether documentation should be built

Configuration example

```
$ mkdir build
$ cd build
$ cmake -D GRAPTOR_PARALLEL=openmp_numa /path/to/graptor/CMakeLists.txt
$ make
```

* Executing sample analytics

Sample analytics programs are located in the graptor/bench directory. They can be executed in the following way:

```
$ bench/CCv_GGVEBO -rounds 3 -c 16 -itimes -s -b ../rMatGraph_J_5_100_b2

Common flags:

* "-rounds [n]" execute the analytic n times and report average execution time
* "-c [n]" partition the graph n-ways for parallel execution
* "-itimes" report execution time and frontier density for each iteration of the analytic
* "-debug" provide additional information to -itimes and provide it immediately
* "-s" the graph is symmetric/undirected
* "-b" the graph file is read from Graptor's custom binary format

## Input graph format

Input graphs are stored in a custom binary format, consisting of:
* A fixed-size header encoding version number of the format, and number of edges and vertices, bit width of vertex IDs and edge IDs.
* A binary dump of V values of the "index" array of the graph in compressed sparse rows format (listing outgoing edges), where V is the number of vertices. Each value is as wide as the edge ID, by default 8 bytes.
* A binary dump of E values of the "neighbour" array in compressed sparse rows format, where E is the number of edges. By default, these values are 4 bytes wide.

Several tools are provided to convert graphs or check properties:
* tools/CvtTextToBin: converts the adjacency graph format from the Problem Based Benchmark Suite (http://www.cs.cmu.edu/~pbbs/benchmarks/graphIO.html) to Graptor's format.
* tools/CvtBinToText: performs the conversion in opposite direction
* tools/CvtToTranspose: transposes a graph
* tools/CvtToUndir: creates a symmetric (undirected) copy of a graph
* tools/CvtToCSV: converts a graph to CSV format
* tools/CvtBinToGrazelle: converts a graph to Grazelle's format
* tools/CvtBinToEdgeList: converts a graph to a coordinate list format
* tools/CvtTwitter: converts specifically an input format used for a Twitter graph
* tools/CvtFriendster: converts specifically an input format used for the Friendster graph
* tools/IsEqual: checks if two graphs are equal
* tools/IsUndir: checks if a graph is undirected/symmetric

## Graph analytics

Currently, Graptor comes with the following example analytics:
* bench/BFSv.C: breadth-first search, identifying parent tree
* bench/BFSLVLv.C: breadth-first search, identifying level each vertex is at
* bench/CCv.C: connected components using label propagation
* bench/MISv.C: maximum independent set
* bench/FMv.C: radius estimation
* bench/PRv.C: PageRank, power iteration
* bench/APRv.C: Accelerated PageRank
* bench/GC_vary.C: Graph coloring
* bench/PR_vary.C: PageRank using reduced precision (16-bit floats)
* bench/APR_vary.C: Accelerated PageRank using reduced precision (16-bit floats)
* bench/BFv.C: Bellman-Ford mock-up
* bench/KCv.C: K-core decomposition
* bench/KCdss*v.C: K-core decomposition (multiple variations)
