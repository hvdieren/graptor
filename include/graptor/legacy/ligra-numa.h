// -*- C++ -*-
// This code is part of the project "Ligra: A Lightweight Graph Processing
// Framework for Shared Memory", presented at Principles and Practice of
// Parallel Programming, 2013.
// Copyright (c) 2013 Julian Shun and Guy Blelloch
//
// This code is part of the project "Accelerating Graph Analytics by Utilising the Memory Locaity of Graph Partitioning", presented at ICPP, 2017 
// Copyright (c) 2017 Jiawen Sun, Hans Vandierendonck, Dimitrios S. Nikolopoulos
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// #include <iostream>
// #include <fstream>
// #include <stdlib.h>
// #include <cstring>
// #include <string>
// #include <assert.h>
// #include <algorithm>
// #include <sys/mman.h>
// #if NUMA && !defined(__APPLE__)
// #include <numa.h>
// #endif
#include "graptor/legacy/parallel.h"
#include "graptor/legacy/utils.h"
#include "graptor/legacy/graph-numa.h"
#include "graptor/legacy/IO.h"
#include "graptor/legacy/parseCommandLine.h"

// using namespace std;

// typedef std::pair<intT, intT> intTpair;
