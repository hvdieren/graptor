// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_CONTRACT_CONTRACT_H
#define GRAPTOR_GRAPH_CONTRACT_CONTRACT_H

namespace contract {

/**
 * Represent a graph by a number of components of pre-defined structure
 * (e.g., clique, path, star/claw).
 * The components form a partition of the edge set of the graph.
 *
 * Per-vertex info:
 * + Path: intermediate vertices have only one pattern - path
 * + Path: end-points may have multiple patterns; of which one is path
 *
 * Per-pattern info:
 * + Path: each path has ID, length, two end-points, set of intermediate
 *         vertices
 *         (size: pattern ID, VID, VID * length;
 *          + length * ( pattern type + pattern ID ) for vertex linkage)
 *         (size baseline: ~ length * (EID index+2 * VID edges))
 */

} // namespace contract

#endif // GRAPTOR_GRAPH_CONTRACT_CONTRACT_H
