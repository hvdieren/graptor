// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_TYPES_H
#define GRAPTOR_DSL_EMAP_TYPES_H

enum class graph_traversal_kind {
    gt_sparse = 0,
    gt_pull = 1,
    gt_push = 2,
    gt_ireg = 3,
    gt_N = 4
};

extern const char * graph_traversal_kind_names[
    static_cast<std::underlying_type_t<graph_traversal_kind>>(
	graph_traversal_kind::gt_N )+1];

static std::ostream &
operator << ( std::ostream & os, graph_traversal_kind gtk ) {
    using T = std::underlying_type_t<graph_traversal_kind>;
    T igtk = (T) gtk;
    if( igtk >= 0 && igtk < (T)graph_traversal_kind::gt_N )
	return os << graph_traversal_kind_names[igtk];
    else
	return os << graph_traversal_kind_names[(int)graph_traversal_kind::gt_N];
}

#endif // GRAPTOR_DSL_EMAP_TYPES_H
