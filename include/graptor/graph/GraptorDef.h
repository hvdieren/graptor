// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_GRAPTORDEF_H
#define GRAPTOR_GRAPH_GRAPTORDEF_H

#ifndef GRAPTOR
#define GRAPTOR 1
#endif // GRAPTOR

#ifndef GRAPTOR_CACHED
#define GRAPTOR_CACHED 1
#endif // GRAPTOR_CACHED

#ifndef GRAPTOR_CSC
#define GRAPTOR_CSC 1
#endif // GRAPTOR_CSC

#define GRAPTOR_WITH_REDUCE 1
#define GRAPTOR_WITH_SELL 2
#define GRAPTOR_MIXED 3

#ifndef GRAPTOR_THRESHOLD_MULTIPLIER
#define GRAPTOR_THRESHOLD_MULTIPLIER 1
#endif // GRAPTOR_THRESHOLD_MULTIPLIER

#ifndef GRAPTOR_DEGREE_BITS
#ifdef __AVX512F__
#define GRAPTOR_DEGREE_BITS 16
#else
#define GRAPTOR_DEGREE_BITS 8
#endif
#endif // GRAPTOR_DEGREE_BITS

#ifndef GRAPTOR_DEGREE_MULTIPLIER
#define GRAPTOR_DEGREE_MULTIPLIER 1
#endif // GRAPTOR_DEGREE_MULTIPLIER

#ifndef GRAPTOR_CSR_INDIR
#define GRAPTOR_CSR_INDIR 1
#endif // GRAPTOR_CSR_INDIR

#ifndef GRAPTOR_SKIP_BITS
#if GRAPTOR_CSR_INDIR
#define GRAPTOR_SKIP_BITS 0
#else
#define GRAPTOR_SKIP_BITS 0
#endif
#endif // GRAPTOR_SKIP_BITS

// Problem: DEG12 does not work easily as common targets will cause allocation
//          of additional vectors, causing the degree to decrease
//          non-monotonically
#ifndef GRAPTOR_DEG12
#define GRAPTOR_DEG12 0
#endif // GRAPTOR_DEG12

enum graptor_mode_t {
    gm_csc_vreduce_not_cached = 0,
    gm_csc_vreduce_cached = 1,
    gm_csc_datapar_not_cached = 2,
    gm_csc_datapar_cached = 3,
    gm_csr_vpush_not_cached = 4,
    gm_csr_vpush_cached = 5,
    gm_csr_datapar_not_cached = 6,
    gm_csr_datapar_cached = 7
};

#define GRAPTOR_MODE_MACRO(c,m,csc) (graptor_mode_t)	\
    ( ((c) == 1 ? 1 : 0)   \
      | ((m) == 1 ? 0 : 2)			\
      | ((csc) == 1 ? 0 : 4) )

#define GRAPTOR_MODE GRAPTOR_MODE_MACRO(GRAPTOR_CACHED,GRAPTOR,GRAPTOR_CSC)

template<graptor_mode_t Mode>
class GraphCSRSIMDDegreeMixed;

template<graptor_mode_t Mode>
class GraphCSxSIMDDegreeMixed;

template<graptor_mode_t Mode, typename Enable = void>
struct GraptorConfig {
    using partition_type = GraphCSxSIMDDegreeMixed<Mode>;
    using remap_type = VEBOReorder;
    static constexpr bool is_csc = true;
    static constexpr bool is_datapar = (((unsigned)Mode) & 2) != 0;
    static constexpr bool is_cached = (((unsigned)Mode) & 1) != 0;
};

template<graptor_mode_t Mode>
struct GraptorConfig<Mode,
		     std::enable_if_t<(((unsigned)Mode) & 4) != 0>> {
    using partition_type = GraphCSRSIMDDegreeMixed<Mode>;
#if VEBO_DISABLE
    using remap_type = VEBOReorderIdempotent<VID,EID>;
#else
    // using remap_type = VEBOReorderSIMD<VID,EID>;
    using remap_type = VEBOReorder;
#endif
    static constexpr bool is_csc = false;
    static constexpr bool is_datapar = (((unsigned)Mode) & 2) != 0;
    static constexpr bool is_cached = (((unsigned)Mode) & 1) != 0;
};

#endif // GRAPTOR_GRAPH_GRAPTORDEF_H

