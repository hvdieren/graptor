// -*- C++ -*-
// This code is part of the project "Ligra: A Lightweight Graph Processing
// Framework for Shared Memory", presented at Principles and Practice of
// Parallel Programming, 2013.
// Copyright (c) 2013 Julian Shun and Guy Blelloch
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

#ifndef GRAPTOR_LEGACY_UTILS_H
#define GRAPTOR_LEGACY_UTILS_H

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sched.h>
#include <errno.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <utility>
#include <algorithm>
#include "parallel.h"
//#include "mm.h"
#include <sys/mman.h>
using namespace std;
#if 0
#ifndef __APPLE__
// Needed to make frequent large allocations efficient with standard
// malloc implementation.  Otherwise they are allocated directly from
// vm.
#include <malloc.h>
static int __ii =  mallopt(M_MMAP_MAX,0);
static int __jj =  mallopt(M_TRIM_THRESHOLD,-1);
#endif
#endif
//#define newA(__E,__n) (__E*) malloc((__n)*sizeof(__E))
//#define newnumaA(__E,__n) (__E*) numa_local_alloc((__n)*sizeof(__E))
//#define numaAlloc(__E,__n,__node) (__E*) numa_alloc_onnode((__n)*sizeof(__E),__node)
//#define numaInterleave(__E,__n) (__E*) numa_alloc_interleaved((__n)*sizeof(__E))

#include <x86intrin.h>
#include <immintrin.h>

#if __AVX512F__
// Conventions changed
using _m512i = __m512i;
#endif

#define VECTORIZED 1 // for now

typedef pair<intT, intT> intTpair;

template <class E>
struct identityF
{
    E operator() (const E& x)
    {
        return x;
    }
};

template <class E>
struct addF
{
    E operator() (const E& a, const E& b) const
    {
        return a+b;
    }
};

template <class E>
struct minF
{
    E operator() (const E& a, const E& b) const
    {
        return (a < b) ? a : b;
    }
};

template <class T>

struct _seq
{
    T* A;
    //mmap_ptr<T> A;
    long n;
    _seq()
    {
        A = NULL;
     //   A = mmap_ptr<T> ();
        n=0;
    }
    _seq(T* _A, long _n) : A(_A), n(_n) {}
    void del()
    {
        delete [] A;
        //A.del();
    }
};

namespace sequence
{
template <class intT>
struct boolGetA
{
    bool* A;
    boolGetA(bool* AA) : A(AA) {}
    intT operator() (intT i)
    {
        return (intT) A[i];
    }
};

template <class ET, class intT>
struct getA
{
    ET* A;
    getA(ET* AA) : A(AA) {}
    ET operator() (intT i)
    {
        return A[i];
    }
};

template<typename E>
struct argmaxF {
    E operator() (const E& a, const E& b) const {
	return ( a.first > b.first ) ? a : b;
    }
};

//Define F function for dense (outdegreecount)
template<class intT>
class FDense
{
    public:
    intTpair operator()(intTpair l, intTpair r)
    {
        return make_pair(l.first+r.first,l.second+r.second);
    }
};


#define nblocks(_n,_bsize) (1 + ((_n)-1)/(_bsize))

#define blocked_for(_i, _s, _e, _bsize, _body)  {	\
    intT _ss = _s;					\
    intT _ee = _e;					\
    intT _n = _ee-_ss;					\
    intT _l = nblocks(_n,_bsize);			\
    parallel_for (intT _i = 0; _i < _l; _i++) {		\
      intT _s = _ss + _i * (_bsize);			\
      intT _e = min(_s + (_bsize), _ee);		\
      _body						\
	}						\
  }

template <class OT, class intT, class F, class G>
OT reduceSerial(intT s, intT e, F f, G g)
{
    OT r = g(s);
    for (intT j=s+1; j < e; j++) r = f(r,g(j));
    return r;
}
template <class OT, class intT, class F, class G>
OT reduce(intT s, intT e, F f, G g)
{
    intT l = nblocks(e-s, _SCAN_BSIZE);
    if (l <= 1) return reduceSerial<OT>(s, e, f , g);
    OT *Sums = new OT [l];
    blocked_for (i, s, e, _SCAN_BSIZE,
                 Sums[i] = reduceSerial<OT>(s, e, f, g););
    OT r = reduce<OT>((intT) 0, l, f, getA<OT,intT>(Sums));
    delete [] Sums;
    return r;
}
//Define for Dense (d_m and outdegree-count for dense operator)

template <class intT, class G>
intTpair reduce(intT s, intT n, G g)
{
    return reduce<intTpair>((intT)s,n,FDense<intT>(),g);
}

template <class intT, class G>
intTpair reduceSerial(intT s, intT n, G g)
{
    return reduceSerial<intTpair>((intT)s,n,FDense<intT>(),g);
}

template <class OT, class intT, class F>
OT reduce(OT* A, intT n, F f)
{
    return reduce<OT>((intT)0,n,f,getA<OT,intT>(A));
}

template <class OT, class intT>
OT plusReduce(OT* A, intT n)
{
    return reduce<OT>((intT)0,n,addF<OT>(),getA<OT,intT>(A));
}
template <class intT>
intT sum(bool *In, intT s, intT n)
{
    return reduce<intT>((intT) s, n, addF<intT>(), boolGetA<intT>(In));
}
template <class intT>
intT sum(bool *In, intT n)
{
    return reduce<intT>((intT) 0, n, addF<intT>(), boolGetA<intT>(In));
}

template <class ET, class intT, class F, class G>
ET scanSerial(ET* Out, intT s, intT e, F f, G g, ET zero, bool inclusive, bool back)
{
    ET r = zero;
    if (inclusive)
    {
        if (back) for (intT i = e-1; i >= s; i--) Out[i] = r = f(r,g(i));
        else for (intT i = s; i < e; i++) Out[i] = r = f(r,g(i));
    }
    else
    {
        if (back)
            for (intT i = e-1; i >= s; i--)
            {
                ET t = g(i);
                Out[i] = r;
                r = f(r,t);
            }
        else
            for (intT i = s; i < e; i++)
            {
                ET t = g(i);
                Out[i] = r;
                r = f(r,t);
            }
    }
    return r;
}

template <class ET, class intT, class F>
ET scanSerial(ET *In, ET* Out, intT n, F f, ET zero)
{
    return scanSerial(Out, (intT) 0, n, f, getA<ET,intT>(In), zero, false, false);
}

// back indicates it runs in reverse direction
template <class ET, class intT, class F, class G>
ET scan(ET* Out, intT s, intT e, F f, G g,  ET zero, bool inclusive, bool back)
{
    intT n = e-s;
    intT l = nblocks(n,_SCAN_BSIZE);
    if (l <= 2) return scanSerial(Out, s, e, f, g, zero, inclusive, back);
    ET *Sums = new ET [nblocks(n,_SCAN_BSIZE)];
    blocked_for (i, s, e, _SCAN_BSIZE,
                 Sums[i] = reduceSerial<ET>(s, e, f, g););
    ET total = scan(Sums, (intT) 0, l, f, getA<ET,intT>(Sums), zero, false, back);
    blocked_for (i, s, e, _SCAN_BSIZE,
                 scanSerial(Out, s, e, f, g, Sums[i], inclusive, back););
    delete [] Sums;
    return total;
}

template <class ET, class intT, class F>
ET scan(ET *In, ET* Out, intT n, F f, ET zero)
{
    return scan(Out, (intT) 0, n, f, getA<ET,intT>(In), zero, false, false);
}

template <class ET, class intT, class F>
ET scanI(ET *In, ET* Out, intT n, F f, ET zero)
{
    return scan(Out, (intT) 0, n, f, getA<ET,intT>(In), zero, true, false);
}

template <class ET, class intT, class F>
ET scanBack(ET *In, ET* Out, intT n, F f, ET zero)
{
    return scan(Out, (intT) 0, n, f, getA<ET,intT>(In), zero, false, true);
}

template <class ET, class intT, class F>
ET scanIBack(ET *In, ET* Out, intT n, F f, ET zero)
{
    return scan(Out, (intT) 0, n, f, getA<ET,intT>(In), zero, true, true);
}

template <class ET, class intT>
ET plusScan(ET *In, ET* Out, intT n)
{
    return scan(Out, (intT) 0, n, addF<ET>(), getA<ET,intT>(In),
                (ET) 0, false, false);
}

#define _F_BSIZE (2*_SCAN_BSIZE)

#if 0 && VECTORIZED
// sums a sequence of n boolean flags
#ifdef __cilk
#if 0
// leverages CILK array notation
intT as_intT( bool b ) { return (intT)b; }

template <class intT>
intT  __attribute__((noinline)) sumFlagsSerial(bool *Fl, intT n)
{
    return __sec_reduce_add( as_intT(Fl[0:n]) );
}
#endif
#else // ifdef __cilk
#endif // ifdef __cilk
#else // ifdef VECTORIZED
// sums a sequence of n boolean flags
// an optimized version that sums blocks of 4 booleans by treating
// them as an integer
// Only optimized when n is a multiple of 512 and Fl is 4byte aligned
constexpr size_t log2_roundup( size_t n ) {
    return (n<2) ? 0 : 1+log2_roundup(n/2);
}

template<typename accTy>
inline int sumFlagsSerialReduce( accTy rr ) {
    int r = 0;
    for( size_t j=0; j < sizeof(accTy); ++j )
	r += (rr>>(j*8)) & 255;
    return (int)r;
}

template<>
inline int sumFlagsSerialReduce<int>( int rr ) {
    static_assert( sizeof(int) == 4, "making assumptions on int size" );
    const int mask1 = 0x00ff00ff;
    const int mask2 = 0x0000ffff;
    int r = ((rr >> 8) & mask1) + (rr & mask2);
    r = ((r >> 16) & mask2) + (r & mask2);
    return r;
}

template<>
inline int sumFlagsSerialReduce<long>( long rr ) {
    static_assert( sizeof(long) == 8, "making assumptions on long size" );
    const long mask1 = 0x00ff00ff00ff00ffL;
    const long mask2 = 0x0000ffff0000ffffL;
    const long mask3 = 0x00000000ffffffffL;
    long r = ((rr >> 8) & mask1) + (rr & mask1);
    r = ((r >> 16) & mask2) + (r & mask2);
    r = ((r >> 32) & mask3) + (r & mask3);
    return (int)r;
}

#if __AVX512F__
template<>
inline int sumFlagsSerialReduce<_m512i>( _m512i r0 ) {
    static_assert( sizeof(_m512i)*8 == 512,
		   "making assumptions on _m512i size" );

    const _m512i vzero = _mm512_set1_epi64( 0 );

    // Step 1: shift bytes and add (64 bytes to add)
    const _m512i mask1l = _mm512_set1_epi16( short(0x00ff) );
    const _m512i r1l = _mm512_and_epi64( mask1l, r0 );
    const _m512i r1a = _mm512_srli_epi32( r0, 8 );
    const _m512i r1ra = _mm512_and_epi64( mask1l, r0 );
    const _m512i r1 = _mm512_add_epi64( r1l, r1ra );

    // Step 2: shift short and add (32 shorts to add)
    const _m512i mask2l = _mm512_set1_epi32( int(0x0000ffff) );
    const _m512i r2l = _mm512_and_epi64( mask2l, r1 );
    const _m512i r2a = _mm512_srli_epi32( r1, 16 );
    const _m512i r2ra = _mm512_and_epi64( mask2l, r2a );
    const _m512i r2 = _mm512_add_epi64( r2l, r2ra );
    
    // Step 3: add up 16 int
    return _mm512_reduce_add_epi32( r2 );
}
#endif // __AVX512F__

template <typename accTy>
int sumFlagsSerialHelper(bool *Fl, size_t n)
{
    int r = 0;
    const size_t VL = sizeof(accTy);
    const size_t l2VL = log2_roundup( VL );
    size_t k = 0;

    // Accelerated part
    if( n >= 128
// 	&& (n & (VL-1)) == 0
	&& ((intptr_t) Fl & (intptr_t(VL)-1)) == 0 ) {
        accTy* IFl = (accTy*) Fl;
        for( k = 0; k < (n >> (l2VL+7)); k++ ) {
            accTy rr = 0;
            for( size_t j=0; j < 128; j++ ) rr += IFl[j];
	    int r2 = sumFlagsSerialReduce( rr );
/* Correctness checking
*/
	    accTy r1 = 0;
	    for( size_t j=0; j < sizeof(accTy); ++j )
		r1 += (rr>>(j*8)) & 255;
	    assert( (int)r1 == r2 );
	    r += r2;
            IFl += 128;
        }
	k <<= l2VL+7;
    }

    // Epilogue, if needed
    for( size_t j=k; j < n; j++) r += (int)Fl[j];

/* Correctness checking
    accTy r2 = r;
    r = 0;
    for (size_t j=0; j < n; j++) r += Fl[j];
    assert( r2 == 0 || r2 == r );
 */
    return r;
}

template <class intT>
intT sumFlagsSerial(bool *Fl, intT n) {
    return sumFlagsSerialHelper<long>( Fl, n );
}

#if 0
template <class intT>
intT sumFlagsSerial(bool *Fl, intT n)
{
    intT r = 0;
    if (n >= 128 && (n & 511) == 0 && ((long) Fl & 3) == 0)
    {
	cerr << "special sumFlagsSerial case\n";
        int* IFl = (int*) Fl;
        for (int k = 0; k < (n >> 9); k++)
        {
            int rr = 0;
            for (int j=0; j < 128; j++) rr += IFl[j];
            r += (rr&255) + ((rr>>8)&255) + ((rr>>16)&255) + ((rr>>24)&255);
            IFl += 128;
        }
    }
    else {
	cerr << "base sumFlagsSerial case " << n << "\n";
	for (intT j=0; j < n; j++) r += Fl[j];
    }
    return r;
}
#endif
#endif // ifdef VECTORIZED

template <class BT, class intT>
intT sumFlagsSerial(BT *Fl, intT n) {
    intT r = 0;
    for( intT i=0; i < n; ++i )
	if( Fl[i] )
	    r++;
    return r;
}


template <class BT, class intT>
intT sumFlagsSerial(BT Fl, intT n) {
    intT r = 0;
    for( intT i=0; i < n; ++i )
	if( Fl[i] )
	    r++;
    return r;
}

template <class ET, class intT, class F>
_seq<ET> packSerial(ET* Out, bool* Fl, intT s, intT e, F f)
{
    if (Out == NULL)
    {
        intT m = sumFlagsSerial(Fl+s, e-s);
        Out = new ET [m];
    }
    intT k = 0;
    for (intT i=s; i < e; i++) if (Fl[i]) Out[k++] = f(i);
    return _seq<ET>(Out,k);
}

template <class ET, class BT, class intT, class F>
_seq<ET> packSerial(ET* Out, BT* Fl, intT s, intT e, F f)
{
    if (Out == NULL)
    {
        intT m = sumFlagsSerial(Fl+s, e-s);
        Out = new ET [m];
    }
    intT k = 0;
    for (intT i=s; i < e; i++) if (Fl[i]) Out[k++] = f(i);
    return _seq<ET>(Out,k);
}

template <class ET, class BT, class intT, class F>
_seq<ET> packSerial(ET* Out, BT Fl, intT s, intT e, F f)
{
    if (Out == NULL)
    {
        intT m = sumFlagsSerial(Fl+s, e-s);
        Out = new ET [m];
    }
    intT k = 0;
    for (intT i=s; i < e; i++) if (Fl[i]) Out[k++] = f(i);
    return _seq<ET>(Out,k);
}

// getA<> version of packSerial
template <class ET, class intT>
_seq<ET> packSerial(ET* Out, bool* Fl, intT s, intT e, ET *array)
{
    // cerr << "packSerial getA\n";
    if (Out == NULL)
    {
        intT m = sumFlagsSerial(Fl+s, e-s);
        Out = new ET [m];
    }
    intT k = 0;
#if __AVX512F__ && 0 // TODO
    intT i = s;
    constexpr size_t words_per_vec = sizeof(__m512i) / 8 / sizeof(intT);
    static_assert( sizeof(intT) == 8, "code assumes intT is epi64" );

    // prologue
    for( ; i < e && (intptr_t(&array[i])&(sizeof(__m512i)-1)) != 0; i++ )
	if (Fl[i]) Out[k++] = array[i]; // f(i);

    assert( 0 && "double check code" );
#if 0
    __m128i vzero_mm = 0; // ...; // TODO!!!
    size_t nhave = 0;
    __m512i collect = _mm512_setzero_epi64();
    for( ; i+words_per_vec < e; i += words_per_vec ) {
	// if( Fl[i] ) Out[k++] = array[i];
	__m128i vflags = _mm_loadu_si64( (void const *)&Fl[i] );
	__mmask16 vflset16 = _mm_cmp_epi8_mask( vflags, vzero_mm, 4 ); // NEQ
	__mmask8 vflset = vflset16 & 255;
	assert( (vflset & 255) == vflset ); // higher bits zero by construction
	__m512i data = _mm512i_load_epi64( (void const *)&array[i] );
	__m512i vcmp = _mm512_maskz_compress_epi64( vflset, data );
	size_t nadd = _mm_popcnt_u64( vflset );
	__mmask8 hi = ((__mask8(1)<<nadd)-1) << nhave;
	__m512i vcmpp = _mm512_maskz_permutexvar_epi64( hi, vidx, vcmp );
	
    }
#endif

    // epilogue
    for(; i < e; i++) if (Fl[i]) Out[k++] = array[i]; // f(i);
#else
    for (intT i=s; i < e; i++) if (Fl[i]) Out[k++] = array[i]; // f(i);
#endif // __AVX512F__
    return _seq<ET>(Out,k);
}

template <class ET, class intT>
_seq<ET> packSerialIdentity(ET* Out, bool* Fl, intT s, intT e)
{
    // cerr << "packSerial identity\n";
    if (Out == NULL)
    {
        intT m = sumFlagsSerial(Fl+s, e-s);
        Out = new ET [m];
    }
    intT k = 0;
    for (intT i=s; i < e; i++) if (Fl[i]) Out[k++] = i; // f(i);
    return _seq<ET>(Out,k);
}

template <class ET, class intT>
_seq<ET> packSerial(ET* Out, bool* Fl, intT s, intT e, getA<ET, intT> f)
{
    return packSerial( Out, Fl, s, e, f.A );
}

template <class ET, class intT>
_seq<ET> packSerial(ET* Out, bool* Fl, intT s, intT e, identityF<intT> f)
{
    return packSerialIdentity( Out, Fl, s, e );
}

template <class ET, class intT, class F>
_seq<ET> packSerial(ET* Out, bool* Fl, intT s, intT e, intT startPos, F f)
{
    if (Out == NULL)
    {
        intT m = sumFlagsSerial(Fl+s, e-s);
        Out = new ET [m];
        // parallel_for (intT i=0; i < m; i++) Out[i] = Out[i]+startPos;
    }
    intT k = 0;
    for (intT i=s; i < e; i++) if (Fl[i]) Out[k++] = f(i) + startPos;
    // parallel_for(intT i=0; i<k; i++) Out[i]=Out[i]+startPos;
    return _seq<ET>(Out,k);
}

// This is the baseline, generic pack.
template <class ET, class intT, class F>
_seq<ET> pack(ET* Out, bool* Fl, intT s, intT e, F f)
{
    intT l = nblocks(e-s, _F_BSIZE);
    if (l <= 1) return packSerial(Out, Fl, s, e, f);
    intT *Sums = new intT [l];
    blocked_for (i, s, e, _F_BSIZE, Sums[i] = sumFlagsSerial(Fl+s, e-s););
    intT m = plusScan(Sums, Sums, l);
    if (Out == NULL) Out = new ET [m];
    blocked_for(i, s, e, _F_BSIZE, packSerial(Out+Sums[i], Fl, s, e, f););
    delete [] Sums;
    return _seq<ET>(Out,m);
}

// This is the generic pack with configurable input type.
template <class ET, class BT, class intT, class F>
_seq<ET> pack(ET* Out, BT* Fl, intT s, intT e, F f)
{
    intT l = nblocks(e-s, _F_BSIZE);
    if (l <= 1) return packSerial(Out, Fl, s, e, f);
    intT *Sums = new intT [l];
    blocked_for (i, s, e, _F_BSIZE, Sums[i] = sumFlagsSerial(Fl+s, e-s););
    intT m = plusScan(Sums, Sums, l);
    if (Out == NULL) Out = new ET [m];
    blocked_for(i, s, e, _F_BSIZE, packSerial(Out+Sums[i], Fl, s, e, f););
    delete [] Sums;
    return _seq<ET>(Out,m);
}

// This is the generic pack with configurable input type.
template <class ET, class BT, class intT, class F>
_seq<ET> pack(ET* Out, BT Fl, intT s, intT e, F f)
{
    intT l = nblocks(e-s, _F_BSIZE);
    if (l <= 1) return packSerial(Out, Fl, s, e, f);
    intT *Sums = new intT [l];
    blocked_for (i, s, e, _F_BSIZE, Sums[i] = sumFlagsSerial(Fl+s, e-s););
    intT m = plusScan(Sums, Sums, l);
    if (Out == NULL) Out = new ET [m];
    blocked_for(i, s, e, _F_BSIZE, packSerial(Out+Sums[i], Fl, s, e, f););
    delete [] Sums;
    return _seq<ET>(Out,m);
}

#if 0
// This version of pack is specialised to the getA<> function and is called
// by filter.
template <class ET, class intT>
_seq<ET> pack(ET* Out, bool* Fl, intT s, intT e, getA<ET, intT> f)
{
    cerr << "modified overridden pack getA\n";
    intT l = nblocks(e-s, _F_BSIZE);
    if (l <= 1) return packSerial(Out, Fl, s, e, f);
    intT *Sums = new intT [l];
    blocked_for (i, s, e, _F_BSIZE, Sums[i] = sumFlagsSerial(Fl+s, e-s););
    intT m = plusScan(Sums, Sums, l);
    if (Out == NULL) Out = new ET [m];
    blocked_for(i, s, e, _F_BSIZE, packSerial(Out+Sums[i], Fl, s, e, f););
    delete [] Sums;
    return _seq<ET>(Out,m);
}

// This version of pack is specialised for the packIndex() function.
_seq<intT> pack(intT* Out, bool* Fl, intT s, intT e, identityF<intT> f)
{
    cerr << "modified overridden pack identity\n";
    intT l = nblocks(e-s, _F_BSIZE);
    if (l <= 1) return packSerial(Out, Fl, s, e, f);
    intT *Sums = new intT [l];
    blocked_for (i, s, e, _F_BSIZE, Sums[i] = sumFlagsSerial(Fl+s, e-s););
    intT m = plusScan(Sums, Sums, l);
    if (Out == NULL) Out = new intT [m];
    blocked_for(i, s, e, _F_BSIZE, packSerial(Out+Sums[i], Fl, s, e, f););
    delete [] Sums;
    return _seq<intT>(Out,m);
}
#endif

template <class ET, class intT>
intT pack(ET* In, ET* Out, bool* Fl, intT n)
{
    return pack(Out, Fl, (intT) 0, n, getA<ET,intT>(In)).n;
}

template <class intT>
_seq<intT> packIndex(bool* Fl, intT n)
{
    return pack((intT *) NULL, Fl, (intT) 0, n, identityF<intT>());
}

template <class ET, class intT, class PRED>
intT filter(ET* In, ET* Out, intT n, PRED p)
{
    bool *Fl = new bool [n];
#if defined(CILK) || defined(CILKP)
    _Pragma( STRINGIFY(cilk grainsize = _F_BSIZE) )
#endif
	parallel_for (intT i=0; i < n; i++) Fl[i] = (bool) p(In[i]);
    intT  m = pack(In, Out, Fl, n);
    delete [] Fl;
    return m;
}
}

// The conditional should be removed by the compiler
// this should work with pointer types, or pairs of integers
template <class ET>
inline bool CAS(ET *ptr, ET oldv, ET newv)
{
    if (sizeof(ET) == 8)
    {
        long* o = (long*) &oldv;
        long* n = (long*) &newv;
        return __sync_bool_compare_and_swap((long*)ptr, *o, *n);
    }
    else if (sizeof(ET) == 4)
    {
        int* o = (int*) &oldv;
        int* n = (int*) &newv;
        return __sync_bool_compare_and_swap((int*)ptr, *o, *n);
    }
    else if (sizeof(ET) == 1)
    {
        char* o = (char*) &oldv;
        char* n = (char*) &newv;
        return __sync_bool_compare_and_swap((char*)ptr, *o, *n);
    }
    else
    {
        std::cout << "CAS bad length: " << sizeof(ET) << std::endl;
        abort();
    }
}

template <class ET>
inline bool writeMin(ET *a, ET b)
{
    ET c;
    bool r=0;
    do c = *a;
    while (c > b && !(r=CAS(a,c,b)));
    return r;
}

//atomically do bitwise-OR of *a with b and store in location a
template <class ET>
void writeOr(ET *a, ET b)
{
    volatile ET newV, oldV;
    do
    {
        oldV = *a;
        newV = oldV | b;
    }
    while ((oldV != newV) && !CAS(a, oldV, newV));
}

template <class ET>
inline void writeAdd(ET *a, ET b)
{
    volatile ET newV, oldV;
    do
    {
        oldV = *a;
        newV = oldV + b;
    }
    while (!CAS(a, oldV, newV));
}

inline unsigned int ligra_hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

inline unsigned long ligra_hash(unsigned long a)
{
    a = (a+0x7ed55d166bef7a1d) + (a<<12);
    a = (a^0xc761c23c510fa2dd) ^ (a>>9);
    a = (a+0x165667b183a9c0e1) + (a<<59);
    a = (a+0xd3a2646cab3487e3) ^ (a<<49);
    a = (a+0xfd7046c5ef9ab54c) + (a<<3);
    a = (a^0xb55a4f090dd4a67b) ^ (a>>32);
    return a;
}

#endif // GRAPTOR_LEGACY_UTILS_H
