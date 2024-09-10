// -*- C++ -*-
#ifndef GRAPHGRIND_MM_ALLOCATION_H
#define GRAPHGRIND_MM_ALLOCATION_H

/***********************************************************************
 * Represents a memory allocation
 ***********************************************************************/

namespace mm {

class allocation {
public:
    allocation() { }
    allocation( int )
	: m_ptr( 0 ), m_size( 0 ), m_mapped( 0 ), m_aligned( 0 ) { }
    allocation( intptr_t ptr, size_t size,
		bool mapped = false, bool aligned = false )
	: m_ptr( ptr ), m_size( size ),
	  m_mapped( mapped ), m_aligned( aligned ) { }

    intptr_t ptr() const { return m_ptr; }
    size_t size() const { return m_size; }
    bool is_mapped() const { return m_mapped; }
    bool is_aligned() const { return m_aligned; }
    void clear() {
	m_ptr = 0;
	m_size = 0;
	m_mapped = false;
	m_aligned = false;
    }
private:
    intptr_t m_ptr; //!< address of the allocation
    size_t m_size;  //!< size of the allocation
    bool m_mapped;  //!< is this mmap/munmap (true), or malloc/free (false)
    bool m_aligned; //!< for mmap'ed data, if ptr was aligned to small page
};
    
} // namespace mm

#endif // GRAPHGRIND_MM_ALLOCATION_H

