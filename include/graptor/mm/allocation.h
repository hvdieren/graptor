// -*- C++ -*-
#ifndef GRAPHGRIND_MM_ALLOCATION_H
#define GRAPHGRIND_MM_ALLOCATION_H

/***********************************************************************
 * Represents a memory allocation
 ***********************************************************************/

namespace mm {

class allocation {
public:
    allocation( intptr_t ptr, size_t size ) : m_ptr( ptr ), m_size( size ) { }

    intptr_t ptr() const { return m_ptr; }
    size_t size() const { return m_size; }
private:
    intptr_t m_ptr;
    size_t m_size;
};
    
} // namespace mm

#endif // GRAPHGRIND_MM_ALLOCATION_H

