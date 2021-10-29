// -*- C++ -*-
#ifndef GRAPHGRIND_MM_TRACK_H
#define GRAPHGRIND_MM_TRACK_H

/***********************************************************************
 * Track memory usage and debugging support
 ***********************************************************************/

#include <mutex>
#include <map>

namespace mm {

struct allocation_info {
    const char * m_description;
    size_t m_size;
};

class tracker {
    using guard_type = std::unique_lock<std::recursive_mutex>;
    
public:
    static size_t num_allocations() {
	guard_type guard( m_mux );
	return m_current.size();
    }
    static size_t total_size() {
	return m_total_size;
    }

    static bool insert_if_new( void * mem, const char * reason, size_t size ) {
	guard_type guard( m_mux );
	if( m_current.find( mem ) != m_current.end() )
	    return false;
	else {
	    m_current.insert(
		std::make_pair( mem,
				allocation_info { reason, size } ) );
	    m_total_size += size;
	    return true;
	}
    }

    // Return a copy of the data as returning a reference/pointer is unsafe
    // in light of the lock being released.
    static const allocation_info lookup( void * mem ) {
	guard_type guard( m_mux );
	auto e = m_current.find( mem );
	if( e == m_current.end() )
	    return allocation_info();
	else
	    return e->second;
    }
    
    static bool remove( void * mem ) {
	guard_type guard( m_mux );
	auto e = m_current.find( mem );
	if( e == m_current.end() )
	    return false;
	else {
	    m_total_size -= e->second.m_size;
	    m_current.erase( e );
	    return true;
	}
    }

private:
    inline static std::recursive_mutex m_mux;
    inline static std::map<void *, allocation_info> m_current;
    inline static size_t m_total_size = 0;
};
    
} // namespace mm

#endif // GRAPHGRIND_MM_TRACK_H

