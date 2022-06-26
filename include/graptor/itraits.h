// -*- C++ -*-

/***********************************************************************
 * Integer traits
 ***********************************************************************/

#ifndef GRAPTOR_ITRAITS_H
#define GRAPTOR_ITRAITS_H

#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <ostream>

// An integer with W bytes
template<unsigned short W, typename = void>
class longint;

template<size_t bytes, typename = void>
struct int_type_of_size;

template<>
struct int_type_of_size<sizeof(uint64_t)> {
    using type = uint64_t;
};

template<>
struct int_type_of_size<sizeof(uint32_t)> {
    using type = uint32_t;
};

template<>
struct int_type_of_size<sizeof(uint16_t)> {
    using type = uint16_t;
};

template<>
struct int_type_of_size<sizeof(uint8_t)> {
    using type = uint8_t;
};

template<size_t bytes>
struct int_type_of_size<bytes, std::enable_if_t<(bytes>8)>> {
    using type = longint<bytes>;
};

template<>
struct int_type_of_size<3> {
    using type = typename int_type_of_size<4>::type;
};

template<size_t bytes>
using int_type_of_size_t = typename int_type_of_size<bytes>::type;

template<size_t bytes>
struct index_type_of_size_select {
    using type = uint64_t; // default size, don't go higher than this
};

template<>
struct index_type_of_size_select<sizeof(uint32_t)> {
    using type = uint32_t;
};

template<>
struct index_type_of_size_select<sizeof(uint16_t)> {
    using type = uint16_t;
};

template<>
struct index_type_of_size_select<sizeof(uint8_t)> {
    using type = uint8_t;
};

template<size_t bytes>
using index_type_of_size = typename index_type_of_size_select<bytes>::type;

// TODO: these need to be reviewed, made configurable
// typedef size_t VID;
typedef unsigned int VID;
// typedef int32_t VID;
// typedef int64_t EID;
typedef size_t EID;
// typedef unsigned int EID;

/***********************************************************************
 * Logical<bytes> template type to aid vectorisation.
 ***********************************************************************/
template<unsigned short Bytes, typename = void>
struct logical;

template<typename T>
struct is_integral_or_logical : std::is_integral<T> { };

template<unsigned short Bytes>
struct is_integral_or_logical<logical<Bytes>> : std::true_type { };

template<typename T>
constexpr bool is_integral_or_logical_v = is_integral_or_logical<T>::value;

template<typename T>
struct is_logical : std::false_type { };

template<unsigned short Bytes>
struct is_logical<logical<Bytes>> : std::true_type { };

template<typename T>
constexpr bool is_logical_v = is_logical<T>::value;

template<unsigned short Bytes_>
struct logical<Bytes_, std::enable_if_t<(Bytes_<=8)>> {
    static constexpr unsigned short Bytes = Bytes_;
    using type = typename int_type_of_size<Bytes>::type;
    using self_type = logical<Bytes>;

    static constexpr self_type false_val() { return self_type( false ); }
    static constexpr self_type true_val () { return self_type( true ); }

    /**
     * Create logical mask from boolean-like value
     *
     * b=0 -> "false"/logical<>(0); b!=0 -> "true"/logical<>(~0)
     *
     * \param b Boolean-like value
     * \return logical mask corresponding to #b
     */
    template<typename B>
    static constexpr self_type get_val( B b ) {
	return b ? true_val() : false_val();
    }

    // This construct occurs that this constructor is used only for
    // bool arguments. This avoids issues with ambiguous overloading
    // where either the bool or int constructor can be used.
    template<class Arg>
    explicit logical(
	Arg val_,
	std::enable_if_t<std::is_same<std::decay_t<Arg>, bool>::value> * =
	nullptr ) : val( val_ ? ~(type)0 : (type)0 ) { }
    explicit logical( type val_ = 0 ) : val( val_ ) {
	static_assert( sizeof(*this) == sizeof(type), "logical<> size check" );
    }
    template<unsigned short C>
    explicit logical( const logical<C> & arg )
	: val( static_cast<std::make_signed_t<type>>(
		   static_cast<std::make_signed_t<typename logical<C>::type>>( arg.get() ) ) ) { }
    // logical( const self_type & arg ) : val( arg.val ) { }

    type data() const { return val; }
    type get() const { return val; }

    bool is_true() const {
	// Checks top bit
	using stype = std::make_signed_t<type>;
	return static_cast<stype>( val ) < stype(0);
    }

    operator type () const { return val; }

    // Refined equality tests: only the top bit matters
    bool operator ! () const { return !is_true(); }
    bool operator == ( self_type l ) const { return !(*this ^ l).is_true(); }
    bool operator != ( self_type l ) const { return (*this ^ l).is_true(); }

    // These are all bit-wise!
    self_type operator |  (self_type l) const { return self_type(val | l.val); }
    self_type operator || (self_type l) const { return self_type(val | l.val); }
    self_type operator &  (self_type l) const { return self_type(val & l.val); }
    self_type operator && (self_type l) const { return self_type(val & l.val); }
    self_type operator ^  (self_type l) const { return self_type(val ^ l.val); }
    self_type operator ~  () const { return self_type( ~val ); }

private:
    type val;
};

// Might almost fold this into the definition of logical for native types
// above, however, there are a few small differences
template<unsigned short Bytes_>
struct logical<Bytes_, std::enable_if_t<(Bytes_>8)>> {
    static constexpr unsigned short Bytes = Bytes_;
    using type = longint<Bytes>;
    using self_type = logical<Bytes>;

    static constexpr self_type false_val() {
	return self_type( type::setzero() );
    }
    static constexpr self_type true_val () {
	return self_type( type::setallone() );
    }
    template<typename B>
    static constexpr self_type get_val( B b ) {
	return b ? true_val() : false_val(); }

    template<class Arg>
    explicit logical(
	Arg val_,
	std::enable_if_t<std::is_same<std::decay_t<Arg>, bool>::value> * =
	nullptr ) : val( val_ ? ~(type)0 : (type)0 ) { }
    explicit logical( type val_ = 0 ) : val( val_ ) {
	static_assert( sizeof(*this) == sizeof(type), "logical<> size check" );
    }
    template<unsigned short C>
    explicit logical( const logical<C> & arg )
	: val( static_cast<std::make_signed_t<type>>(
		   static_cast<std::make_signed_t<typename logical<C>::type>>( arg.get() ) ) ) { }
    // explicit logical( typename type::type val_ ) : val( type(val_) ) { }

    type data() const { return val; }
    auto get() const { return val.get(); } // urgh

    bool is_true() const {
	// Checks top bit
	using stype = std::make_signed_t<type>;
	return static_cast<stype>( val ) < stype(0);
    }

    operator type () const { return val; }

    // Refined equality tests: only the top bit matters
    bool operator ! () const { return !is_true(); }
    bool operator == ( self_type l ) const { return !(*this ^ l).is_true(); }
    bool operator != ( self_type l ) const { return (*this ^ l).is_true(); }

    // These are all bit-wise!
    self_type operator |  (self_type l) const { return self_type(val | l.val); }
    self_type operator || (self_type l) const { return self_type(val | l.val); }
    self_type operator &  (self_type l) const { return self_type(val & l.val); }
    self_type operator && (self_type l) const { return self_type(val & l.val); }
    self_type operator ^  (self_type l) const { return self_type(val ^ l.val); }
    self_type operator ~  () const { return self_type( ~val ); }

private:
    type val;
};

template<unsigned short Bytes>
std::ostream & operator << ( std::ostream & os, logical<Bytes> v ) {
    return os << "L<" << v.data() << '>';
}

inline std::ostream & operator << ( std::ostream & os, logical<1> v ) {
    return os << "L<" << ( v.data() ? "true" : "false" ) << '>';
}

template<typename T>
struct add_logical {
    using type = logical<sizeof(T)>;
};

template<>
struct add_logical<bool> {
    using type = bool;
};

template<typename T>
inline T as_type( bool );

template<>
inline bool as_type<bool>( bool val ) {
    return val;
}

template<>
inline logical<1> as_type<logical<1>>( bool val ) {
    return val ? logical<1>::true_val() : logical<1>::false_val();
}

template<>
inline logical<2> as_type<logical<2>>( bool val ) {
    return val ? logical<2>::true_val() : logical<2>::false_val();
}

template<>
inline logical<4> as_type<logical<4>>( bool val ) {
    return val ? logical<4>::true_val() : logical<4>::false_val();
}

template<>
inline logical<8> as_type<logical<8>>( bool val ) {
    return val ? logical<8>::true_val() : logical<8>::false_val();
}

/***********************************************************************
 * Utility to calculate array lenght in the context of element types
 * that do not take a full number of bytes.
 ***********************************************************************/
template<typename T>
struct mm_get_total_bytes {
    static constexpr size_t get( size_t num_elems ) {
	return num_elems * sizeof( T );
    }
};

#endif // GRAPTOR_ITRAITS_H
