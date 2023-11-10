configure_file(build_avx512.sh.in build_avx512.sh @ONLY)
configure_file(mce_avx512_arg.sh.in mce_avx512_arg.sh @ONLY)
configure_file(mce_avx512_lib.sh.in mce_avx512_lib.sh @ONLY)

get_property(include_dirs TARGET MCE PROPERTY INCLUDE_DIRECTORIES)
get_property(link_libs TARGET MCE PROPERTY LINK_LIBRARIES)

add_custom_target(
	MCE_BUILD_AVX512
	COMMAND env include_dirs="${include_dirs}"
		    link_libs="${link_libs}"
		    ${CMAKE_CURRENT_BINARY_DIR}/build_avx512.sh
	COMMENT "build executables for experiments"
	DEPENDS graptorlib)
