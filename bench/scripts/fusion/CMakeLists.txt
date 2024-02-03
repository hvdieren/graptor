configure_file(build_avx512.sh.in build_avx512.sh @ONLY)
configure_file(fusion_avx512_arg.sh.in fusion_avx512_arg.sh @ONLY)
configure_file(fusion_avx512_lib.sh.in fusion_avx512_lib.sh @ONLY)
configure_file(gbbs_arg.sh.in gbbs_arg.sh @ONLY)
configure_file(gbbs_lib.sh.in gbbs_lib.sh @ONLY)

get_property(include_dirs TARGET BFSv_VEBOGraptorPullDataParCached PROPERTY INCLUDE_DIRECTORIES)
get_property(link_libs TARGET BFSv_VEBOGraptorPullDataParCached PROPERTY LINK_LIBRARIES)

add_custom_target(
	FUSION_BUILD_AVX512
	COMMAND env include_dirs="${include_dirs}"
		    link_libs="${link_libs}"
		    ${CMAKE_CURRENT_BINARY_DIR}/build_avx512.sh
	COMMENT "build executables for experiments"
	DEPENDS graptorlib)
