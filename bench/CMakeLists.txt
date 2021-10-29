# pick up all benchmarks
list(APPEND benchmarks CCv BFSv PRv APRv BFv MISv FMv BFSLVLv)

# determine what graph types to compile
#list(APPEND graphtypes VEBOPartCCSR GGVEBO)
list(APPEND graphtypes GGVEBO)

# Add include directory
string(APPEND graptor_cflags " -I${CMAKE_SOURCE_DIR}/bench")

# set executable files
foreach(bb ${benchmarks})
    foreach(g ${graphtypes})
	add_executable(${bb}_${g} ${CMAKE_SOURCE_DIR}/bench/${bb}.C)
	set_target_properties(${bb}_${g} PROPERTIES COMPILE_FLAGS "-DGRAPHTYPE=${g} ${graptor_cflags}")

	# link options
	target_link_libraries(${bb}_${g} ${EXTRA_LIBS} graptorlib)

	list(APPEND executables ${bb}_${g})
    endforeach()
endforeach()
