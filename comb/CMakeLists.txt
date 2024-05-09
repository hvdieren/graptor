list(APPEND comb_exec MC MCE)

foreach(bb ${comb_exec})
    add_executable(${bb} ${bb}.C)
    target_link_libraries(${bb} ${EXTRA_LIBS} graptorlib)
    set_target_properties(${bb} PROPERTIES COMPILE_FLAGS "${graptor_cxxflags}")
endforeach()

foreach(bb intersect_ds)
    foreach(op 0 1 2)
	add_executable(${bb}_op${op} ${bb}.C)
	target_link_libraries(${bb}_op${op} ${EXTRA_LIBS} graptorlib)
	set_target_properties(${bb}_op${op} PROPERTIES COMPILE_FLAGS "-DOPERATION=${op} ${graptor_cxxflags}")
    endforeach()
endforeach()

set_target_properties(MCE PROPERTIES COMPILE_FLAGS ${graptor_cxxflags})
target_link_libraries(MCE ${EXTRA_LIBS} graptorlib)

add_subdirectory(scripts)
