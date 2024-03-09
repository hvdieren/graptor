list(APPEND comb_exec MC MCE)

foreach(bb ${comb_exec})
    add_executable(${bb} ${bb}.C)
    target_link_libraries(${bb} ${EXTRA_LIBS} graptorlib)
    set_target_properties(${bb} PROPERTIES COMPILE_FLAGS "${graptor_cxxflags}")
endforeach()


set_target_properties(MCE PROPERTIES COMPILE_FLAGS ${graptor_cxxflags})
target_link_libraries(MCE ${EXTRA_LIBS} graptorlib)

add_subdirectory(scripts)
