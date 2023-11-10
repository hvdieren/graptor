add_executable(MCE MCE.C)
set_target_properties(MCE PROPERTIES COMPILE_FLAGS ${graptor_cxxflags})
target_link_libraries(MCE ${EXTRA_LIBS} graptorlib)

add_subdirectory(scripts)
