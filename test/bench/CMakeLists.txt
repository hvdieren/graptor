# Note: currently does not work on MacOSX as it does not respect
#       case-sensitivity of target names

set(graptor_undir_graphs rMatGraph_J_5_100_b2 orkut_undir_b2)
set(graptor_dir_graphs LiveJournal_dir_b2)
set(graptor_graph_dir ${PROJECT_SOURCE_DIR})
# BF
set(graptor_bench BFS BFSLVL CC FM MIS KC PR APR)
set(graptor_graph_types GGVEBO_vl1_default GGVEBO_vl16_default)
#set(graptor_graph_types GGVEBO_vl1_default)

function(graptor_create_test bench cfglst graph)
    string(REGEX REPLACE " " "_" cfg "${cfglst}")
    if(NOT TARGET ${bench}_${cfg})
	add_custom_command(
	    OUTPUT  "${bench}_${cfg}.C"
	    COMMAND ${PROJECT_SOURCE_DIR}/test/bench/create_bench.sh ${cfglst} ${PROJECT_SOURCE_DIR}/${bench}v.C "${bench}_${cfg}.C"
	    DEPENDS ${PROJECT_SOURCE_DIR}/test/bench/create_bench.sh ${PROJECT_SOURCE_DIR}/${bench}v.C
	    )

	add_executable(${bench}_${cfg} ${bench}_${cfg}.C)
	target_include_directories(${bench}_${cfg} PUBLIC ${PROJECT_SOURCE_DIR}/test/bench)
	set_target_properties(${bench}_${cfg} PROPERTIES COMPILE_FLAGS ${graptor_cflags})
	target_link_libraries(${bench}_${cfg} ${EXTRA_LIBS})
    endif()

    set(graptor_test_flags "")
    if(${graph} MATCHES "undir")
    	string(APPEND graptor_test_flags " -s")
    endif()

    string(REGEX MATCH "_vl[0-9]+_" graptor_find_vl ${cfg})
    if(${graptor_find_vl})
    	string(APPEND graptor_test_flags " -l ${graptor_find_vl}")
    else()
    	string(APPEND graptor_test_flags " -l 1")
    endif()
	
    add_test(NAME ${bench}_${cfg}_${graph}
	     COMMAND ${bench}_${cfg} -rounds 5 -c 16 -itimes -b ${graptor_test_flags} ${graptor_graph_dir}/${graph}
	     WORKING_DIRECTORY test/bench)

    set_property(TEST ${bench}_${cfg}_${graph}
		      PROPERTY
		      ENVIRONMENT ${graptor_ld_preload})
endfunction()

foreach(gtype ${graptor_graph_types})
    foreach(bench ${graptor_bench})
        # Check if the benchmark does only undirected graphs, or also
	# directed graphs
    	if(${bench} STREQUAL CC
	   OR ${bench} STREQUAL FM
	   OR ${bench} STREQUAL MIS
	   OR ${bench} STREQUAL KC)
	    set(graptor_graphs ${graptor_undir_graphs})
	else()
	    set(graptor_graphs ${graptor_undir_graphs})
	    list(APPEND graptor_graphs ${graptor_dir_graphs})
	endif()

	foreach(graph ${graptor_graphs})
	    graptor_create_test(${bench} ${gtype} ${graph})

	    if(${bench} STREQUAL CC
		OR ${bench} EQUAL FM
		OR ${bench} EQUAL KC
		OR ${bench} MATCHES /^BFS/
		)
		graptor_create_test(${bench} "${gtype} algorithmic_cadum" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_cadUm" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_caDum" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_caDUm" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_cAdum" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_cAdUm" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_cADum" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_cADUm" ${graph})
	    endif()
	    if(${bench} STREQUAL CC
		OR ${bench} STREQUAL FM
		OR ${bench} STREQUAL KC
		OR ${bench} STREQUAL BF
		OR ${bench} MATCHES /^BFS/
		)
		graptor_create_test(${bench} "${gtype} algorithmic_Cadum" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_CadUm" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_CaDum" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_CaDUm" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_CAdum" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_CAdUm" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_CADum" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_CADUm" ${graph})
	    endif()
	    if(${bench} MATCHES /PR$/)
		graptor_create_test(${bench} "${gtype} algorithmic_caDUM" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_caDUm" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_caDuM" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_caDum" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_cadUM" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_cadUm" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_caduM" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_cadum" ${graph})
	    endif()
	    if(${bench} STREQUAL MIS)
		graptor_create_test(${bench} "${gtype} algorithmic_cadum" ${graph})
		graptor_create_test(${bench} "${gtype} algorithmic_cadUm" ${graph})
	    endif()
	endforeach()
    endforeach()
endforeach()
