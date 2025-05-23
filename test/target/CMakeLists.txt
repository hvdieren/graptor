function(graptor_create_target_test name)
    add_executable("${name}" "${name}.C")
    set_target_properties("${name}" PROPERTIES COMPILE_FLAGS "${graptor_cflags}")
    target_link_libraries("${name}" ${EXTRA_LIBS} graptorlib)
    add_test(NAME "target_${name}" COMMAND "${name}" WORKING_DIRECTORY test/target)
endfunction()
    
function(graptor_create_target_utest name op type vl)
    add_executable("${name}_${op}_${type}_${vl}" "${name}.C")
    set_target_properties("${name}_${op}_${type}_${vl}" PROPERTIES COMPILE_FLAGS "${graptor_cflags} -DBINOP=${op} -DTYPE=${type} -DVECTOR_LENGTH=${vl}")
    target_include_directories("${name}_${op}_${type}_${vl}" PUBLIC ${PROJECT_SOURCE_DIR}/test/target)
    target_link_libraries("${name}_${op}_${type}_${vl}" ${EXTRA_LIBS} graptorlib)
    add_test(NAME "target_${name}_${op}_${type}_${vl}" COMMAND "${name}_${op}_${type}_${vl}" WORKING_DIRECTORY test/target)
endfunction()

function(graptor_create_target_utest_cvt name t_type u_type vl)
    add_executable("${name}_${t_type}_${u_type}_${vl}" "${name}.C")
    set_target_properties("${name}_${t_type}_${u_type}_${vl}" PROPERTIES COMPILE_FLAGS "${graptor_cflags} -DT_TYPE=${t_type} -DU_TYPE=${u_type} -DVECTOR_LENGTH=${vl}")
    target_include_directories("${name}_${t_type}_${u_type}_${vl}" PUBLIC ${PROJECT_SOURCE_DIR}/test/target)
    target_link_libraries("${name}_${t_type}_${u_type}_${vl}" ${EXTRA_LIBS} graptorlib)
    add_test(NAME "target_${name}_${t_type}_${u_type}_${vl}" COMMAND "${name}_${t_type}_${u_type}_${vl}" WORKING_DIRECTORY test/target)
endfunction()

function(graptor_create_target_utest_arith)
    foreach(type uint8_t int8_t uint16_t int16_t uint32_t int32_t uint64_t int64_t)
	foreach(vl ${graptor_vl})
	    foreach(binop add sub min max bitwise_and bitwise_or bitwise_andnot)
		graptor_create_target_utest(binop ${binop} ${type} ${vl})
	    endforeach()
	endforeach()
    endforeach()

    foreach(type float double)
        foreach(vl ${graptor_vl})
	    foreach(binop add sub mul)
		graptor_create_target_utest(binop ${binop} ${type} ${vl})
	    endforeach()
        endforeach()
    endforeach()

    foreach(type bool logical_1 logical_2 logical_4 logical_8)
        foreach(vl ${graptor_vl})
	    foreach(binop logical_and logical_andnot logical_or)
		graptor_create_target_utest(binop ${binop} ${type} ${vl})
	    endforeach()
        endforeach()
    endforeach()
endfunction()

function(graptor_create_target_utest_cvt_logical)
    foreach(t_type logical_1 logical_2 logical_4 logical_8)
	foreach(u_type logical_1 logical_2 logical_4 logical_8)
	    foreach(vl 1 8 16 32 64 128 256)
		graptor_create_target_utest_cvt(unop_cvt ${t_type} ${u_type} ${vl})
	    endforeach()
	endforeach()
    endforeach()
    foreach(t_type logical_2 logical_4 logical_8)
	foreach(u_type logical_2 logical_4 logical_8)
	    foreach(vl 4)
		graptor_create_target_utest_cvt(unop_cvt ${t_type} ${u_type} ${vl})
	    endforeach()
	endforeach()
    endforeach()
    foreach(t_type logical_1 logical_2 logical_4 logical_8)
	foreach(vl 1 8 16 32 64 128 256)
	    graptor_create_target_utest_cvt(unop_cvt ${t_type} bool ${vl})
	    graptor_create_target_utest_cvt(unop_cvt bool ${t_type} ${vl})
	endforeach()
    endforeach()
    foreach(t_type logical_1)
	foreach(vl 1 8 16 32 64 128 256)
	    graptor_create_target_utest_cvt(unop_cvt ${t_type} bitfield2 ${vl})
	    graptor_create_target_utest_cvt(unop_cvt bitfield2 ${t_type} ${vl})
	endforeach()
    endforeach()
endfunction()

function(graptor_create_target_utest_cvt_fp)
    foreach(t_type float double)
	foreach(vl ${graptor_vl})
	    foreach(u_type float double)
		graptor_create_target_utest_cvt(unop_cvt ${t_type} ${u_type} ${vl})
	    endforeach()
	endforeach()
    endforeach()
    foreach(vl ${graptor_vl})
	#graptor_create_target_utest_cvt(unop_cvt float int32_t ${vl})
	#graptor_create_target_utest_cvt(unop_cvt double int32_t ${vl})
	graptor_create_target_utest_cvt(unop_cvt int32_t float ${vl})
	graptor_create_target_utest_cvt(unop_cvt int32_t double ${vl})
    endforeach()
endfunction()

function(graptor_create_target_utest_cvt_int)
    foreach(t_type uint8_t int8_t uint16_t int16_t uint32_t int32_t uint64_t int64_t)
	foreach(u_type uint8_t int8_t uint16_t int16_t uint32_t int32_t uint64_t int64_t)
	    foreach(vl 1 8 16 32 64 128 256)
		graptor_create_target_utest_cvt(unop_cvt ${t_type} ${u_type} ${vl})
	    endforeach()
	endforeach()
    endforeach()
    foreach(t_type uint16_t int16_t uint32_t int32_t uint64_t int64_t)
	foreach(u_type uint16_t int16_t uint32_t int32_t uint64_t int64_t)
	    foreach(vl 4)
		graptor_create_target_utest_cvt(unop_cvt ${t_type} ${u_type} ${vl})
	    endforeach()
	endforeach()
    endforeach()
    foreach(t_type uint8_t)
	foreach(vl 1 8 16 32 64 128 256)
	    graptor_create_target_utest_cvt(unop_cvt ${t_type} bitfield2 ${vl})
	    graptor_create_target_utest_cvt(unop_cvt bitfield2 ${t_type} ${vl})
	endforeach()
    endforeach()
endfunction()

function(graptor_create_target_utest_cmp)
    foreach(type uint8_t int8_t uint16_t int16_t uint32_t int32_t uint64_t int64_t float double)
        foreach(vl ${graptor_vl})
	    foreach(binop cmpeq cmpne cmpgt cmpge cmplt cmple)
		graptor_create_target_utest(binop_ttm ${binop} ${type} ${vl})
		graptor_create_target_utest(binop_ttv ${binop} ${type} ${vl})
	    endforeach()
	endforeach()
    endforeach()
endfunction()

    
#graptor_create_target_test(vladd)
#graptor_create_target_test(vladd4)
#graptor_create_target_test(vladd4b)

#set(graptor_vl 1 4 8 16 32 64 128 256)
set(graptor_vl 1 8 16 32 64 128 256)

graptor_create_target_utest_cmp()
graptor_create_target_utest_cvt_logical()
graptor_create_target_utest_cvt_fp()
graptor_create_target_utest_cvt_int()
graptor_create_target_utest_arith()



