# Generate the .f files using m4 if it hasn't been done yet

find_program(m4_exec m4 REQUIRED)

file(GLOB m4_files *.m4)

foreach(m4_file IN LISTS m4_files)
  get_filename_component(fname ${m4_file} NAME_WE)
  set(outfile "${PROJECT_SOURCE_DIR}/src/${fname}.f")
  if(NOT EXISTS ${outfile})
    message(STATUS "generating ${fname}.f")
    execute_process(
      COMMAND ${m4_exec} ${m4_file} -I${CMAKE_CURRENT_LIST_DIR}
      OUTPUT_FILE ${outfile})
  endif()
endforeach()
