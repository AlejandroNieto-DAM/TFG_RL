execute_process(COMMAND "/home/nietoff/tfg/build/turtlebot3_autorace-master/turtlebot3_autorace_core/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/nietoff/tfg/build/turtlebot3_autorace-master/turtlebot3_autorace_core/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
