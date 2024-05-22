execute_process(COMMAND "/home/nietoff/tfg/build/turtlebot3_machine_learning-master/turtlebot3_dqn/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/nietoff/tfg/build/turtlebot3_machine_learning-master/turtlebot3_dqn/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
