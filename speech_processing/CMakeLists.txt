cmake_minimum_required(VERSION 3.0.2)
project(speech_processing)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)

 add_message_files(
   FILES
   command_and_age.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  )

catkin_package(
    CATKIN_DEPENDS message_runtime 
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  )


include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)


catkin_install_python(PROGRAMS src/speech_processing_server.py  src/speech_processing_client.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
