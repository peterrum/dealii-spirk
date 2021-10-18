CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

FIND_PACKAGE(deal.II 9.3 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

SET( TARGET_SRC hyperdeal.cc )

INCLUDE_DIRECTORIES(include ${CMAKE_CURRENT_BINARY_DIR}/include)

PROJECT(irk)

ADD_LIBRARY(irk  dummy.cc)

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug\nPlease call 'make' to build the debug project"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release\nPlease call 'make' to build the release project"
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()

DEAL_II_SETUP_TARGET(irk)

ADD_EXECUTABLE( "irk-2D" "main.cc")
DEAL_II_SETUP_TARGET("irk-2D")
target_compile_definitions("irk-2D" PUBLIC -DIRK_DIMENSION=2)
TARGET_LINK_LIBRARIES("irk-2D" "irk")

ADD_EXECUTABLE( "irk-3D" "main.cc")
DEAL_II_SETUP_TARGET("irk-3D")
target_compile_definitions("irk-3D" PUBLIC -DIRK_DIMENSION=3)
TARGET_LINK_LIBRARIES("irk-3D" "irk")
