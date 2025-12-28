cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_COMPILER "nvc++")
project(openacc_test)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# Use the modern CMake way to find and enable OpenACC
find_package(OpenACC REQUIRED)
find_package(rclcpp REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(test_openacc_node src/test_openacc.cpp)

target_compile_options(test_openacc_node PRIVATE -Minfo=accel)

# Link against the imported OpenACC CXX target
# This automatically handles flags and libraries for the chosen compiler
target_link_libraries(test_openacc_node OpenACC::OpenACC_CXX)

ament_target_dependencies(test_openacc_node
  "rclcpp"
)

install(TARGETS
  test_openacc_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
