cmake_minimum_required(VERSION 3.8)
project(fdpose_perception NONE)
find_package(ament_cmake REQUIRED)

# Keep the source CLI and the installed runtime on the same implementation.
set(repo_root "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
install(DIRECTORY "${repo_root}/apps" DESTINATION share/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)
install(DIRECTORY "${repo_root}/src/fdonpose" DESTINATION share/${PROJECT_NAME}/src
        FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)
ament_package()
