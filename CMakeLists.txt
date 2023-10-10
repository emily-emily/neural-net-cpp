cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(NeuralNetworkCpp VERSION 0.1)

add_subdirectory(lib)

set(SRCS
  main.cpp
  Utility.cpp
  CSVReader.cpp
  FeatureScaler.cpp
  NeuralNetwork.cpp
  Layer.cpp
)

add_executable(neuralnet ${SRCS})

target_include_directories(neuralnet PUBLIC lib/json/single_include)

target_link_libraries(neuralnet PRIVATE nlohmann_json::nlohmann_json)
