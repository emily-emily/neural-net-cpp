cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(NeuralNetworkCpp VERSION 0.1)

set(SRCS
  main.cpp
  Utility.cpp
  CSVReader.cpp
  FeatureScaler.cpp
  NeuralNetwork.cpp
  Layer.cpp
)

add_executable(neuralnet ${SRCS})
