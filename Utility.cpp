#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include "Utility.h"

void printVector(Vector v, bool endl) {
  std::cout << "[";
  for (int i = 0; i < v.size(); i++) {
    if (i != 0) std::cout << ", ";
    std::cout << std::setw(10) << v[i];
  }
  std::cout << "]";
  if (endl) std::cout << std::endl;
}

void printMatrix(Matrix m, bool endl) {
  std::cout << "[";
  for (int i = 0; i < m.size(); i++) {
    if (i != 0) std::cout << " ";
    printVector(m[i], i != m.size() - 1);
  }
  std::cout << "]";
  if (endl) std::cout << std::endl;
}

void printDataPoint(DataPoint p) {
  printVector(p.inputs, false);
  std::cout << " ";
  printVector(p.inputs, true);
}

void printData(Data d) {
  for (auto& p : d) printDataPoint(p);
}

Data splitInputOutput(Matrix& data, std::vector<int> outputCols) {
  if (data.size() == 0) {
    return Data();
  }

  Data newData;
  std::vector<bool> isOutput(data[0].size());
  for (int i = 0; i < isOutput.size(); i++) {
    isOutput[i] = (std::find(outputCols.begin(), outputCols.end(), i) != outputCols.end());
  }

  for (auto& row : data) {
    Vector inputRow, outputRow;
    for (int i = 0; i < row.size(); i++) {
      if (isOutput[i])
        outputRow.push_back(row[i]);
      else
        inputRow.push_back(row[i]);
    }
    newData.push_back(DataPoint{inputRow, outputRow});
  }

  return newData;
}

std::pair<Data, Data> splitTrainTest(Data& data, double testRatio) {
  Data train, test;

  std::vector<int> indices(data.size());
  for (int i = 0; i < indices.size(); i++) indices[i] = i;

  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(indices.begin(), indices.end(), g);

  int numTest = data.size() * testRatio;

  for (int i = 0; i < numTest; i++) {
    test.push_back(data[indices[i]]);
  }
  for (int i = numTest; i < data.size(); i++) {
    train.push_back(data[indices[i]]);
  }

  return std::pair<Data, Data>(train, test);
}
