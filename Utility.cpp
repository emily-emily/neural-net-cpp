#include <iostream>
#include <algorithm>
#include "Utility.h"

void printVector(Vector v, bool endl) {
  std::cout << "[";
  for (int i = 0; i < v.size(); i++) {
    if (i != 0) std::cout << ", ";
    std::cout << v[i];
  }
  std::cout << "]";
  if (endl) std::cout << std::endl;
}

void printMatrix(Matrix m) {
  std::cout << "[";
  for (int i = 0; i < m.size(); i++) {
    printVector(m[i], i != m.size() - 1);
  }
  std::cout << "]" << std::endl;
}

void printDataPoint(DataPoint p) {
  printVector(p.inputs, false);
  std::cout << " ";
  printVector(p.inputs, true);
}

void printData(std::vector<DataPoint> d) {
  for (auto& p : d) printDataPoint(p);
}

std::vector<DataPoint> splitInputOutput(Matrix data, std::vector<int> outputCols) {
  if (data.size() == 0) {
    return std::vector<DataPoint>();
  }

  std::vector<DataPoint> newData;
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
