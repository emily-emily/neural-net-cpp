#ifndef UTILITY_H
#define UTILITY_H

#include <utility>
#include <vector>

// a collection of useful things

typedef std::vector<double> Vector;
typedef std::vector<std::vector<double>> Matrix;
typedef double(*Function)(double);
struct DataPoint {
  Vector inputs;
  Vector expectedOutputs;
};

void printVector(Vector v, bool endl=true);
void printMatrix(Matrix m);
void printDataPoint(DataPoint p);
void printData(std::vector<DataPoint> d);

// separates data columns into a input and expected output matrices
std::vector<DataPoint> splitInputOutput(Matrix data, std::vector<int> outputCols);

// TODO: train test split

#endif
