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

typedef std::vector<DataPoint> Data;

void printVector(Vector v, bool endl=true);
void printMatrix(Matrix m, bool endl=true);
void printDataPoint(DataPoint p);
void printData(Data d);

// separates data columns into a input and expected output matrices
Data splitInputOutput(Matrix& data, std::vector<int> outputCols);

// separates data into train and test sets
std::pair<Data, Data> splitTrainTest(Data& data, double testRatio=0.2);

#endif
