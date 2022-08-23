#include <iostream>
#include <string>
#include "CSVReader.h"
#include "NeuralNetwork.h"
#include "FeatureScaler.h"

void printstuff(NeuralNetwork& nn) {
  std::cout << "epoch " << nn.getEpoch() << " -------------------------------------------------------------" << std::endl;
  std::cout << std::endl;

  std::cout << "weights" << std::endl;
  nn.printWeights();
  std::cout << std::endl;

  std::cout << "biases" << std::endl;
  nn.printBiases();
  std::cout << std::endl;

  std::cout << "weight gradients" << std::endl;
  nn.printGradientsW();
  std::cout << std::endl;

  std::cout << "bias gradients" << std::endl;
  nn.printGradientsB();
  std::cout << std::endl;
}

void testNetwork(NeuralNetwork& nn, Data& test, FeatureScaler fs) {
  for (auto& point : test) {
    std::cout << "expected:  ";
    printVector(fs.unscaleOutput(point.expectedOutputs));
    std::cout << "predicted: ";
    printVector(fs.unscaleOutput(nn.predict(point.inputs)));
    std::cout << std::endl;
  }
}

int main() {
  std::cout.precision(6);

  // fetch and process data
  const std::string datafile = "../data.csv"; // program runs from ./build

  CSVReader reader(datafile);
  Matrix rawData = reader.getData();
  Data data = splitInputOutput(rawData, {3});

  auto trainTest = splitTrainTest(data);
  Data train = trainTest.first;
  Data test = trainTest.second;

  // apply feature scaling
  FeatureScaler fs(train);
  train = fs.scaleData(train);
  test = fs.scaleData(test);

  // new neural network
  NeuralNetwork nn({3, 8, 8, 8, 1}, NeuralNetwork::ActivationFunction::LEAKY_RELU);

  // train
  for (int i = 0; i < 500; i++) {
    nn.learn(train, 0.1);
  }

  // test
  testNetwork(nn, test, fs);

  // save
  try {
    nn.save("test.nn");
  }
  catch(...) {
    std::cout << "could not save" << std::endl;
  }
  std::cout << "successfully saved" << std::endl;

  // load saved network
  NeuralNetwork nn2("test.nn");
  nn2.learn(train, 0.1);
  printstuff(nn2);
  testNetwork(nn2, test, fs);

  return 0;
}
