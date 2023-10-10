#include <iostream>
#include <string>
#include "CSVReader.h"
#include "NeuralNetwork.h"
#include "FeatureScaler.h"
#include <cmath>

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

// test the network using some data
// comp: function. Given the output and expected result, determine whether the output is correct
float testNetwork(NeuralNetwork& nn, Data& testData, FeatureScaler fs, bool (*comp)(Vector, Vector), bool verbose=false) {
  int correctCount = 0;
  for (auto& point : testData) {
    auto expected = fs.unscaleOutput(point.expectedOutputs);
    auto prediction = fs.unscaleOutput(nn.predict(point.inputs));

    if (comp(expected, prediction))
      correctCount += 1;

    if (verbose) {
      std::cout << "expected:  ";
      printVector(expected);
      std::cout << "predicted: ";
      printVector(prediction);
      std::cout << std::endl;
    }
  }
  return float(correctCount) / testData.size();
}

// this is specific to my sample data (classification)
// compares the expected output and the predicted output to determine whether the network was correct
bool compare(Vector expected, Vector prediction) {
  int expectedY = int(expected[0]);
  int predictedY = int(round(prediction[0]));
  return expectedY == predictedY;
}

int main() {
  std::cout.precision(6);

  // fetch and process data
  const std::string datafile = "../data.csv"; // program runs from ./build

  CSVReader reader(datafile);
  Matrix rawData = reader.getData({});
  Data data = splitInputOutput(rawData, {2});

  auto trainTest = splitTrainTest(data);
  Data train = trainTest.first;
  Data test = trainTest.second;

  // apply feature scaling
  FeatureScaler fs(train);
  train = fs.scaleData(train);
  test = fs.scaleData(test);

  // new neural network
  NeuralNetwork nn({6, 8, 8, 8, 1}, NeuralNetwork::ActivationFunction::LEAKY_RELU);

  float accuracy = testNetwork(nn, test, fs, compare);
  std::cout << "Network test accuracy before: " << accuracy << std::endl;

  // train
  for (int i = 0; i < 500; i++) {
    nn.learn(train, 0.1);
  }

  // test
  accuracy = testNetwork(nn, test, fs, compare);
  std::cout << "Network test accuracy after:  " << accuracy << std::endl;

  // save
  try {
    nn.save("test2.nn");
  }
  catch(...) {
    std::cout << "could not save" << std::endl;
  }
  std::cout << "successfully saved" << std::endl;

  // load saved network
  NeuralNetwork nn2("test2.nn");
  nn2.learn(train, 0.1);
  // printstuff(nn2);
  // testNetwork(nn2, test, fs);

  return 0;
}
