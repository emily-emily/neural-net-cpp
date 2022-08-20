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

int main() {
  std::cout.precision(5);
  const std::string datafile = "../data.csv"; // program runs from ./build

  CSVReader reader(datafile);
  Matrix rawData = reader.getData();

  Data data = splitInputOutput(rawData, {3});
  // printData(data);

  FeatureScaler fs(data);
  data = fs.scaleData(data);
  // printData(data);

  NeuralNetwork nn({3, 8, 8, 8, 1}, NeuralNetwork::ActivationFunction::LEAKY_RELU);

  printstuff(nn);

  for (int i = 0; i < 50; i++) {
    nn.learn(data, 0.1);
    // printstuff(nn);
  }

  return 0;
}
