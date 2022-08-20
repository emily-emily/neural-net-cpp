#include <cmath>
#include "NeuralNetwork.h"

#include <iostream>

const Function NeuralNetwork::activationFunctions[ActivationFunction::NUM_FUNCTIONS] = {
  [](double a) { return 1 / (1 + exp(-a)); }, // sigmoid
  [](double a) { return (a > 0) ? a : 0; }, // relu
  [](double a) { return (a > 0) ? a : 0.01*a; } // leaky relu
};

const Function NeuralNetwork::activationDerivatives[ActivationFunction::NUM_FUNCTIONS] = {
  [](double a) { // sigmoid
    double activation = NeuralNetwork::activationFunctions[SIGMOID](a);
    return activation * (1 - activation);
  },
  [](double a) { return (a > 0) ? 1.0 : 0; }, // relu
  [](double a) { return (a > 0) ? 1.0 : 0.01; } // leaky relu
};

NeuralNetwork::NeuralNetwork(std::vector<int> layerSizes, ActivationFunction f) {
  for (int i = 0; i < layerSizes.size() - 1; i++) {
    layers.push_back(Layer(layerSizes[i], layerSizes[i + 1], f));
  }
}

NeuralNetwork::NeuralNetwork(std::vector<int> layerSizes, std::vector<ActivationFunction> fs) {
  for (int i = 1; i < layerSizes.size(); i++) {
    layers.push_back(Layer(layerSizes[i], layerSizes[i + 1], fs[i]));
  }
}

void NeuralNetwork::learn(std::vector<DataPoint> data, double learnRate) {
  // TODO: separate data into batches; add batchSize

  // TODO: learn on each batch

  // for now, just full batch training
  learnBatch(data, learnRate);
  epoch++;
}

void NeuralNetwork::learnBatch(std::vector<DataPoint> batch, double learnRate) {
  
  clearGradients(); // temp, so gradients are still there to print after the batch
  std::cerr << "cost: " << cost(batch) << std::endl;

  // run through all points in the batch
  for (auto& point : batch) {
    updateAllGradients(point);
  }

  // apply gradients to each layer
  // divide by batch size to average gradients in the batch
  applyGradients(learnRate / batch.size());

  // clear gradients for the next batch
  // clearGradients();
}

unsigned int NeuralNetwork::getEpoch() { return epoch; }

void NeuralNetwork::printWeights() {
  for (auto& l : layers) l.printWeights();
}

void NeuralNetwork::printBiases() {
  for (auto& l : layers) l.printBiases();
}

void NeuralNetwork::printGradientsW() {
  for (auto& l : layers) l.printGradientsW();
}

void NeuralNetwork::printGradientsB() {
  for (auto& l : layers) l.printGradientsB();
}

double NeuralNetwork::nodeCost(double output, double expected) {
  double diff = output - expected;
  return diff * diff;
}

double NeuralNetwork::nodeCostDerivative(double output, double expected) {
  return 2 * (output - expected);
}

double NeuralNetwork::cost(DataPoint point) {
  Vector outputs = propagateForward(point.inputs);
  double cost = 0;
  for (int i = 0; i < outputs.size(); i++) {
    cost += nodeCost(outputs[i], point.expectedOutputs[i]);
  }
  return cost;
}

double NeuralNetwork::cost(std::vector<DataPoint> points) {
  double totalCost = 0;
  for (auto p : points) {
    totalCost += cost(p);
  }
  return totalCost / points.size();
}

Vector NeuralNetwork::propagateForward(Vector inputs) {
  for (auto& l : layers) {
    inputs = l.runLayer(inputs);
  }
  return inputs;
}

void NeuralNetwork::applyGradients(double learnRate) {
  for (auto& l : layers) {
    l.applyGradients(learnRate);
  }
}

void NeuralNetwork::clearGradients() {
  for (auto& l : layers) {
    l.resetGradients();
  }
}

void NeuralNetwork::updateAllGradients(DataPoint point) {
  // run inputs through the network
  propagateForward(point.inputs);

  // update gradients of output layer
  Layer& outputLayer = layers[layers.size() - 1];
  Vector nodeValues = outputLayer.calculateOutputLayerNodeValues(point.expectedOutputs);
  outputLayer.updateGradients(nodeValues);

  // update gradients of hidden layers
  for (int i = layers.size() - 2; i >= 0; i--) {
    Layer& hiddenLayer = layers[i];
    nodeValues = hiddenLayer.calculateHiddenLayerNodeValues(layers[i + 1], nodeValues);
    hiddenLayer.updateGradients(nodeValues);
  }
}
