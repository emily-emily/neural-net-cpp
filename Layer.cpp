#include <cmath>
#include "NeuralNetwork.h"

#include <iostream>

NeuralNetwork::Layer::Layer(int nIn, int nOut, ActivationFunction f)
    : numNodesIn(nIn)
    , numNodesOut(nOut)
    , activationFunction(activationFunctions[f])
    , activationDerivative(activationDerivatives[f]) {
  weights = Matrix(numNodesIn);
  costGradientW = Matrix(numNodesIn);
  for (int in = 0; in < numNodesIn; in++) {
    weights[in] = Vector(numNodesOut);
    costGradientW[in] = Vector(numNodesOut);
    for (int out = 0; out < numNodesOut; out++) {
      // randomly seed weights with a number between -1 and 1
      weights[in][out] = (rand() / double(RAND_MAX) * 2 - 1) / sqrt(numNodesIn);
    }
  }
  biases = Vector(numNodesOut);
  costGradientB = Vector(numNodesOut);
  inputs = Vector(numNodesOut);
  weightedInputs = Vector(numNodesOut);
  activations = Vector(numNodesOut);
}

void NeuralNetwork::Layer::printWeights() { printMatrix(weights); }

void NeuralNetwork::Layer::printBiases() { printVector(biases); }

void NeuralNetwork::Layer::printGradientsW() { printMatrix(costGradientW); }

void NeuralNetwork::Layer::printGradientsB() { printVector(costGradientB); }

Vector NeuralNetwork::Layer::runLayer(Vector layerInputs) {
  inputs = layerInputs;
  // calculate weighted inputs
  for (int out = 0; out < numNodesOut; out++) {
    weightedInputs[out] = biases[out];
    for (int in = 0; in < numNodesIn; in++) {
      weightedInputs[out] += inputs[in] * weights[in][out];
    }
    // apply activation function
    activations[out] = activationFunction(weightedInputs[out]);
  }

  return activations;
}

void NeuralNetwork::Layer::updateGradients(Vector nodeValues) {
  for (int out = 0; out < numNodesOut; out++) {
    for (int in = 0; in < numNodesIn; in++) {
      // partial derivative of cost wrt weight
      double x = inputs[in] * nodeValues[out];
      costGradientW[in][out] += x;
    }
    costGradientB[out] += nodeValues[out];
  }
}

void NeuralNetwork::Layer::applyGradients(double learnRate) {
  for (int out = 0; out < numNodesOut; out++) {
    biases[out] -= costGradientB[out] * learnRate;
    for (int in = 0; in < numNodesIn; in++) {
      weights[in][out] -= costGradientW[in][out] * learnRate;
    }
  }
}

void NeuralNetwork::Layer::resetGradients() {
  costGradientW = Matrix(numNodesIn);
  for (int in = 0; in < numNodesIn; in++) {
    costGradientW[in] = Vector(numNodesOut);
  }
  costGradientB = Vector(numNodesOut);
}

Vector NeuralNetwork::Layer::calculateOutputLayerNodeValues(Vector expected) {
  int len = expected.size();
  Vector nodeValues = Vector(len);
  for (int i = 0; i < len; i++) {
    double cDerivative = nodeCostDerivative(activations[i], expected[i]);
    double aDerivative = activationDerivative(weightedInputs[i]);
    nodeValues[i] = cDerivative * aDerivative;
  }

  return nodeValues;
}

Vector NeuralNetwork::Layer::calculateHiddenLayerNodeValues(Layer oldLayer, Vector oldNodeValues) {
  Vector newNodeValues = Vector(numNodesOut);

  for (int newNode = 0; newNode < newNodeValues.size(); newNode++) {
    double newNodeValue = 0;
    for (int oldNode = 0; oldNode < oldNodeValues.size(); oldNode++) {
      // partial derivative of weighted input wrt input
      double weightedInputDerivative = oldLayer.weights[newNode][oldNode];
      newNodeValue += weightedInputDerivative * oldNodeValues[oldNode];
    }
    newNodeValue *= activationDerivative(weightedInputs[newNode]);
    newNodeValues[newNode] = newNodeValue;
  }

  return newNodeValues;
}
