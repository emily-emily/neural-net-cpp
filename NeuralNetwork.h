#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include "nlohmann/json.hpp"
#include "Utility.h"

using json = nlohmann::json;

// TODO: save and load neural network
class NeuralNetwork {
  public:
    enum ActivationFunction {
      SIGMOID = 0,
      RELU,
      LEAKY_RELU,
      NUM_FUNCTIONS
    };

    // create a neural network with layers with layerSizes nodes, and the same
    //   activation function in each layer
    NeuralNetwork(std::vector<int> layerSizes, ActivationFunction f=SIGMOID);
    // create a neural network specifying activation functions for different layers
    NeuralNetwork(std::vector<int> layerSizes, std::vector<ActivationFunction>);
    // create a neural network from a save
    NeuralNetwork(std::string filename);

    // runs one epoch
    void learn(Data data, double learnRate);

    // runs the inputs through the network and returns the output
    Vector predict(Vector inputs);

    // returns the number of epochs
    unsigned int getEpoch();

    // print internal values for debugging
    void printWeights();
    void printBiases();
    void printGradientsW();
    void printGradientsB();

    // saves the current neural network
    void save(std::string filename);

  private:
    // one layer in the neural network
    class Layer {
      public:
        Layer(int numNodesIn, int numNodesOut, ActivationFunction f=SIGMOID);
        Layer(json layerData);
        
        // print internal values for debugging
        void printWeights();
        void printBiases();
        void printGradientsW();
        void printGradientsB();

        // returns the JSON data for this layer
        json toJSON();

        // runs the inputs through the layer and returns the output
        Vector runLayer(Vector inputs);

        // updates the gradients in the current layer
        void updateGradients(Vector nodeValues);

        // applies gradients to weights and biases in the current layer
        void applyGradients(double learnRate);

        // resets all gradients
        void resetGradients();

        // calculates intermediate stuff for backward propagation (partial derivatives)
        Vector calculateOutputLayerNodeValues(Vector expected);
        // the same as above but for a hidden layer
        // takes the next layer (oldLayer) and its node values (oldNodeValues)
        Vector calculateHiddenLayerNodeValues(Layer oldLayer, Vector oldNodeValues);
        
      private:
        int numNodesIn, numNodesOut;
        Matrix weights;
        Matrix costGradientW;
        Vector biases;
        Vector costGradientB;
        ActivationFunction activationFunctionID;
        Function activationFunction;
        Function activationDerivative;
        // save these for gradient descent
        Vector inputs;
        Vector weightedInputs;
        Vector activations;
    };

    // sets of different activation functions and their derivatives
    static const Function activationFunctions[];
    static const Function activationDerivatives[];

    std::vector<Layer> layers;
    unsigned int epoch = 0;

    // calculates the error/cost for one output node
    static double nodeCost(double output, double expected);

    // derivative of the cost function
    static double nodeCostDerivative(double output, double expected);

    // runs input through the network and calculates the error/cost for one data point
    double cost(DataPoint point);

    // runs inputs through the network and calculates the error/cost for several data points
    double cost(Data points);

    // applies gradients on all layers
    void applyGradients(double learnRate);

    // resets all gradients to 0 in each layer
    void clearGradients();

    // runs one iteration of Gradient Descent
    void learnBatch(Data batch, double learnRate);

    // runs inputs through the network and updates all the gradients
    void updateAllGradients(DataPoint point);
};

#endif
