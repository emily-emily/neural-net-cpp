#include "FeatureScaler.h"

FeatureScaler::FeatureScaler(const std::vector<DataPoint>& data) {
  inputMin = data[0].inputs;
  inputMax = data[0].inputs;
  outputMin = data[0].expectedOutputs;
  outputMax = data[0].expectedOutputs;

  for (auto& point : data) {
    // handle inputs
    for (int i = 0; i < point.inputs.size(); i++) {
      if (point.inputs[i] < inputMin[i]) inputMin[i] = point.inputs[i];
      if (point.inputs[i] > inputMax[i]) inputMax[i] = point.inputs[i];
    }

    // handle outputs
    for (int i = 0; i < point.expectedOutputs.size(); i++) {
      if (point.expectedOutputs[i] < outputMin[i]) outputMin[i] = point.expectedOutputs[i];
      if (point.expectedOutputs[i] > outputMax[i]) outputMax[i] = point.expectedOutputs[i];
    }
  }
}


std::vector<DataPoint> FeatureScaler::scaleData(const std::vector<DataPoint>& data) {
  std::vector<DataPoint> scaled;
  for (auto& point : data) {
    scaled.push_back(DataPoint{scaleInput(point.inputs), scaleOutput(point.expectedOutputs)});
  }
  return scaled;
}

Vector FeatureScaler::scaleInput(Vector input) {
  Vector scaled;
  for (int i = 0; i < input.size(); i++) {
    scaled.push_back(scaleNumber(inputMin[i], inputMax[i], input[i]));
  }
  return scaled;
}

Vector FeatureScaler::scaleOutput(Vector output) {
  Vector scaled;
  for (int i = 0; i < output.size(); i++) {
    scaled.push_back(scaleNumber(outputMin[i], outputMax[i], output[i]));
  }
  return scaled;
}

Vector FeatureScaler::unscaleOutput(Vector output) {
  Vector unscaled;
  for (int i = 0; i < output.size(); i++) {
    unscaled.push_back(unscaleNumber(outputMin[i], outputMax[i], output[i]));
  }
  return unscaled;
}

double FeatureScaler::scaleNumber(double min, double max, double x) { return (x - min) / (max - min); }

double FeatureScaler::unscaleNumber(double min, double max, double x) { return x * (max - min) + min; }
