#ifndef FEATURESCALER_H
#define FEATURESCALER_H

#include "Utility.h"

// feature scaling using min-max normalization
class FeatureScaler {
  public:
    FeatureScaler(const Data& data);

    Data scaleData(const Data& data);
    Vector scaleInput(Vector input);
    Vector scaleOutput(Vector output);
    Vector unscaleOutput(Vector output);
    std::vector<Vector> unscaleManyOutputs(std::vector<Vector> outputs);

  private:
    Vector inputMin, inputMax;
    Vector outputMin, outputMax;

    // apply normalization to a number
    static double scaleNumber(double min, double max, double x);

    // scale a number back to the original scale
    static double unscaleNumber(double min, double max, double x);
};

#endif
