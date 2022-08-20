#ifndef FEATURESCALER_H
#define FEATURESCALER_H

#include "Utility.h"

// feature scaling using min-max normalization
class FeatureScaler {
  public:
    FeatureScaler(const std::vector<DataPoint>& data);

    std::vector<DataPoint> scaleData(const std::vector<DataPoint>& data);
    Vector scaleInput(Vector input);
    Vector scaleOutput(Vector output);
    Vector unscaleOutput(Vector output);

  private:
    Vector inputMin, inputMax;
    Vector outputMin, outputMax;

    // apply normalization to a number
    static double scaleNumber(double min, double max, double x);

    // scale a number back to the original scale
    static double unscaleNumber(double min, double max, double x);
};

#endif
