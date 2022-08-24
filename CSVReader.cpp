#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>
#include "CSVReader.h"

CSVReader::CSVReader(std::string filename, std::string delimeter, bool header)
  :filename(filename), delimeter(delimeter), header(header) {}

// currently ignores non-double values
// TODO: implement one-hot encoding?
std::vector<std::vector<double>> CSVReader::getData(std::vector<int> oneHotEncoding) {
  std::vector<std::vector<std::string>> rawData;
  std::vector<std::vector<double>> data;
  std::ifstream file;
  file.open(filename);

  if (!file.is_open()) {
    std::cerr << "Failed to open file " << filename << std::endl;
    return std::vector<std::vector<double>>();
  }

  // extract all data as strings
  std::string line;
  if (header) std::getline(file, line);
  while (std::getline(file, line)) {
    rawData.push_back(split(line, delimeter));
  }

  // find unique values in columns for one-hot encoding
  std::map<int, std::vector<std::string>> encodedValues;
  for (int& col : oneHotEncoding) {
    encodedValues[col] = std::vector<std::string>();
    for (int i = 0; i < rawData.size(); i++) {
      // save if key has not been seen before
      if (std::find(encodedValues[col].begin(), encodedValues[col].end(), rawData[i][col]) == encodedValues[col].end()) {
        encodedValues[col].push_back(rawData[i][col]);
      }
    }
  }

  // convert to double
  for (auto& row : rawData) {
    std::vector<double> rowDouble;

    for (int col = 0; col < row.size(); col++) {
      // one-hot encoding
      if (encodedValues.count(col) > 0) {
        // the column that should be 1
        int val = std::distance(
          encodedValues[col].begin(),
          std::find(encodedValues[col].begin(), encodedValues[col].end(), row[col])
        );

        // insert new columns
        for (int newCols = 0; newCols < encodedValues[col].size(); newCols++) {
          rowDouble.push_back(val == newCols);
        }
      }
      // directly convert to double
      else {
        try {
          rowDouble.push_back(std::stod(row[col]));
        }
        catch(...) {
          std::cerr << "Failed to parse " << row[col] << std::endl;
          rowDouble.push_back(0);
        }
      }
    }

    data.push_back(rowDouble);
  }

  file.close();

  return data;
}

std::vector<std::string> CSVReader::split(std::string line, std::string delimeter) {
  std::vector<std::string> v;
  int start = 0;
  int delimPos = line.find(delimeter, start);
  while (delimPos != std::string::npos) {
    v.push_back(line.substr(start, delimPos - start));
    start = delimPos + 1;
    delimPos = line.find(delimeter, start);
  }
  v.push_back(line.substr(start, delimPos)); // last column

  return v;
}
