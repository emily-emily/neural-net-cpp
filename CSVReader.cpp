#include <fstream>
#include <iostream>
#include "CSVReader.h"

CSVReader::CSVReader(std::string filename, std::string delimeter, bool header)
  :filename(filename), delimeter(delimeter), header(header) {}

// currently ignores non-double values
// TODO: implement one-hot encoding?
std::vector<std::vector<double>> CSVReader::getData() {
  std::vector<std::vector<double>> data;
  std::ifstream file;
  file.open(filename);

  if (!file.is_open()) {
    std::cerr << "Failed to open file " << filename << std::endl;
    return std::vector<std::vector<double>>();
  }

  std::string line;
  if (header) std::getline(file, line);
  while (std::getline(file, line)) {
    std::vector<std::string> rowStr = split(line, delimeter);
    std::vector<double> rowDouble;
    for (auto& value : rowStr) {
      try {
        rowDouble.push_back(std::stod(value));
      }
      catch(...) { // one-hot encoding
        // std::cerr << "Failed to parse " << value << std::endl;
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
