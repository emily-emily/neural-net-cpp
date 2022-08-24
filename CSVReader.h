#include <string>
#include <vector>

// reads csv data
class CSVReader {
  public:
    CSVReader(std::string filename, std::string delimeter=",", bool header=true);

    // reads data from the file, applying one-hot encoding to the specified columns
    std::vector<std::vector<double>> getData(std::vector<int> oneHotEncoding);

  private:
    bool header = true;
    std::string filename;
    std::string delimeter = ",";

    // splits a line by a delimeter
    std::vector<std::string> split(std::string line, std::string delimeter);
};
