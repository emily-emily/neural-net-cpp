#include <string>
#include <vector>

// reads csv data
class CSVReader {
  public:
    CSVReader(std::string filename, std::string delimeter=",", bool header=true);

    // reads data from the file
    std::vector<std::vector<double>> getData();

  private:
    bool header = true;
    std::string filename;
    std::string delimeter = ",";

    // splits a line by a delimeter
    std::vector<std::string> split(std::string line, std::string delimeter);
};
