# Neural Network in C++

This project was developed on Windows using Git Bash.

## What's this?
Neural Network in C++ is a project I created to explore neural networks. Inspired by [3Blue1Brown](https://www.youtube.com/c/3blue1brown)'s videos on neural networks, I set out to implement one myself to better understand how things fit together.

This project defines a neural network class with feature scaling, one-hot encoding, saving/loading the network. The layers and activation functions of the network can also be changed.

I have also included a Python notebook to generate sample data for the purposes of testing the network.

As it is currently, the main program takes the generated data, trains the network, tests the network on the test set, and saves the network details to a file.

## Setting Up
### Prereqs
- MingGW
    - Install package for compiling C++
- CMake
- Make (makefiles)
    - For Windows: `winget install GnuWin32.Make`, then add `C:\Program Files (x86)\GnuWin32\bin` to PATH
- Python (for generating sample data)

### Setup
1. Clone [nlohmann json](https://github.com/nlohmann/json) in `./lib`
2. Create a `CMakeLists.txt` file in `./lib` with this code: `add_subdirectory(json)`
3. Create a folder for build files `./build`
4. Obtain sample data. `generate_data.ipynb` generates some simple data for classification.

### Build
```
cd build
cmake .. -G "MinGW Makefiles"
make
```

### Run
The executable is created at `./build/neuralnet.exe`.

## General notes for future me
- CMake creates Makefiles, you only need to run CMake when you change CMakeLists.
- Makefiles are what define how to compile stuff. Run `make` when you change source code and want to recompile.
- Pls remember to start ssh agent before trying to pull/push.

## Todo
- Integrate the feature scaler into the network class; they don't need to be separate
- Test function can also go into the network class
- May need testing for loading a saved network.
