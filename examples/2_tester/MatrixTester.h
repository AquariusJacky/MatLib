#ifndef MATRIXTESTER_H
#define MATRIXTESTER_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

#include "MatLib/Matrix.h"

enum {

  // Unary operation
  ZEROS,
  ONES,
  TRANSPOSE,
  ABS,
  SUM,

  // Binary operation with value
  FILL,
  IDENTITY,
  SCALE,
  MAXPOOLING,

  // Binary operation with matrix
  ADD,
  DOT,
  CONVOLUTION,

  TOTAL_OPERATION,
  UNKNOWN
};

class RunningUnit {
 public:
  friend class MatrixTester;
  RunningUnit();
  ~RunningUnit();

  int init(const std::string& name, const int& operation_id,
           const CPU::Matrix& mat, const bool& isCUDA);
  int init(const std::string& name, const int& operation_id,
           const CPU::Matrix& mat, const float& val, const bool& isCUDA);
  int init(const std::string& name, const int& operation_id,
           const CPU::Matrix& matA, const CPU::Matrix& matB,
           const bool& isCUDA);

  void run();
  std::string name() { return test_name; }
  CPU::Matrix result() { return result_matrix; }
  float time() { return runtime; }

  // result may also be a single value: sum

 private:
  std::string test_name;
  int operation_type;
  CPU::Matrix* input_matrices;
  float operation_value;
  float runtime;
  bool useCUDA;

  bool ran;
  int result_type;
  enum Result {
    MATRIX,
    VALUE,
  };
  CPU::Matrix result_matrix;
  float result_value;

  void run_cpu();
  void run_gpu();
};

class MatrixTester {
 public:
  MatrixTester();
  ~MatrixTester();

  void createTest(const std::string& test_name,
                  const std::string& operation_name, const CPU::Matrix& matA,
                  const bool& isCUDA = false);
  void createTest(const std::string& test_name,
                  const std::string& operation_name, const CPU::Matrix& matA,
                  const float& value, const bool& isCUDA = false);
  void createTest(const std::string& test_name,
                  const std::string& operation_name, const CPU::Matrix& matA,
                  const CPU::Matrix& matB, const bool& isCUDA = false);
  void runTest(const std::string& test_name);

  void printResult(const std::string& name);
  void printError(const std::string& nameA, const std::string& nameB);
  void printTime(const std::string& name);

 private:
  std::vector<RunningUnit*> testerVec;

  int isValidTesterID(const int& testerID);
  int findTester(const std::string& name);

  void runTest(const int& testerID);
  void printResult(const int& testerID);
  void printError(const int& testerAID, const int& testerBID);
  void printTime(const int& testerID);

  static const std::unordered_map<std::string, int>& getOperationType() {
    // Local static variable - guaranteed to be initialized only once
    // Thread-safe in C++11 and later due to magic statics
    static const std::unordered_map<std::string, int> operationTypes = {
        {"zeros", ZEROS},
        {"ones", ONES},
        {"transpose", TRANSPOSE},  // Not implemented in GPU yet
        {"absolute", ABS},         // Not implemented in GPU yet
        {"sum", SUM},

        {"fill", FILL},
        {"identity", IDENTITY},
        {"scale", SCALE},
        {"maxPooling", MAXPOOLING},

        {"add", ADD},
        {"dot", DOT},
        {"conv", CONVOLUTION}};
    return operationTypes;
  }
};

#endif