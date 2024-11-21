#ifndef MATRIXTESTERCOMPARATOR_H
#define MATRIXTESTERCOMPARATOR_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

#include "Matrix.h"
#include "Matrix_CUDA.cuh"

enum {
  UNKNOWN = 0,

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

  // Binary operation with matrix
  ADD,
  DOT,
  CONVOLUTION,

  TOTAL_OPERATION
};

class RunningUnit {
 public:
  RunningUnit();
  ~RunningUnit();

  int init(const std::string& name, const int& operation_id, const Matrix& mat,
           const bool& isCUDA);
  int init(const std::string& name, const int& operation_id, const Matrix& mat,
           const float& val, const bool& isCUDA);
  int init(const std::string& name, const int& operation_id, const Matrix& matA,
           const Matrix& matB, const bool& isCUDA);

  void run();
  std::string name() { return test_name; }
  Matrix result() { return result_matrix; }
  float time() { return runtime; }

  // result may also be a single value: sum

 private:
  std::string test_name;
  int operation_type;
  Matrix* input_matrices;
  float operation_value;
  float runtime;
  bool useCUDA;

  bool ran;
  Matrix result_matrix;
  float result_value;

  void run_cpu();
  void run_gpu();
};

class MatrixTester {
 public:
  MatrixTester();
  ~MatrixTester();

  void createTest(const std::string& test_name,
                  const std::string& operation_name, const Matrix& matA,
                  const bool& isCUDA = false);
  void createTest(const std::string& test_name,
                  const std::string& operation_name, const Matrix& matA,
                  const float& value, const bool& isCUDA = false);
  void createTest(const std::string& test_name,
                  const std::string& operation_name, const Matrix& matA,
                  const Matrix& matB, const bool& isCUDA = false);
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
        {"transpose", TRANSPOSE},
        {"absolute", ABS},
        {"sum", SUM},

        {"fill", FILL},
        {"identity", IDENTITY},
        {"scale", SCALE},

        {"addition", ADD},
        {"dot", DOT},
        {"convolution", CONVOLUTION}};
    return operationTypes;
  }
};

#endif