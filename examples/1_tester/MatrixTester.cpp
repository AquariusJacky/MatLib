#include "MatrixTester.h"

MatrixTester::MatrixTester() {}
MatrixTester::~MatrixTester() {
  for (int i = 0; i < (int)testerVec.size(); i++) {
    delete testerVec[i];
  }
}

void MatrixTester::createTest(const std::string& test_name,
                              const std::string& operation_name,
                              const CPUMatrix& mat, const bool& isCUDA) {
  if (getOperationType().count(operation_name) == 0) {
    std::cout << "In " << test_name << ", " << operation_name
              << " is not a valid operation." << std::endl;
    return;
  }
  std::unordered_map<std::string, int> opertaionType = getOperationType();
  int operation_id = opertaionType[operation_name];

  RunningUnit* newTester = new RunningUnit;
  if (newTester->init(test_name, operation_id, mat, isCUDA) == 1) {
    delete newTester;
  }

  std::cout << "Created test \"" << test_name
            << "\" with operation: " << operation_name << "." << std::endl;
  testerVec.push_back(newTester);
}

void MatrixTester::createTest(const std::string& test_name,
                              const std::string& operation_name,
                              const CPUMatrix& mat, const float& val,
                              const bool& isCUDA) {
  if (getOperationType().count(operation_name) == 0) {
    std::cout << "In " << test_name << ", " << operation_name
              << " is not a valid operation." << std::endl;
    return;
  }
  std::unordered_map<std::string, int> opertaionType = getOperationType();
  int operation_id = opertaionType[operation_name];

  RunningUnit* newTester = new RunningUnit;
  if (newTester->init(test_name, operation_id, mat, val, isCUDA) == 1) {
    delete newTester;
  }

  std::cout << "Created test \"" << test_name
            << "\" with operation: " << operation_name << "." << std::endl;
  testerVec.push_back(newTester);
}

void MatrixTester::createTest(const std::string& test_name,
                              const std::string& operation_name,
                              const CPUMatrix& matA, const CPUMatrix& matB,
                              const bool& isCUDA) {
  if (getOperationType().count(operation_name) == 0) {
    std::cout << "In " << test_name << ", " << operation_name
              << " is not a valid operation." << std::endl;
    return;
  }
  std::unordered_map<std::string, int> opertaionType = getOperationType();
  int operation_id = opertaionType[operation_name];

  RunningUnit* newTester = new RunningUnit;
  if (newTester->init(test_name, operation_id, matA, matB, isCUDA) == 1) {
    delete newTester;
  }

  std::cout << "Created test \"" << test_name
            << "\" with operation: " << operation_name << "." << std::endl;
  testerVec.push_back(newTester);
}

void MatrixTester::runTest(const std::string& test_name) {
  int testerID = findTester(test_name);
  if (testerID == -1) return;
  runTest(testerID);
}

void MatrixTester::printResult(const std::string& name) {
  int testerID = findTester(name);
  if (testerID == -1) return;
  printResult(testerID);
}

void MatrixTester::printError(const std::string& nameA,
                              const std::string& nameB) {
  int testerAID = findTester(nameA);
  int testerBID = findTester(nameB);
  if (testerAID == -1 || testerBID == -1) return;
  printError(testerAID, testerBID);
}

void MatrixTester::printTime(const std::string& name) {
  int testerID = findTester(name);
  if (testerID == -1) return;
  printTime(testerID);
}

void MatrixTester::runTest(const int& testerID) {
  if (isValidTesterID(testerID) == 0) return;
  testerVec[testerID]->run();
}

void MatrixTester::printResult(const int& testerID) {
  if (isValidTesterID(testerID) == 0) return;
  std::cout << "\"" << testerVec[testerID]->name() << "\" result:\n"
            << testerVec[testerID]->result();
}

void MatrixTester::printError(const int& testerAID, const int& testerBID) {
  if (isValidTesterID(testerAID) == 0 || isValidTesterID(testerBID) == 0)
    return;

  RunningUnit* test1 = testerVec[testerAID];
  RunningUnit* test2 = testerVec[testerBID];

  if (test1->result_type != test2->result_type) {
    throw std::runtime_error("The result type of test 1 and 2 don't match");
  }

  float error;
  if (test1->result_type == RunningUnit::Result::MATRIX) {
    CPUMatrix matA = test1->result();
    CPUMatrix matB = test2->result();
    if (matA.m() != matB.m() || matA.n() != matB.n()) {
      std::cout << "CPUMatrix dimensions do not match" << std::endl;
    }
    CPUMatrix diff = matA - matB;
    error = diff.sum();
  } else {
    float valA = test1->result_value;
    float valB = test2->result_value;
    error = ((valA >= valB) ? (valA - valB) : (valB - valA));
  }

  std::cout << "Error between \"" << testerVec[testerAID]->name() << "\" and \""
            << testerVec[testerBID]->name() << "\": " << error << std::endl;
}

void MatrixTester::printTime(const int& testerID) {
  if (isValidTesterID(testerID) == 0) return;
  std::cout << "\"" << testerVec[testerID]->name()
            << "\" runtime: " << testerVec[testerID]->time() << " ms"
            << std::endl;
}

int MatrixTester::isValidTesterID(const int& testerID) {
  if (testerID < 0 || testerID >= (int)testerVec.size()) {
    std::cout << "Invalid tester ID: " << testerID << std::endl;
    return 0;
  }
  return 1;
}

int MatrixTester::findTester(const std::string& name) {
  int testerID = -1;
  for (int i = 0; i < (int)testerVec.size(); i++) {
    if (testerVec[i]->name() == name) testerID = i;
  }
  if (testerID == -1)
    std::cout << "Can't find tester with name \"" << name << "\"" << std::endl;

  return testerID;
}