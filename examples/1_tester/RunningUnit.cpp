#include "MatrixTester.h"

RunningUnit::RunningUnit() {
  operation_value = 0;
  operation_type = UNKNOWN;
  result_value = 0;
  result_type = Result::MATRIX;
}
RunningUnit::~RunningUnit() { delete[] input_matrices; }

int RunningUnit::init(const std::string& name, const int& operation_id,
                      const CPU::Matrix& mat, const bool& isCUDA) {
  if (operation_id != ZEROS && operation_id != ONES &&
      operation_id != TRANSPOSE && operation_id != ABS && operation_id != SUM) {
    std::cout << "In " << test_name << ", the number of inputs is incorrect."
              << std::endl;
    return 1;
  }

  // m x n > 0
  if (mat.size() == 0) {
    std::cout << "CPU::Matrix size has to be larger than 0." << std::endl;
    return 1;
  }

  if (operation_id == SUM) result_type = Result::VALUE;

  input_matrices = new CPU::Matrix[1];
  input_matrices[0] = mat;
  test_name = name;
  operation_type = operation_id;
  useCUDA = isCUDA;
  ran = false;

  return 0;
}

int RunningUnit::init(const std::string& name, const int& operation_id,
                      const CPU::Matrix& mat, const float& val,
                      const bool& isCUDA) {
  if (operation_id != FILL && operation_id != SCALE &&
      operation_id != IDENTITY && operation_id != MAXPOOLING) {
    std::cout << "In " << test_name << ", the number of inputs is incorrect."
              << std::endl;
    return 1;
  }

  // m x n > 0
  if (mat.size() == 0) {
    std::cout << "CPU::Matrix size has to be larger than 0." << std::endl;
    return 1;
  }

  input_matrices = new CPU::Matrix[1];
  input_matrices[0] = mat;
  operation_value = val;
  test_name = name;
  operation_type = operation_id;
  useCUDA = isCUDA;
  ran = false;

  return 0;
}

int RunningUnit::init(const std::string& name, const int& operation_id,
                      const CPU::Matrix& matA, const CPU::Matrix& matB,
                      const bool& isCUDA) {
  if (operation_id != ADD && operation_id != DOT &&
      operation_id != CONVOLUTION) {
    std::cout << "In " << test_name << ", the number of inputs is incorrect."
              << std::endl;
    return 1;
  }
  switch (operation_id) {
    case ADD:
      if (matA.m() != matB.m() || matA.n() != matB.n()) {
        std::cout << "CPU::Matrix size has to be the same." << std::endl;
        return 1;
      }
      break;
    case DOT:
      if (matA.n() != matB.m()) {
        std::cout << "Number of rows in matA has to be equal to the number of "
                     "columns in matB."
                  << std::endl;
        return 1;
      }
      break;
    case CONVOLUTION:
      if (matA.size() < matB.size()) {
        std::cout << "Mask size cannot be larger than the matrix." << std::endl;
        return 1;
      }
    default:
      break;
  }
  // m x n > 0
  if (matA.size() == 0) {
    std::cout << "CPU::Matrix size has to be larger than 0." << std::endl;
    return 1;
  }

  input_matrices = new CPU::Matrix[2];
  input_matrices[0] = matA;
  input_matrices[1] = matB;
  test_name = name;
  operation_type = operation_id;
  useCUDA = isCUDA;
  ran = false;

  return 0;
}

void RunningUnit::run() {
  std::cout << "Running test \"" << test_name << "\"..." << std::endl;
  if (useCUDA)
    run_gpu();
  else
    run_cpu();
  ran = true;
}

void RunningUnit::run_cpu() {
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
  result_matrix = input_matrices[0];
  switch (operation_type) {
    case ZEROS:
      start_time = std::chrono::high_resolution_clock::now();
      result_matrix.zeros();
      end_time = std::chrono::high_resolution_clock::now();
      break;
    case ONES:
      start_time = std::chrono::high_resolution_clock::now();
      result_matrix.ones();
      end_time = std::chrono::high_resolution_clock::now();
      break;
    case TRANSPOSE:
      start_time = std::chrono::high_resolution_clock::now();
      result_matrix.T();
      end_time = std::chrono::high_resolution_clock::now();
      break;
    case ABS:
      start_time = std::chrono::high_resolution_clock::now();
      result_matrix.abs();
      end_time = std::chrono::high_resolution_clock::now();
      break;
    case SUM:
      start_time = std::chrono::high_resolution_clock::now();
      result_value = result_matrix.sum();
      end_time = std::chrono::high_resolution_clock::now();
      break;
    case FILL:
      start_time = std::chrono::high_resolution_clock::now();
      result_matrix.fill(operation_value);
      end_time = std::chrono::high_resolution_clock::now();
      break;
    case IDENTITY:
      start_time = std::chrono::high_resolution_clock::now();
      result_matrix.I(operation_value);
      end_time = std::chrono::high_resolution_clock::now();
      break;
    case SCALE:
      start_time = std::chrono::high_resolution_clock::now();
      result_matrix *= operation_value;
      end_time = std::chrono::high_resolution_clock::now();
      break;
    case MAXPOOLING:
      start_time = std::chrono::high_resolution_clock::now();
      result_matrix.maxPooling(operation_value);
      end_time = std::chrono::high_resolution_clock::now();
      break;
    case ADD:
      start_time = std::chrono::high_resolution_clock::now();
      result_matrix += input_matrices[1];
      end_time = std::chrono::high_resolution_clock::now();
      break;
    case DOT:
      start_time = std::chrono::high_resolution_clock::now();
      result_matrix.dot(input_matrices[1]);
      end_time = std::chrono::high_resolution_clock::now();
      break;
    case CONVOLUTION:
      start_time = std::chrono::high_resolution_clock::now();
      result_matrix.convolution(input_matrices[1], 1,
                                CPU::Matrix::PaddingType::FULL);
      end_time = std::chrono::high_resolution_clock::now();
      break;
  }
  std::chrono::duration<float, std::milli> duration = (end_time - start_time);
  runtime = duration.count();
}

void RunningUnit::run_gpu() {
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
  switch (operation_type) {
    case ZEROS: {
      start_time = std::chrono::high_resolution_clock::now();
      GPU::Matrix gpu_mat(input_matrices[0]);
      gpu_mat.fill(0);
      gpu_mat.toCPU(result_matrix);
      end_time = std::chrono::high_resolution_clock::now();
    } break;
    case ONES: {
      start_time = std::chrono::high_resolution_clock::now();
      GPU::Matrix gpu_mat(input_matrices[0]);
      gpu_mat.fill(1);
      gpu_mat.toCPU(result_matrix);
      end_time = std::chrono::high_resolution_clock::now();
    } break;
    // case TRANSPOSE: {
    //   start_time = std::chrono::high_resolution_clock::now();
    //   GPU::Matrix gpu_mat(input_matrices[0]);
    //   gpu_mat.T();
    //   gpu_mat.toCPU(result_matrix);
    //   end_time = std::chrono::high_resolution_clock::now();
    // } break;
    // case ABS: {
    //   start_time = std::chrono::high_resolution_clock::now();
    //   GPU::Matrix gpu_mat(input_matrices[0]);
    //   gpu_mat.abs(0);
    //   gpu_mat.toCPU(result_matrix);
    //   end_time = std::chrono::high_resolution_clock::now();
    // } break;
    // case SUM: {
    //   start_time = std::chrono::high_resolution_clock::now();
    //   GPU::Matrix gpu_mat(input_matrices[0]);
    //   gpu_mat.sum();
    //   gpu_mat.toCPU(result_matrix);
    //   end_time = std::chrono::high_resolution_clock::now();
    // } break;
    case FILL: {
      start_time = std::chrono::high_resolution_clock::now();
      GPU::Matrix gpu_mat(input_matrices[0]);
      gpu_mat.fill(operation_value);
      gpu_mat.toCPU(result_matrix);
      end_time = std::chrono::high_resolution_clock::now();
    } break;
    case IDENTITY: {
      start_time = std::chrono::high_resolution_clock::now();
      GPU::Matrix gpu_mat(input_matrices[0]);
      gpu_mat.fill(0);
      gpu_mat.toCPU(result_matrix);
      end_time = std::chrono::high_resolution_clock::now();
    } break;
    case SCALE: {
      std::chrono::high_resolution_clock::now();
      GPU::Matrix gpu_mat(input_matrices[0]);
      gpu_mat.scale(operation_value);
      gpu_mat.toCPU(result_matrix);
      end_time = std::chrono::high_resolution_clock::now();
    } break;
    case MAXPOOLING: {
      start_time = std::chrono::high_resolution_clock::now();
      GPU::Matrix gpu_input_matA(input_matrices[0]);
      gpu_input_matA.maxPooling(operation_value);
      gpu_input_matA.toCPU(result_matrix);
      end_time = std::chrono::high_resolution_clock::now();
    } break;
    case ADD: {
      start_time = std::chrono::high_resolution_clock::now();
      GPU::Matrix gpu_input_matA(input_matrices[0]);
      GPU::Matrix gpu_input_matB(input_matrices[1]);
      gpu_input_matA.add(gpu_input_matB);
      gpu_input_matA.toCPU(result_matrix);
      end_time = std::chrono::high_resolution_clock::now();
    } break;
    case DOT: {
      start_time = std::chrono::high_resolution_clock::now();
      GPU::Matrix gpu_input_matA(input_matrices[0]);
      GPU::Matrix gpu_input_matB(input_matrices[1]);
      gpu_input_matA.dot(gpu_input_matB);
      gpu_input_matA.toCPU(result_matrix);
      end_time = std::chrono::high_resolution_clock::now();
    } break;
    case CONVOLUTION: {
      start_time = std::chrono::high_resolution_clock::now();
      GPU::Matrix gpu_input_matA(input_matrices[0]);
      GPU::Matrix gpu_input_matB(input_matrices[1]);
      gpu_input_matA.convolution(gpu_input_matB);
      gpu_input_matA.toCPU(result_matrix);
      end_time = std::chrono::high_resolution_clock::now();
    } break;
  }
  std::chrono::duration<float, std::milli> duration = (end_time - start_time);
  runtime = duration.count();
}