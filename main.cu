#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <cuda.h>
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

#define BLOCK_SIZE 128
#define BLOCKS_COUNT(n) ((size_t)ceil(((float)n)/(float)BLOCK_SIZE))

/**
 * Class declarations
 * */
class Rastrigin;
class DEInstance;
class DifferentialEvolution;

/**
 * Type declarations
 * */
typedef std::vector<double> params_type;
typedef DEInstance instance_type;

/**
 * Kernels
 * */

__global__ void transformNumber(double* src, int size, double lower, double upper) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    src[idx] = src[idx] * (upper - lower) + lower;
  }
}

class RandomNumberGenerator {
protected:
    curandGenerator_t generator;

    double* _generateNumbers(size_t n);
public:
    RandomNumberGenerator() {
      curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937);
      curandSetPseudoRandomGeneratorSeed(generator, time(nullptr));
    }

    params_type generate(size_t n, double lowerBound, double upperBound);
};


class Rastrigin {
protected:
    size_t mDimensions;
    double mLowerLimit;
    double mUpperLimit;
    double mA;

public:
    Rastrigin(size_t dimensions, double lowerLimit, double upperLimit, double A)
        : mDimensions(dimensions), mLowerLimit(lowerLimit), mUpperLimit(upperLimit), mA(A) {}

    bool isWithinConstraint(const params_type& params) const;
    double evaluate(const params_type& params) const;
    thrust::host_vector<instance_type> generateParamSet(size_t n) const;
};


class DEInstance {
protected:
    params_type* mParams;
    double mValue;

public:
    __host__ __device__
    DEInstance() : mParams(nullptr), mValue(-1) {}

    __host__ __device__
    DEInstance(params_type* params, const Rastrigin& func) : mParams(params), mValue(func.evaluate(*params)) {}

    __host__ __device__
    DEInstance(const DEInstance& other) : mParams(other.mParams), mValue(other.mValue) {}

    __host__ __device__
    bool isEmpty() {
      return mParams == nullptr;
    }

    const params_type* getParams() const{
      return mParams;
    }

    __host__ __device__
    double getValue() const {
      return mValue;
    }

    __host__ __device__
    bool operator>(const DEInstance& other) const { return mValue > other.mValue; }

    __host__ __device__
    bool operator>=(const DEInstance& other) const { return mValue >= other.mValue; }

    __host__ __device__
    bool operator<(const DEInstance& other) const { return mValue < other.mValue; }

    __host__ __device__
    bool operator<=(const DEInstance& other) const { return mValue <= other.mValue; }
};

class DifferentialEvolution {
protected:
    size_t mIterations, mPopulationSize;
    bool mOptimized;
    thrust::host_vector<instance_type> mInstances, mMutations;
    size_t mBestIndex;
    double mMutationParam, mCrossoverParam;

    void initialize(const Rastrigin& function);
    instance_type getMutation(const Rastrigin& function) const;
    void performIteration(const Rastrigin& function);
    void determineBest(const Rastrigin& function);
    size_t getRandomIndex() const;
    bool shouldMutate() const;

public:
    struct randomized_comparison {
    protected:
        double mThreshold;
        thrust::minstd_rand rng;
        thrust::uniform_real_distribution<double> dist;

    public:
        explicit randomized_comparison(double threshold) : mThreshold(threshold), rng(), dist(thrust::uniform_real_distribution<double>(0, 1)) {}

        __host__ __device__
        DEInstance operator()(const DEInstance& mutated, const DEInstance& original) {
          if (dist(rng) >= mThreshold)
            return original;

          return mutated.getValue() < original.getValue() ? mutated : original;
        }
    };

    DifferentialEvolution(size_t iterations, size_t populationSize, double mutationParam, double crossoverParam)
    : mIterations(iterations), mOptimized(false), mInstances(thrust::host_vector<instance_type>(populationSize)), mMutations(thrust::host_vector<instance_type>(populationSize)),
      mBestIndex(0), mPopulationSize(populationSize), mMutationParam(mutationParam), mCrossoverParam(crossoverParam) {}

    void optimize(Rastrigin function);
    instance_type getBest() const;
};

/**
 * Structure that will compare mutated and existing variants and then replace the latter if the former is more optimized
 * */
void display(const params_type& params) {
  for (auto it = params.cbegin(); it != params.cend(); it++)
    std::cout << *it << " ";
  std::cout << std::endl;
}

bool isGpuAvailable() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);

  return nDevices > 0;
}

int main() {
  if (!isGpuAvailable()) {
    std::cerr << "No GPU found, please try running in Google Colab" << std::endl;
    return EXIT_FAILURE;
  }

  Rastrigin func(3, -5.12, 5.12, 10);
  DifferentialEvolution optimizer(100, 2000, 0.25, 0.75);

  optimizer.optimize(func);

  instance_type bestSolution = optimizer.getBest();

  std::cout << "The best solution: " << bestSolution.getValue() << std::endl;
  display(*bestSolution.getParams());

  return 0;
}

/**
 * Implementation of random generator
 * */
params_type RandomNumberGenerator::generate(size_t n, double lowerBound, double upperBound) {
  // Generate the numbers in uniform format
  // Numbers will be on the device
  double* numbersDev = _generateNumbers(n);

  // Transform numbers from [0:1] to [lowerBound:upperBound]
  transformNumber<<<BLOCKS_COUNT(n), BLOCK_SIZE>>>(numbersDev, n, lowerBound, upperBound);

  // Copy from device to host and turn into more comfortable format
  auto* numbersHost = new double[n];
  cudaMemcpy(numbersHost, numbersDev, sizeof(double) * n, cudaMemcpyDeviceToHost);

  params_type result(numbersHost, numbersHost + n);

  // Free data and return vector
  cudaFree(numbersDev);
  free(numbersHost);
  return result;
}

double *RandomNumberGenerator::_generateNumbers(size_t n) {
  size_t bytesCount = n * sizeof(double);
  double *hostData, *devData;
  hostData = new double[n];

  cudaMalloc((void**)&devData, bytesCount);

  curandGenerateUniformDouble(generator, devData, n);

  cudaMemcpy(hostData, devData, bytesCount, cudaMemcpyDeviceToHost);

  return devData;
}

/**
 * Implementation of Rastrigin function
 * */
bool Rastrigin::isWithinConstraint(const params_type& params) const {
  // Check whether every parameter is within the bounds
  for(double param : params)
    if (param < mLowerLimit || mUpperLimit < param)
      return false;

  return true;
}

double Rastrigin::evaluate(const params_type& params) const {
  // If the number is out of bounds, return max number (because it's minimization function
  if (!isWithinConstraint(params))
    return std::numeric_limits<double>::max();

  double res = mA * ((double)mDimensions);

  for (int i = 0; i < mDimensions; i++) {
    res += pow(params[i], 2) - mA * cos(2 * M_PI * params[i]);
  }

  return res;
}

thrust::host_vector<instance_type> Rastrigin::generateParamSet(size_t n) const {
  // Generate numbers in flat format
  RandomNumberGenerator generator;
  params_type paramsFlat = generator.generate(n * mDimensions, mLowerLimit, mUpperLimit);

  // Distribute the numbers in 2D vectors
  thrust::host_vector<instance_type> result(n);

  params_type *params;
  for(unsigned long i = 0; i < n; i++) {
    params = new params_type(paramsFlat.cbegin() + i * mDimensions, paramsFlat.cbegin() + (i + 1) * mDimensions);
    result[i] = instance_type(params, *this);
  }

  return result;
}

/**
 * Implementation of Differential Evolution
 * */
void DifferentialEvolution::initialize(const Rastrigin& function) {
  mInstances = function.generateParamSet(mInstances.size());
}

instance_type DifferentialEvolution::getMutation(const Rastrigin& function) const {
  /* Generate three distinct indices */
  size_t i1, i2, i3;

  i1 = getRandomIndex();

  do {
    i2 = getRandomIndex();
  } while(i1 == i2);

  do {
    i3 = getRandomIndex();
  } while(i1 == i3 || i2 == i3);

  // Copy the vector in i1
  auto* params = new params_type(*mInstances[i1].getParams());

  auto it1 = mInstances[i2].getParams()->cbegin(),
    it2 = mInstances[i3].getParams()->cbegin();
  // Perform v1 + F * (v2 - v3)
  int i = 0;
  for(auto it = params->begin(); it != params->end(); it++) {
    (*it) += mMutationParam * (*(it1 + i) - *(it2 + i));
    i++;
  }

  instance_type result(params, function);
  return result;
}

size_t DifferentialEvolution::getRandomIndex() const {
  return rand() % mPopulationSize;
}

bool DifferentialEvolution::shouldMutate() const {
  return (rand() / (double)RAND_MAX) < mCrossoverParam;
}

void DifferentialEvolution::performIteration(const Rastrigin& function) {
  // Prepare the set of mutated instances
  for (int i = 0; i < mPopulationSize; i++)
    mMutations[i] = getMutation(function);

  // Generate the index of random mutation
  size_t certainMutationIndex = getRandomIndex();

  if (mMutations[certainMutationIndex].getValue() < mInstances[certainMutationIndex].getValue())
    mInstances[certainMutationIndex] = mMutations[certainMutationIndex];

  thrust::device_vector<instance_type> mutationsTemp(mMutations), instancesTemp(mInstances);

  // Transform part of the mutations to the instances if:
  // 1) random value is less than threshold
  // 2) mutated value is bigger than initial one
  thrust::transform(
    mutationsTemp.begin(),
    mutationsTemp.end(),
    instancesTemp.begin(),
    instancesTemp.begin(),
    randomized_comparison(mCrossoverParam)
  );

  mInstances = thrust::host_vector<instance_type>(instancesTemp);
}

void DifferentialEvolution::determineBest(const Rastrigin& function) {
  size_t bestIndex = 0;

  for(int i = 1; i < mInstances.size(); i++) {
    if (mInstances[i] < mInstances[bestIndex]) {
      bestIndex = i;
    }
  }

  mBestIndex = bestIndex;
}

void DifferentialEvolution::optimize(Rastrigin function) {
  // Initialize the instances
  initialize(function);

  // Before hit the maximum iterations count
  // Optimize the function evaluation
  for(int i = 0; i < mIterations; i++) {
    performIteration(function);
  }

  mOptimized = true;
  // Get the best param in the set
  determineBest(function);
}

instance_type DifferentialEvolution::getBest() const {
  if (!mOptimized)
    throw std::runtime_error("Can't get the best value before optimization");

  return mInstances[mBestIndex];
}
