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
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

#define BLOCK_SIZE 128
#define BLOCKS_COUNT(n) ((size_t)ceil(((float)n)/(float)BLOCK_SIZE))

/**
 * Main parameters
 * */
const size_t FUNC_DIMENSIONS = 10;
const double LOWER_BOUND = -5.12, UPPER_BOUND = 5.12;
const double FUNC_COEF_A = 10;
const size_t ITERATIONS_COUNT = 100, POPULATION_SIZE = 2000;
const double ALGORITHM_MUTATION_PARAM = 0.25, ALGORITHM_CROSSOVER_PARAM = 0.75;

/**
 * Class declarations
 * */
class Rastrigin;

class DEInstance;

class DifferentialEvolution;

/**
 * Type declarations
 * */
// We use aliases for names to easily manage and give them more descriptive names
typedef std::vector<double> params_type;
typedef DEInstance instance_type;

/**
 * Kernels
 * */

/**
 * Kernel for transforming numbers from uniform ([0;1]) format to range [lower;upper]
 * */
__global__ void transformNumber(double *src, int size, double lower, double upper) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    src[idx] = src[idx] * (upper - lower) + lower;
  }
}

/**
 * Wrapper class for CuRAND library. It's used only for generating a vector of double numbers in [lowerBound;upperBound] range
 * */
class RandomNumberGenerator {
protected:
    curandGenerator_t generator;

    double *_generateNumbers(size_t n);

public:
    RandomNumberGenerator() {
      curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937);
      curandSetPseudoRandomGeneratorSeed(generator, time(nullptr));
    }

    ~RandomNumberGenerator() {
      curandDestroyGenerator(generator);
    }

    params_type generate(size_t n, double lowerBound, double upperBound);
};

/**
 * Class that implements Rastrigin function
 * It includes: calculating value by vector of parameters, checking whether the value is inside the boundaries,
 * and generating a vector of algorithm instances with randomized parameters
 * */
class Rastrigin {
protected:
    size_t mDimensions;
    double mLowerLimit;
    double mUpperLimit;
    double mA;

public:
    struct evaluate_function {
        double mA;

        explicit evaluate_function(double A) : mA(A) {}

        __host__ __device__
        double operator()(double x) const {
          return x * x - mA * cos(2 * M_PI * x);
        }
    };

    Rastrigin(size_t dimensions, double lowerLimit, double upperLimit, double A)
        : mDimensions(dimensions), mLowerLimit(lowerLimit), mUpperLimit(upperLimit), mA(A) {}

    bool isWithinConstraint(const params_type &params) const;

    double evaluate(const params_type &params) const;

    thrust::host_vector<instance_type> generateParamSet(size_t n) const;
};

/**
 * Class that represents one set of parameters for Differential Evolution algorithm
 * It contains pointer to the parameters (needed to avoid copying vector when we pass it to device) and the evaluated value
 * */
class DEInstance {
protected:
    params_type *mParams;
    double mValue;

public:
    // Constructor for creating empty object
    __host__ __device__
    DEInstance() : mParams(nullptr), mValue(-1) {}

    // Constructor for initializing the class with parameters and function value
    __host__ __device__
    DEInstance(params_type *params, const Rastrigin &func) : mParams(params), mValue(func.evaluate(*params)) {}

    // Constructor for shallow copying from another object. It's used mostly when objects are passed frequently
    __host__ __device__
    DEInstance(const DEInstance &other) : mParams(other.mParams), mValue(other.mValue) {}

    __host__ __device__
    bool isEmpty() const {
      return mParams == nullptr;
    }

    // Device shouldn't have the access to parameters that are located in host's memory
    const params_type *getParams() const {
      return mParams;
    }

    __host__ __device__
    double getValue() const {
      return mValue;
    }

    __host__ __device__
    bool operator>(const DEInstance &other) const { return mValue > other.mValue; }

    __host__ __device__
    bool operator>=(const DEInstance &other) const { return mValue >= other.mValue; }

    __host__ __device__
    bool operator<(const DEInstance &other) const { return mValue < other.mValue; }

    __host__ __device__
    bool operator<=(const DEInstance &other) const { return mValue <= other.mValue; }
};

/**
 * Class that implements Differential Evolution algorithm
 * */
class DifferentialEvolution {
protected:
    // Number of required iterations and population size
    size_t mIterations, mPopulationSize;

    // Flag for determining whether the function is already optimized
    bool mOptimized;

    // Vectors for storing current population and potential population (mutated instances)
    thrust::host_vector<instance_type> mInstances, mMutations;

    // The best found instance, it's empty by default
    instance_type mBestValue;

    // The real value that determine: mMutationParam - mutation coefficient, mCrossoverParam - chances of replacing in crossover
    double mMutationParam, mCrossoverParam;

    // Method for initializing the population
    void initialize(const Rastrigin &function);

    // Returns the single mutation instance
    instance_type getMutation(const Rastrigin &function) const;

    // Performs everything that needs to be done in single iteration
    void performIteration(const Rastrigin &function);

    // Find and saves the best found value
    void determineBest();

    // Generates the random index within the current iteration
    size_t getRandomIndex() const;

public:
    /**
     * Functor for replacing mutations with better evaluated values with the randomization factor
     * Basically, combines two operations:
     * 1) Chooses the current value over mutated if random value is greater than threshold
     * 2) Replaces the current value with mutated if the latter's performance is better
     * */
    struct randomized_comparison {
    protected:
        double mThreshold;
        thrust::minstd_rand rng;
        thrust::uniform_real_distribution<double> dist;

    public:
        explicit randomized_comparison(double threshold) : mThreshold(threshold), rng(),
                                                           dist(thrust::uniform_real_distribution<double>(0, 1)) {}

        __host__ __device__
        DEInstance operator()(const DEInstance &mutated, const DEInstance &original) {
          if (dist(rng) >= mThreshold)
            return original;

          return mutated.getValue() < original.getValue() ? mutated : original;
        }
    };

    /**
     * Functor for determining the best value. It should be used with reduce operation
     * */
    struct determine_best {
        __host__ __device__
        instance_type operator()(const DEInstance &best, const instance_type &current) {
          if (best.isEmpty())
            return current;

          return current.getValue() < best.getValue() ? current : best;
        }
    };

    /**
     * I know it's a lot of parameters, but all of them should be initialized
     * */
    DifferentialEvolution(size_t iterations, size_t populationSize, double mutationParam, double crossoverParam)
        : mIterations(iterations), mOptimized(false), mInstances(thrust::host_vector<instance_type>(populationSize)),
          mMutations(thrust::host_vector<instance_type>(populationSize)), mBestValue(), mPopulationSize(populationSize),
          mMutationParam(mutationParam), mCrossoverParam(crossoverParam) {}

    /**
     * The main public method - performs optimization and returns the best found value
     * */
    instance_type optimize(Rastrigin function);

    // Returns the best found value if object is optimized, otherwise throws an error
    instance_type getBest() const;
};

void printParams(const params_type &params) {
  for (double param: params)
    std::cout << param << " ";
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

  Rastrigin func(FUNC_DIMENSIONS, LOWER_BOUND, UPPER_BOUND, FUNC_COEF_A);

  DifferentialEvolution optimizer(
      ITERATIONS_COUNT,
      POPULATION_SIZE,
      ALGORITHM_MUTATION_PARAM,
      ALGORITHM_CROSSOVER_PARAM
  );

  instance_type bestSolution = optimizer.optimize(func);

  std::cout << "The best solution: " << bestSolution.getValue() << std::endl;
  printParams(*bestSolution.getParams());

  return 0;
}

/**
 * Implementation of RandomNumberGenerator
 * */
params_type RandomNumberGenerator::generate(size_t n, double lowerBound, double upperBound) {
  // Generate the numbers in uniform format
  // Numbers will be on the device
  double *numbersDev = _generateNumbers(n);

  // Transform numbers from [0:1] to [lowerBound:upperBound]
  transformNumber<<<BLOCKS_COUNT(n), BLOCK_SIZE>>>(numbersDev, n, lowerBound, upperBound);

  // Copy from device to host and turn into more comfortable format
  auto *numbersHost = new double[n];
  cudaMemcpy(numbersHost, numbersDev, sizeof(double) * n, cudaMemcpyDeviceToHost);

  // Initialize the result variable in vector format
  params_type result(numbersHost, numbersHost + n);

  // Free data and return vector
  cudaFree(numbersDev);
  free(numbersHost);
  return result;
}

double *RandomNumberGenerator::_generateNumbers(size_t n) {
  // Allocate host memory
  size_t bytesCount = n * sizeof(double);
  double *hostData, *devData;
  hostData = new double[n];

  // Allocate device memory
  cudaMalloc((void **) &devData, bytesCount);

  // Generate numbers in [0;1] range and copy to the host
  curandGenerateUniformDouble(generator, devData, n);

  cudaMemcpy(hostData, devData, bytesCount, cudaMemcpyDeviceToHost);

  return devData;
}

/**
 * Implementation of Rastrigin function
 * */
bool Rastrigin::isWithinConstraint(const params_type &params) const {
  // Check whether every parameter is within the bounds
  for (double param: params)
    if (param < mLowerLimit || mUpperLimit < param)
      return false;

  return true;
}

double Rastrigin::evaluate(const params_type &params) const {
  // If the number is out of bounds, return max number (because it's minimization function)
  if (!isWithinConstraint(params))
    return std::numeric_limits<double>::max();

  // Use thrust::transform_reduce to map x => x^2 - A * cos(2 * pi * x), then find the sum
  // The initial value is A * n, so the result is
  // A * n + (params | map((x) => x^2 - A * cos(2 * pi *x)) | reduce(+))
  double res = mA * ((double) mDimensions);
  thrust::device_vector<double> devParams(params.begin(), params.end());

  return thrust::transform_reduce(
      devParams.begin(),
      devParams.end(),
      evaluate_function(mA),
      res,
      thrust::plus<double>()
  );
}

thrust::host_vector<instance_type> Rastrigin::generateParamSet(size_t n) const {
  // Generate numbers in flat format
  RandomNumberGenerator generator;
  params_type paramsFlat = generator.generate(n * mDimensions, mLowerLimit, mUpperLimit);

  // Distribute the numbers in 2D vectors
  thrust::host_vector<instance_type> result(n);

  params_type *params;
  // This part can't be parallelized, because it requires allocating host memory. The calculating of function is based
  // on the value of parameters that will be inaccessible from device or too expensive to copy every set of parameters
  // For each instance, extract generated parameters and create new instance of DEInstance
  for (unsigned long i = 0; i < n; i++) {
    params = new params_type(paramsFlat.cbegin() + i * mDimensions, paramsFlat.cbegin() + (i + 1) * mDimensions);
    result[i] = instance_type(params, *this);
  }

  return result;
}

/**
 * Implementation of Differential Evolution
 * */
void DifferentialEvolution::initialize(const Rastrigin &function) {
  // Delegate the generation of set of params to the Rastrigin function
  mInstances = function.generateParamSet(mInstances.size());
}

instance_type DifferentialEvolution::getMutation(const Rastrigin &function) const {
  /* Generate three distinct indices */
  // This method will cause infinite if population size less than 3
  size_t i1, i2, i3;

  i1 = getRandomIndex();

  // Generate the index until i1 and i2 are distinct
  do {
    i2 = getRandomIndex();
  } while (i1 == i2);

  // Generate the index until i1, i2, and i3 are distinct
  do {
    i3 = getRandomIndex();
  } while (i1 == i3 || i2 == i3);

  // Copy the vector in i1
  auto *params = new params_type(*mInstances[i1].getParams());

  auto it1 = mInstances[i2].getParams()->cbegin(),
      it2 = mInstances[i3].getParams()->cbegin();

  // Perform v1 + F * (v2 - v3)
  int i = 0;
  for (auto it = params->begin(); it != params->end(); it++) {
    (*it) += mMutationParam * (*(it1 + i) - *(it2 + i));
    i++;
  }

  instance_type result(params, function);
  return result;
}

size_t DifferentialEvolution::getRandomIndex() const {
  return rand() % mPopulationSize;
}

void DifferentialEvolution::performIteration(const Rastrigin &function) {
  // Prepare the set of mutated instances
  for (int i = 0; i < mPopulationSize; i++)
    mMutations[i] = getMutation(function);

  // Generate the index for certain mutation. For this index, check the condition for replacing
  size_t certainMutationIndex = getRandomIndex();

  if (mMutations[certainMutationIndex].getValue() < mInstances[certainMutationIndex].getValue())
    mInstances[certainMutationIndex] = mMutations[certainMutationIndex];

  // Copy instances to the device memory
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

void DifferentialEvolution::determineBest() {
  // Delegate finding the best values to the specific functor
  thrust::device_vector<instance_type> devInstances(mInstances.begin(), mInstances.end());

  mBestValue = thrust::reduce(
      devInstances.begin(),
      devInstances.end(),
      instance_type(),
      determine_best()
  );
}

instance_type DifferentialEvolution::optimize(Rastrigin function) {
  // Initialize the instances
  initialize(function);

  // Before hit the maximum iterations count
  // Optimize the function evaluation
  for (int i = 0; i < mIterations; i++) {
    performIteration(function);
  }

  mOptimized = true;

  // Get the best param in the set
  determineBest();

  return getBest();
}

instance_type DifferentialEvolution::getBest() const {
  if (!mOptimized)
    throw std::runtime_error("Can't get the best value before optimization");

  return mBestValue;
}
