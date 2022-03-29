#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <limits>

typedef std::vector<double> instance_type;
typedef std::vector<instance_type> list_type;

class Rastrigin {
protected:
    size_t mDimensions;
    double mLowerLimit;
    double mUpperLimit;
    double mA;

public:
    Rastrigin(size_t dimensions, double lowerLimit, double upperLimit, double A)
        : mDimensions(dimensions), mLowerLimit(lowerLimit), mUpperLimit(upperLimit), mA(A) {}

    bool isWithinConstraint(const instance_type& params) const;
    double evaluate(const instance_type& params) const;
    instance_type generateRandomParam() const;
};

class DifferentialEvolution {
protected:
    size_t mIterations, mPopulationSize;
    bool mOptimized;
    list_type mInstances, mMutations;
    size_t mBestIndex;
    double mMutationParam, mCrossoverParam;

    void initialize(const Rastrigin& function);
    instance_type getMutation() const;
    void performIteration(const Rastrigin& function);
    void determineBest(const Rastrigin& function);
    size_t getRandomIndex() const;
    bool shouldMutate() const;
public:
    DifferentialEvolution(size_t iterations, size_t populationSize, double mutationParam, double crossoverParam)
    : mIterations(iterations), mOptimized(false), mInstances(list_type(populationSize)), mMutations(list_type(populationSize)),
      mBestIndex(0), mPopulationSize(populationSize), mMutationParam(mutationParam), mCrossoverParam(crossoverParam) {}

    void optimize(Rastrigin function);
    instance_type getBest() const;
};

void display(const instance_type& params) {
  for (auto it = params.cbegin(); it != params.cend(); it++)
    std::cout << *it << " ";
  std::cout << std::endl;
}

int main() {
  Rastrigin func(3, -5.12, 5.12, 10);
  DifferentialEvolution optimizer(100, 2000, 0.25, 0.75);

  optimizer.optimize(func);

  instance_type bestSolution = optimizer.getBest();

  std::cout << "The best solution: " << func.evaluate(bestSolution) << std::endl;
  display(bestSolution);

  return 0;
}

/**
 * Implementation of Rastrigin function
 * */
bool Rastrigin::isWithinConstraint(const instance_type& params) const {
  // Check whether every parameter is within the bounds
  for(double param : params)
    if (param < mLowerLimit || mUpperLimit < param)
      return false;

  return true;
}

double Rastrigin::evaluate(const instance_type& params) const {
  // If the number is out of bounds, return max number (because it's minimization function
  if (!isWithinConstraint(params))
    return std::numeric_limits<double>::max();

  double res = mA * ((double)mDimensions);

  for (int i = 0; i < mDimensions; i++) {
    res += pow(params[i], 2) - mA * cos(2 * M_PI * params[i]);
  }

  return res;
}

instance_type Rastrigin::generateRandomParam() const {
  instance_type result(mDimensions);

  for(int i = 0; i < mDimensions; i++) {
    result[i] = mLowerLimit + ((double)rand()) / RAND_MAX * (mUpperLimit - mLowerLimit);
  }

  return result;
}

/**
 * Implementation of Differential Evolution
 * */
void DifferentialEvolution::initialize(const Rastrigin& function) {
  for(int i = 0; i < mPopulationSize; i++) {
    mInstances[i] = function.generateRandomParam();
  }
}

instance_type DifferentialEvolution::getMutation() const {
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
  instance_type temp = mInstances[i1];

  auto it1 = mInstances[i2].cbegin(),
    it2 = mInstances[i3].cbegin();
  // Perform v1 + F * (v2 - v3)
  for(int i = 0; i < temp.size(); i++) {
    temp[i] += mMutationParam * (*(it1 + i) - *(it2 + i));
  }

  return temp;
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
    mMutations[i] = getMutation();

  // Generate the index of random mutation
  size_t certainMutationIndex = getRandomIndex();

  // For each element: if index equals to the one above or random number is below crossoverParam
  // Try replacing it by new entry
  for (int i = 0; i < mPopulationSize; i++) {
    if (i == certainMutationIndex || shouldMutate()) {
      // If the mutated one shows better results, then replace
      if (function.evaluate(mInstances[i]) > function.evaluate(mMutations[i])) {
        mInstances[i] = mMutations[i];
      }
    }
  }
}

void DifferentialEvolution::determineBest(const Rastrigin& function) {
  size_t bestIndex = 0;
  double bestValue = function.evaluate(mInstances[0]);

  double currentValue;
  for(int i = 1; i < mInstances.size(); i++) {
    currentValue = function.evaluate(mInstances[i]);
    if (currentValue < bestValue) {
      bestValue = currentValue;
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
