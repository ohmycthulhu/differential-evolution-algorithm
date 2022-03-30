#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <limits>

class Rastrigin;
class DEInstance;

typedef std::vector<double> params_type;
typedef DEInstance instance_type;
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

    bool isWithinConstraint(const params_type& params) const;
    double evaluate(const params_type& params) const;
    instance_type generateRandomParam() const;
};


class DEInstance {
protected:
    params_type mParams;
    double mValue;
public:
    DEInstance() : mParams(), mValue() {}
    DEInstance(const params_type& params, const Rastrigin& func) : mParams(params), mValue(func.evaluate(params)) {}
    DEInstance(const DEInstance& other) : mValue(other.mValue) {
      mParams = other.mParams;
    }

    const params_type* getParams() const{
      return &mParams;
    }

    double getValue() const {
      return mValue;
    }

    bool operator>(const DEInstance& other) const { return mValue > other.mValue; }
    bool operator>=(const DEInstance& other) const { return mValue >= other.mValue; }
    bool operator<(const DEInstance& other) const { return mValue < other.mValue; }
    bool operator<=(const DEInstance& other) const { return mValue <= other.mValue; }
};

class DifferentialEvolution {
protected:
    size_t mIterations, mPopulationSize;
    bool mOptimized;
    list_type mInstances, mMutations;
    size_t mBestIndex;
    double mMutationParam, mCrossoverParam;

    void initialize(const Rastrigin& function);
    instance_type getMutation(const Rastrigin& function) const;
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

void display(const params_type& params) {
  for (auto it = params.cbegin(); it != params.cend(); it++)
    std::cout << *it << " ";
  std::cout << std::endl;
}

int main() {
  Rastrigin func(3, -5.12, 5.12, 10);
  DifferentialEvolution optimizer(100, 2000, 0.25, 0.75);

  optimizer.optimize(func);

  instance_type bestSolution = optimizer.getBest();

  std::cout << "The best solution: " << bestSolution.getValue() << std::endl;
  display(*bestSolution.getParams());

  return 0;
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

instance_type Rastrigin::generateRandomParam() const {
  params_type params(mDimensions);

  for(int i = 0; i < mDimensions; i++) {
    params[i] = mLowerLimit + ((double)rand()) / RAND_MAX * (mUpperLimit - mLowerLimit);
  }

  instance_type result(params, *this);
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
  params_type params = params_type(*mInstances[i1].getParams());

  auto it1 = mInstances[i2].getParams()->cbegin(),
    it2 = mInstances[i3].getParams()->cbegin();
  // Perform v1 + F * (v2 - v3)
  for(int i = 0; i < params.size(); i++) {
    params[i] += mMutationParam * (*(it1 + i) - *(it2 + i));
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

  // For each element: if index equals to the one above or random number is below crossoverParam
  // Try replacing it by new entry
  for (int i = 0; i < mPopulationSize; i++) {
    if (i == certainMutationIndex || shouldMutate()) {
      // If the mutated one shows better results, then replace
      if (mMutations[i] < mInstances[i]) {
        mInstances[i] = mMutations[i];
      }
    }
  }
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
