#include <iostream>
#include <vector>
#include<random>

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

    bool isWithinConstraint(double value) const;
    double evaluate(const instance_type& params) const;
    instance_type generateRandomParam() const;
};

class DifferentialEvolution {
protected:
    size_t mIterations, mPopulationSize;
    bool mOptimized;
    list_type mInstances;
    size_t mBestIndex;
    double mMutationParam, mCrossoverParam;

    void initialize(const Rastrigin& function);
    instance_type getMutation() const;
    void performIteration(const Rastrigin& function);
    void determineBest(const Rastrigin& function);
    size_t getRandomIndex() const;
public:
    DifferentialEvolution(size_t iterations, size_t populationSize, double mutationParam, double crossoverParam)
    : mIterations(iterations), mOptimized(false), mInstances(list_type()), mBestIndex(0),
      mPopulationSize(populationSize), mMutationParam(mutationParam), mCrossoverParam(crossoverParam) {}

    void optimize(Rastrigin function);
    instance_type getBest() const;
};

int main() {
  Rastrigin func(3, -5.12, 5.12, 10);
  DifferentialEvolution optimizer(20, 200, 0.5, 0.5);

  optimizer.optimize(func);

  std::cout << "Hello, World!" << std::endl;
  return 0;
}

/**
 * Implementation of Rastrigin function
 * */
bool Rastrigin::isWithinConstraint(double value) const {}
double Rastrigin::evaluate(const instance_type& params) const {}
instance_type Rastrigin::generateRandomParam() const {}

/**
 * Implementation of Differential Evolution
 * */
void DifferentialEvolution::initialize(const Rastrigin& function) {
  mInstances = list_type(mPopulationSize);

  for(int i = 0; i < mPopulationSize; i++) {
    mInstances[i] = function.generateRandomParam();
  }
}

instance_type DifferentialEvolution::getMutation() const {
  /* TODO: Implement the method */
}

size_t DifferentialEvolution::getRandomIndex() const {
  /* TODO: Implement the method */
}

void DifferentialEvolution::performIteration(const Rastrigin& function) {
  /* TODO: Implement the method */
  // Prepare the set of mutated instances

  // Generate the index of random mutation

  // For each element: if index equals to the one above or random number is below crossoverParam
  // Try replacing it by new entry
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
  /* TODO: Remove i < 1 */
  for(int i = 0; i < mIterations && i < 1; i++) {
    performIteration(function);
  }

  // Get the best param in the set
  determineBest(function);
}

instance_type DifferentialEvolution::getBest() const {
  if (!mOptimized)
    throw std::runtime_error("Can't get the best value before optimization");

  return mInstances[mBestIndex];
}
