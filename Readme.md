# Programming 3. Differential Evolution Algorithm
### M1 DSAI, Elvin Bayramov

The description of the project can be found in 'Description.pdf'.

The implementation of DE in C++ consists of implementing classes for differential evolution (`DifferentialEvolution`, `DEInstance`), Rastrigin function (`Rastrigin`), and random number generator (`RandomNumberGenerator`).

The optimization process consists of the following steps:
1. Initialize the population
2. For n iterations:
   1. Generate set of mutated instances
   2. Generate an index and replace the element with corresponding mutated instance if the latter has better performance.
   3. Compare each element with mutated one (with some probability) and if the mutated instance has better performance, replace the original instance by it.
3. Get the best instance in the current population

The main goal of the project is to apply knowledge about parallalization and use tools to parallelize different parts of the application. Here we use: `CUDA` (in form of memory allocation and kernel calls), `CuRAND` for generation random numbers, `thrust::reduce`, `thrust::transform`, `thrust::transform_reduce` for comparing values, combining two vectors, and calculating evaluation function correspondingly.

The following parts are parallelized:
• Rastrigin function evaluation: uses `thrust::transform_reduce` with custom functor (`Rastrigin::evaluate_function`) for calculating $x - A cos(2 \pi x)$ and thrust::plus to parallelize calculation of $A n + sum_i^n (x_i - A sin(2 \pi x_i))$
• Generating initial population (step 1). For population with size n and dimensions m, use `CuRAND` to generate $n \times m$ numbers in range [0;1] and transformNumber kernel to transform into [min;max] range.
• Comparing and replacing (step 2.3).  Use `thrust::transform` and custom functor (`DifferentialEvolution::randomized_comparison`) to compare and replace the instance in parent population. It also uses  `uniform_real_distribution` with `minstd_rand` to randomize the comparison (it skips the comparison with certain probability).
• Getting the best instance (step 3). Uses `thrust::reduce` and custom functor (`DifferentialEvolution::determine_best`) to compare and store the best found set of parameters.
