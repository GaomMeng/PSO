# Particle Swarm Optimization (PSO) Implementation

## Overview

This code implements a Particle Swarm Optimization algorithm for function optimization. Key features include:

1. A flexible `Particle` class to represent individual particles in the swarm
2. A `PSO` class that manages the optimization process
3. Customizable parameters for fine-tuning the algorithm
4. A sample test function to demonstrate usage

## Theoretical Background

Particle Swarm Optimization is a population-based optimization technique inspired by bird flocking behavior. It consists of:

- A swarm of particles, each representing a potential solution
- Particle movement influenced by local and global best known positions
- Iterative improvement of the swarm to find the optimal solution

## Implementation Details

### Particle Class

The `Particle` class represents individual particles in the swarm. It includes:

- Position and velocity vectors
- Personal best position
- Fitness value calculation

### PSO Class

The `PSO` class manages the optimization process. Key methods:

- `update_vel`: Updates particle velocities
- `update_pos`: Updates particle positions and evaluates fitness
- `update`: Runs the PSO algorithm and returns results

### Key Parameters

- `x_dim`: Dimension of the problem space
- `x_bound`: Boundary of the search space
- `Population_size`: Number of particles in the swarm
- `Iteration_number`: Number of iterations to run
- `Inertia_weight`: Inertia weight for velocity update
- `Individual_learning_factor`: Cognitive component weight
- `Social_learning_factor`: Social component weight
- `Max_vel`: Maximum velocity allowed for particles

## Usage

1. Define your optimization problem by creating a class with a `test` method:

```python
class YourProblem:
    def __init__(self):
        pass

    def test(self, config):
        x = config['x']
        # Your objective function here
        y = your_objective_function(x)
        return y
```

2. Set up the parameters:

```python
args = set_param()  # Use the provided set_param function or create your own
```

3. Initialize and run the PSO:

```python
your_problem = YourProblem()
pso_config = {'name': 'your_problem_name', 'plm': your_problem}
pso = PSO(args, pso_config)
pso_dict = pso.update()
```

4. Access the results:

```python
best_fitness = pso_dict['best_f']
best_position = pso_dict['best_x']
```

## Example

The code includes a test case optimizing the function `y = 100 - x * x`. Run the script to see it in action:

```
python pso_test.py
```

## Customization

- Modify the `set_param` function to change default parameters
- Create your own problem class to optimize different functions
- Adjust the PSO class for advanced variants of the algorithm

## Notes

- The code uses NumPy for efficient numerical computations
- Random seed is not fixed, allowing for different runs to produce varied results
- The implementation includes bounds checking to keep particles within the specified search space

## References

- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. Proceedings of ICNN'95 - International Conference on Neural Networks.
- Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. 1998 IEEE International Conference on Evolutionary Computation Proceedings.
