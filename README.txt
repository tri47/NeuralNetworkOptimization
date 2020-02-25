Steps to run the code:
1. Code link: https://github.com/tri47/NeuralNetworkOptimization
2. Set up the environment with the libraries in enviroment.yml file (using conda or pip)
The libraries include:
- python=3.7.2
- matplotlib=3.0.3
- numpy=1.16.2
- pandas=0.24.1
- mlrose==1.3.0

3. OPTIMIZATION PROBLEMS:
Run the program to produce the graphs for 3 optimization problems:
For problem 1- Four-Peak:
    python optimization_problems.py  --problem=1
For problem 2- Count 1 with all-or-nothing leading 1's:
    python optimization_problems.py  --problem=2
For problem 3- Snapsack:
    python optimization_problems.py  --problem=3

4. LEARNING WEIGHTS OF NEURAL NETWORK
Run the program to produce the graphs for 3 algorithms:
For Random Hill CLimbing:
    python NN_weight_optimization.py  --alg="rhc"
For Simulated Annealing:
    python NN_weight_optimization.py  --alg="sa"
For Genetic Algorithms:
    python NN_weight_optimization.py  --alg="ga"

The data set is already included. It was obtained from:
http://archive.ics.uci.edu/ml/datasets/wilt

Reference:
1.	Johnson, B., Tateishi, R., Hoan, N., 2013. A hybrid pansharpening approach and multiscale object-based image analysis for mapping diseased pine and oak trees. International Journal of Remote Sensing, 34 (20), 6969-6982.
