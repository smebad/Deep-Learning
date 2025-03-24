# Deep Learning Journey - Day 15: Weights Initialization

## What is Weight Initialization?

Weight initialization is the process of setting the initial values of the weights in a neural network before training begins. The choice of initial weights can significantly impact the convergence speed and overall performance of the model. Poor initialization can lead to problems such as vanishing or exploding gradients, causing training to stall or fail.

## Why is Weight Initialization Important?
- **Avoiding Vanishing/Exploding Gradients:** Poor initialization can cause gradients to shrink or explode during backpropagation, making learning unstable.
- **Speeding Up Convergence:** Proper initialization helps the model reach optimal weights faster.
- **Preventing Dead Neurons:** If weights are set poorly, some neurons may stop contributing to learning.
- **Ensuring Symmetry Breaking:** If all weights are initialized to the same value, neurons learn the same features, reducing the network's capacity.

## Common Weight Initialization Techniques
1. **Zero Initialization:** All weights are set to zero. This is a poor choice because all neurons in a layer learn the same features.
2. **Random Initialization:** Weights are assigned small random values to break symmetry.
3. **Xavier (Glorot) Initialization:** Designed for sigmoid and tanh activations, Xavier initialization ensures the variance of activations remains stable across layers.
4. **He Initialization:** Optimized for ReLU and its variants, He initialization prevents the variance from growing too large as it propagates through the network.

## Experiments in This Notebook

### 1. Using ReLU Activation with Zero Initialization
#### Steps:
- Loaded the **U-shape dataset**.
- Created a neural network with:
  - An input layer with **10 neurons** and ReLU activation.
  - An output layer with **1 neuron** and sigmoid activation.
- **Set all initial weights to zero**.
- Compiled and trained the model using **Adam optimizer**.
- **Plotted decision boundaries** to visualize model performance.

#### Observations:
- The network failed to learn effectively because **all neurons in a layer updated identically**, leading to **symmetry** and poor performance.

### 2. Using Sigmoid Activation with Zero Initialization
#### Steps:
- Used the same dataset and network structure, but replaced **ReLU with sigmoid** in the hidden layer.
- **Initialized all weights to zero**.
- Trained the model and plotted decision boundaries.

#### Observations:
- With sigmoid activation, the model still failed to learn properly.
- The vanishing gradient problem was evident, as updates were very small.
- The decision boundary was poor, showing that the model was unable to generalize.

## Comparing ReLU vs. Sigmoid with Zero Initialization
| Activation Function | Learning Performance | Issues |
|---------------------|---------------------|--------|
| ReLU               | Failed to break symmetry | All neurons updated identically |
| Sigmoid            | Suffered from vanishing gradients | Small updates, slow learning |

### Key Takeaways:
- **Zero initialization is a bad practice**, as it leads to symmetry in neurons and poor learning.
- **ReLU performs better with proper initialization methods like He Initialization**.
- **Sigmoid activation is sensitive to weight initialization**, often suffering from vanishing gradients.
- Using **randomized initialization** techniques like **Xavier or He initialization** can significantly improve training efficiency.

## Next Steps:
In the next notebook of Day 16, I will explore **better weight initialization techniques (e.g., Xavier and He) and their impact on deep learning performance**.