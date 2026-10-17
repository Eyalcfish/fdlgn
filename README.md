# Fully discrete logic gate network(FDLGN)

**Abstract**
Standard approaches to training logic gate networks rely on continuous relaxations of discrete truth tables to enable gradient-based optimization. This document outlines the theoretical framework for **Fully Discrete Logic Gate Networks (FDLGN)**, a purely discrete learning paradigm that eliminates floating-point relaxations and gradient descent. FDLGN is grounded in the equivalence between data compression and statistical generalization, framing the machine learning problem as the systematic resolution of Boolean "don't care" states. By first constructing a large, unoptimized Lookup Table (LUT) circuit that exactly memorizes half of a training dataset, and subsequently applying a portfolio of local Boolean logic optimization algorithms evaluated against the remaining half of the dataset, FDLGN seeks to transform an over-parameterized memorization structure into a generalized logic circuit using only discrete operations and integer error summation.

**1. Theoretical Intuition**
The conceptual foundation of FDLGN rests on three core observations connecting digital logic synthesis and machine learning:

**1.1 The Duality of Compression and Machine Learning**
In information theory and statistical learning, finding a model that generalizes across unseen data is fundamentally linked to compression. A circuit that memorizes a dataset by allocating isolated hardware logic to every individual data point possesses maximum complexity and zero generalization capability. Conversely, compressing a Boolean structure by identifying shared logical patterns, removing redundancies, and simplifying gate structures forces the circuit to capture underlying structural rules rather than isolated instances. Under this view, Boolean logic optimization acts directly as a learning mechanism.

**1.2 The Transition from "Don't Cares" to "Yes Cares"**
For any finite training dataset over a binary input space, the observed data points represent only a small fraction of all possible input combinations.

• **At Training Time:** All unobserved input states act as Boolean **"don't cares"** states for which the training set imposes no explicit output constraint. A naive circuit can assign arbitrary outputs to these states while still achieving 100% accuracy on the observed training subset.

• **At Deployment Time:** When the circuit is evaluated on unseen real-world data, those previously unobserved states are actively encountered. Consequently, training-time **"don't cares" must become deployment-time "yes cares"** states that require accurate, generalized outputs.

**1.3 Resolving the "Don't Care" Gap via Logic Optimization**
In Boolean logic synthesis, optimization algorithms simplify Lookup Table (LUT) structures precisely by exploiting "don't care" states, assigning specific `0` or `1` values to unconstrained truth-table entries so that neighboring logic gates can be merged, simplified, or eliminated. Different optimization algorithms resolve these "don't care" entries in different ways. FDLGN leverages a held-out portion of the dataset to guide which compression heuristic resolves the "don't care" gap most effectively for unseen inputs.

**2. The FDLGN Two-Stage Methodology**
FDLGN divides the training process into two distinct, purely discrete phases over a dataset partitioned into two halves: **Subset A** (the construction half) and **Subset B** (the optimization validation half).

**2.1 Stage 1: Exact Subset Simulation ("Dumb" Structure Construction)**
In the first stage, an expansive, unoptimized network of LUTs is constructed to simulate **Subset A** with 100% accuracy.
• Rather than initializing a network with random wiring and random truth tables, Stage 1 directly synthesizes a large, explicit Boolean structure guaranteed to output the exact target values for every sample in Subset A.
• While this initial circuit is structurally bloated ("dumb") and treats all inputs outside Subset A as unconstrained "don't cares," it establishes functional, data-driven wiring across the input space from the outset of training.

**2.2 Stage 2: Local Sub-Circuit Optimization ("Smart" Compression)**
Once the initial circuit achieves 100% equivalence on Subset A, training proceeds to an iterative optimization phase utilizing a portfolio of distinct Boolean logic compression algorithms:

1. **Section Isolation:** The overarching LUT network is partitioned into small, localized sections of gates.

2. **Candidate Compression Generation:** For a given small section of gates, each optimization algorithm in the portfolio is executed separately on that local structure. Each algorithm attempts to compress and simplify the section, implicitly filling the local "don't care" states according to its specific reduction rules.

3. **Validation Evaluation:** Each candidate compression (alongside the baseline option of applying **no optimization** to that section) is evaluated against **Subset B** (the remaining half of the dataset).

4. **Greedy Selection:** The candidate transformation that yields the lowest loss on Subset B is selected and permanently applied to that section of the network. If all compression candidates degrade performance on Subset B, the unoptimized section is retained.

5. **Epoch Progression:** Sequentially applying this local evaluation-and-selection procedure across every small section of the network constitutes **one training epoch**. Multiple epochs can be executed as local simplifications propagate across the circuit structure.

**2.3 Loss Function**
FDLGN requires no continuous surrogate loss (such as cross-entropy or soft margin relaxations). The loss function evaluated during Stage 2 is strictly the discrete count of samples in the evaluation set that fail to achieve full equivalence with the ground-truth target output

**3. Hypothesized Architectural Properties**
Based on its formulation, the FDLGN framework is designed to offer two structural characteristics that contrast with continuous Differentiable Logic Gate Networks:

• **Informed Training-Time Wiring:** Because Stage 1 constructs the circuit directly from Subset A rather than relying on fixed random connectivity, the topology begins with functional routing pathways that Stage 2 refines and compacts.

• **Purely Discrete Computation:** Both stages operate strictly within the domain of Boolean algebra. Forward evaluation uses native LUT bitwise lookups, and loss evaluation requires only a simple integer summation of non-equivalent outputs, completely avoiding floating-point arithmetic, probability distributions, and gradient backpropagation.

**4. Scope and Status**
This document serves exclusively as a theoretical specification of the FDLGN architecture and its underlying intuition. Empirical validation, selection of the specific local LUT optimization portfolio, and comparative benchmarking against continuous differentiable baselines remain subjects for separate experimental investigation.
