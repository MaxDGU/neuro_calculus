# Neuro Change Tracker: A Calculus of Learning

## 1. Project Vision
   - Build an open-source, generally deployable system to track changes in neural network weights during training.
   - Accommodate diverse neural network architectures (MLPs, CNNs, Transformers, etc.).
   - Support various training regimes (supervised, unsupervised, meta-learning, RL, etc.).
   - Remain agnostic to specific loss functions and tasks.
   - Aim for modular, efficient, and clean code.

## 2. Core Functionality
   - **Weight Tracking:**
     - Mechanism to capture and store weight snapshots at different training stages (e.g., per epoch, per N steps).
     - Efficient storage and retrieval of weight data.
   - **Change Computation:**
     - Methods to calculate differences between weight snapshots (e.g., L1/L2 norm, cosine similarity/distance between weight vectors/matrices).
     - Layer-wise and neuron-wise change analysis.
   - **Visualization Interface:**
     - Tools to visualize weight changes over time.
     - Heatmaps of weight differences, trajectory plots of individual weights or aggregated statistics.
   - **Integration Hooks:**
     - Simple APIs/callbacks for popular deep learning frameworks (PyTorch, TensorFlow/Keras, JAX).
     - Allow users to easily integrate the tracker into their existing training pipelines.

## 3. Technical Design - Phase 1 (Initial MVP)
   - **Language/Framework:** Python (due to its prevalence in ML).
   - **Data Storage:**
     - Initial: HDF5 or similar for efficient numerical data storage.
     - Future: Explore more scalable database solutions if needed.
   - **Core Library:**
     - `Tracker` class: Manages weight snapshots and configurations.
     - `Snapshot` utility: Handles saving/loading individual weight states.
     - `Comparer` class: Implements different metrics for weight comparison.
   - **Framework Integration (Proof of Concept):**
     - Start with PyTorch integration due to its popularity and ease of hooking into the training loop.
     - Create a simple callback that can be passed to a PyTorch `Trainer` or used in a manual training loop.

## 4. Development Milestones (Iterative)

   - **M1: Basic Weight Snapshotting (PyTorch)**
     - Implement `Tracker` and `Snapshot` for a simple PyTorch MLP.
     - Save weights at the beginning and end of training, and at each epoch.
     - Store basic metadata (epoch, step, timestamp).
   - **M2: Basic Change Calculation**
     - Implement `Comparer` with L2 distance between full weight tensors.
     - Calculate and log the total L2 change between consecutive snapshots.
   - **M3: Simple CLI/Logger Output**
     - Output basic change statistics to the console or a log file.
   - **M4: Layer-wise Change Calculation**
     - Extend `Comparer` to calculate changes per layer.
   - **M5: Initial Visualization (Matplotlib/Seaborn)**
     - Plot total weight change over epochs.
     - Plot per-layer weight change over epochs.
   - **M6: Refactor for Modularity**
     - Ensure clear separation of concerns between tracking, comparison, and storage.
     - Define clear interfaces.
   - **M7: Documentation - Phase 1**
     - README with setup and basic usage.
     - Docstrings for core classes and functions.
   - **M8: TensorFlow/Keras Integration (POC)**
     - Adapt integration hooks for TensorFlow/Keras.
   - **M9: More Advanced Comparison Metrics**
     - Cosine similarity, neuron-level changes.
   - **M10: More Advanced Visualizations**
     - Heatmaps, weight trajectory plots.

## 5. Future Considerations / Advanced Features
   - **Performance Optimization:** For very large models and frequent snapshots.
   - **Distributed Training:** How to handle weight tracking in distributed settings.
   - **Gradient Tracking:** Extend to track gradients as well.
   - **Activation Tracking:** Potentially track changes in activations.
   - **Scalable Backend:** Database solutions for large-scale experiments.
   - **Web-based UI:** Interactive dashboard for exploring results.
   - **Plugin System:** Allow users to easily add new comparison metrics or visualization types.
   - **Support for Quantized/Pruned Models:** How to track changes in such models.
   - **Theoretical Analysis Integration:** Tools to connect observed changes with theoretical measures of learning dynamics.

## 6. Contribution Guidelines (for Open Source)
   - Coding standards (e.g., PEP 8).
   - Testing requirements.
   - Issue tracking and PR process.

--- 