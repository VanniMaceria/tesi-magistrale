# tesi-magistrale
This research project, developed as part of my Master's Thesis in Computer Science (IoT Curriculum), focuses on the challenges of deploying **Federated Learning (FL)** in resource-constrained environments. 

While FL is widely adopted for privacy-preserving distributed learning, its execution on **Microcontroller Units (MCUs)** and low-power IoT devices remains a significant challenge due to hardware limitations in memory, processing power, and energy autonomy.

## 🔬 Research Goal
The primary objective of this study is to bridge the gap between high-performance Federated Learning and constrained edge computing. The project aims to identify, implement, and compare various **model compression and optimization techniques** to minimize the footprint of neural networks during the federated training process.

## 📈 Methodology
The research follows a two-phase comparative approach:

### Phase 1: Baseline Characterization
The development of a standard Federated Learning environment to establish a performance baseline. During this phase, key metrics are collected from uncompressed models, including:
* **Global and Local Accuracy**.
* **Bandwidth Consumption**
* **Computational Overhead**
* **Theoretical Energy Impact**

### Phase 2: Optimization & Comparative Analysis
The core of the thesis involves the application of complexity-reduction techniques to the baseline. The study evaluates the trade-off between model performance and resource efficiency using:
* **Pruning**: Systematic removal of redundant weights to sparsify the model.
* **Quantization**: Reducing weight precision (e.g., from Float32 to Int8/Float16) to optimize memory and bandwidth.
* **Knowledge Distillation**: Utilizing "Teacher-Student" architectures to transfer knowledge to ultra-lightweight models.

## 🧪 Experimental Setup
The project utilizes a simulation-driven approach to perform tests. By varying parameters such as the number of participating clients, data distribution (IID vs. Non-IID), and local training intensity (epochs/rounds), the research seeks to identify the most "efficient" technique for IoT deployments—prioritizing energy and bandwidth savings even at the cost of marginal accuracy loss. [Flower](https://flower.ai/) is the framework that has been choosen to perform this task.

## 🛠️ Project Structure
* **baseline_flower**: This directory hosts the **simulated federated environment**. It is designed to orchestrate decentralized training rounds and serve as the primary testbed for performance characterization. By executing various optimization strategies within this environment, we extract critical metrics such as global accuracy, communication overhead, and resource utilization, which are essential for the subsequent comparative analysis.

---
*Note: The results from this simulation phase are intended to validate the feasibility of a subsequent deployment on physical MCU hardware.*
