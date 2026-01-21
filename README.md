# GNN\_KAN for Root Cause Analysis in Microservices

This repository contains code and experiments for the paper **“GNN\_KAN: Diagnosing Non-Linear Fault Propagation in Scale-Free Microservices via Kolmogorov-Arnold Networks”**.

### Research Direction (Short Overview)

- **Problem**: Modern cloud-native systems decompose applications into many microservices. Failures propagate along complex, often scale-free service graphs, making it hard to quickly localize the true root cause (e.g., the specific service and resource that first failed).
- **Limitation of existing methods**:  
  - Statistical and random-walk based RCA methods (e.g., PageRank-style or change-point detection) assume mostly linear fault propagation and struggle with non-linear resource saturation (such as memory leaks or queue buildup).  
  - Standard GNNs and GATs use scalar attention and MLPs; on scale-free graphs their attention tends to collapse onto hub nodes (like API gateways), and ReLU-based MLPs do not naturally encode soft resource limits or saturation behaviors.
- **Key Idea of GNN\_KAN**:  
  - Model RCA as a learning-to-rank problem on dynamic service graphs, where each node’s score reflects how likely it is to be the root cause.  
  - Replace scalar attention and MLP transformations with **Kolmogorov–Arnold Networks (KANs)** on graph edges and nodes. KANs decompose multivariate mappings into interpretable one-dimensional basis functions that can explicitly encode **resource constraints and saturation effects**.  
  - Introduce a **Rotation-based Periodic Kernel (RPK)** as a bounded, tensor-product basis tailored to capture non-linear resource coupling (e.g., high memory + high latency) and cyclic contention patterns, while keeping computation linear in the number of edges \(O(|E|)\).
- **Dual-Path, Dual-Basis Design**:  
  - Use two complementary graphs: a **propagation graph** (service dependencies) and a **similarity graph** (similar anomaly patterns), and couple them layer-wise in a dual-path GNN.  
  - Combine **RPK** with Chebyshev-based KANs in an **orthogonal frequency filtering** ensemble: RPK focuses on low-frequency, cumulative faults (like memory leaks), while Chebyshev captures high-frequency, transient faults (like network spikes). A simple max operator over the two scores per node acts as a non-parametric selector.
- **Main Findings (Qualitative)**:  
  - On complex, deep topologies (e.g., the Train-Ticket benchmark with 50+ services), GNN\_KAN significantly improves Precision@1 and Top-5 recall over BARO, GAT, and MLP-based GNNs, effectively narrowing the operator’s search space from all services to a small candidate set.  
  - Learned KAN edge functions exhibit clear **saturation zones** and **zero-gradient regions** that align with physical resource limits (e.g., turning “on” only when memory exceeds ~85%), providing interpretable evidence for RCA instead of opaque scalar weights.  
  - Attention collapse in GAT is mitigated because GNN\_KAN learns **functional shapes on edges**, decoupling topological hubness from causal importance.

### High-Level Contribution

GNN\_KAN proposes a **resource-constrained graph neural architecture** for microservice RCA that:

- Integrates **Kolmogorov–Arnold Networks** into message passing to embed resource constraints directly into the model structure.
- Achieves **linear-time inference** in the number of edges and remains practical for online RCA on large microservice graphs.
- Provides **interpretable, function-shaped evidence** (rather than opaque attention scores) for why a service is flagged as the root cause, especially for non-linear, saturation-driven faults such as memory leaks.

For implementation details, experimental setups, and full results, please refer to `PAPER/Conference-LaTeX-template_10-17-19/paper.tex`.
