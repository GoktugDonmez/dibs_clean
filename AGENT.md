# 1. General Project Goal

The primary objective is to re-implement the DiBS (Differentiable Bayesian Structure Learning) paper in PyTorch. This serves two purposes:
1.  To gain a deep, practical understanding of the model's components, particularly its gradient structures.
2.  To build a stable foundation that can be extended to explore **deep ensembles** as a scalable and efficient alternative to traditional Bayesian methods for uncertainty quantification in causal discovery.

Once a stable DiBS implementation is achieved, the project will focus on deep ensembles completely:
*   **Nonlinear SCMs:** Extending the model to handle nonlinear causal relationships using neural networks.
*   **Deep Ensembles:** Replacing the Bayesian treatment of parameters with deep ensembles for uncertainty quantification.
*   **Intervention Learning:** Handling datasets with unknown interventions.
*   **Relaxing Assumptions:** Challenging common assumptions like additive noise and uniform variable types.
*   **Benchmarking:** Rigorous comparison against state-of-the-art baselines using metrics like SHD and log-likelihood.

---

# 2. Our Collaboration Protocol

To ensure our collaboration is effective and safe, we will adhere to the following protocol:

1.  **My Role is Research Assistant:** I will act as a thinking partner. My primary functions are to help analyze, debug, compare implementations, find information, and structure code. I am not just a code generator.
2.  **Functionality is Sacred:** I will **never** knowingly change the logical or mathematical functionality of the code without explicit prior discussion and approval.
3.  **The "Propose, Justify, Await" Protocol:** For any code modification, I will follow a strict three-step process:
    *   **Propose:** Clearly state the change I intend to make.
    *   **Justify:** Provide a detailed rationale for the change, referencing our shared context.
    *   **Await:** Always wait for an explicit "yes" or "proceed" before making any change with my tools.
4.  **Atomic Changes:** I will propose changes in the smallest logical units possible to make them easy to review and safe to implement.

---

# 3. Progress Log

*Reverse-chronological order (newest on top).*

**2025-07-09**
*   **Vectorised version:** The faster vectorised version of `debug/dibs_score_func_refactored.py` is implemented in `debug/dibs_vectorised`.py 

*   **Issues learning:** The model still does not learn correctly. chain is learned n=3 but when samples increased from 100 to 300 it is not correct too. sometimes it gets closely with n=4 n=5. 

*   **Decision:** Established a new three-part structure for `AGENT.md` to track the project goal, our collaboration protocol, and a running progress log. The protocol is now stored here to ensure it's synced across different machines via the git repository.
*   **Status:** The main implementation challenge is achieving a numerically stable and correct DiBS model in PyTorch. The current approach uses the score function estimator, as the reparameterization trick was found to be unstable.
*   **Key Files:** The most up-to-date code is in `debug/dibs_score_func_refactored.py`.
*   **Next Step:** (DONE) Begin the analysis of the gradient flow in `debug/dibs_score_func_refactored.py`, focusing on the correct usage of `requires_grad` and `.detach()` to prevent gradient leakage (This is done now, .deteach added in training loop).