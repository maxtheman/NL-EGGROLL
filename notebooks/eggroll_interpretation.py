import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import io
    import base64
    return base64, io, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # EGGROLL & Nested Learning Interpretation

    ## 1. The "Non-Differentiable" Link: Gaussian Smoothing
    The user asked: *"RNNs aren't differentiable, I wonder if eggroll is related somehow mathematically to the non-differentiable aspect?"*

    **Mathematical Answer:**
    EGGROLL (and ES in general) solves a **smoothed** version of the problem.
    Even if the true objective $f(\theta)$ is non-differentiable (e.g., a step function or discrete RNN), the **expected objective** $J(\mu)$ is smooth:
    $$ J(\mu) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} [ f(\mu + \sigma \epsilon) ] $$
    This operation is a **convolution** of the objective $f$ with a Gaussian kernel. The convolution of *any* bounded function with a Gaussian is infinitely differentiable ($C^\infty$).

    ### Visualization
    Below, we show how a discontinuous "Step Function" becomes a smooth "Sigmoid-like" function when convolved with Gaussian noise.
    """)
    return


@app.cell
def _(base64, io, mo, np, plt):
    # Define a step function (non-differentiable)
    def step_function(x):
        return np.where(x > 0, 1.0, 0.0)

    # Define the smoothed objective J(mu) via Monte Carlo
    def smoothed_objective(mu, sigma=0.5, samples=1000):
        noise = np.random.normal(0, sigma, (samples, len(mu)))
        # Broadcast mu to shape (samples, len(mu))
        # We want to evaluate f(mu_i + noise) for a range of mu values
        # Let's do it per mu point for clarity
        results = []
        for m in mu:
            perturbations = m + noise[:, 0] # 1D noise
            vals = step_function(perturbations)
            results.append(np.mean(vals))
        return np.array(results)

    x_vals = np.linspace(-2, 2, 200)
    y_step = step_function(x_vals)
    y_smooth = smoothed_objective(x_vals, sigma=0.5)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_vals, y_step, label="Original Non-Diff Objective $f(x)$", linewidth=2, linestyle="--")
    ax.plot(x_vals, y_smooth, label="Smoothed Objective $J(\mu)$ (ES)", linewidth=3)
    ax.set_title("Gaussian Smoothing of a Step Function")
    ax.legend()
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')

    mo.md(f"![Smoothing](data:image/png;base64,{img_str})")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. EGGROLL as Associative Memory (Nested Learning)

    In the **Nested Learning** framework, an optimizer is an **Associative Memory** that compresses the "context flow" (gradients) into a state.

    **Hypothesis:**
    EGGROLL is a **Hebbian Associative Memory**.

    - **Keys ($K$)**: The perturbations $E_i$ (exploration directions).
    - **Values ($V$)**: The fitness scores $f_i$ (rewards).
    - **Memory Update**: $\Delta \mu \propto \sum_i f_i E_i$.

    This update rule maximizes the **correlation** (dot product) between the update direction and the successful perturbations.

    $$ \text{Maximize } \sum_i f_i \langle \Delta \mu, E_i \rangle $$

    This is exactly **Hebbian Learning**: "Neurons that fire together, wire together."
    - "Fire together": The perturbation $E_i$ led to high reward $f_i$.
    - "Wire together": We update the weights $\mu$ in the direction of $E_i$.

    Unlike Adam (which minimizes reconstruction error of gradients), EGGROLL **constructs** the gradient from the correlation of noise and reward.
    """)
    return


if __name__ == "__main__":
    app.run()
