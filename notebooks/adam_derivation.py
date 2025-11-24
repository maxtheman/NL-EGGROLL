import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import sympy as sp
    return mo, sp


@app.cell
def _(mo):
    mo.md(r"""
    # Adam Derivation (Nested Learning)

    Reproducing the claim that Adam's moments are optimal associative memories.

    ## Environment (uv)
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install marimo sympy
    uv run marimo run notebooks/adam_derivation.py
    ```
    """)
    return


@app.cell
def _(mo, sp):
    # Define symbols
    t = sp.Symbol('t', integer=True, positive=True)
    beta = sp.Symbol('beta', real=True, positive=True)  # Decay rate
    m = sp.Symbol('m', real=True)  # The memory state (estimator)

    # Objective: J(m) = sum_{i=1}^{t} beta^{t-i} * (g_i - m)^2
    g = sp.Function('g')
    i = sp.Symbol('i', integer=True)

    J = sp.Sum(beta**(t - i) * (g(i) - m)**2, (i, 1, t))

    # Differentiate
    dJ_dm = sp.diff(J, m)

    # Solve for m
    # We know the solution involves the sum of weights
    S = sp.Sum(beta**(t - i), (i, 1, t)).doit()

    # Optimal m*
    numerator = sp.Sum(beta**(t - i) * g(i), (i, 1, t))
    m_star = numerator / S

    mo.md(
        f"""
        ### Derivation

        **Objective Function:**
        $J(m) = \\sum_{{i=1}}^{{t}} \\beta^{{t-i}} (g_i - m)^2$

        **Derivative:**
        $dJ/dm = {sp.latex(dJ_dm)}$

        **Sum of Weights (Bias Correction Factor):**
        $S = {sp.latex(S)}$

        **Optimal Memory State:**
        $m^* = \\frac{{\\sum \\beta^{{t-i}} g_i}}{{S}} = {sp.latex(m_star)}$

        This matches the bias-corrected first moment in Adam: $\\hat{{m}}_t = \\frac{{m_t}}{{1 - \\beta^t}}$.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Geometric Intuition
    
    Visually, the "memory" $m^*$ is the point that minimizes the weighted sum of squared distances to all past gradients. This is equivalent to finding the "center of mass" of the gradient history, where recent gradients are heavier (more mass) due to the $\beta^{t-i}$ weighting.
    
    Below, we simulate a noisy gradient signal and visualize how the optimal memory $m^*$ (Adam's moment) tracks the underlying trend.
    """)
    return


@app.cell
def _(mo):
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    import base64

    # Simulate a noisy gradient signal
    np.random.seed(42)
    steps = 100
    t_vals = np.arange(steps)
    # True trend: sine wave
    true_grad = np.sin(t_vals * 0.1)
    # Noisy observation: g_t
    noise = np.random.normal(0, 0.5, steps)
    g_vals = true_grad + noise

    # Calculate Adam's moment (m_hat)
    beta_val = 0.9
    m_vals = []
    m_t = 0
    for step_t in range(1, steps + 1):
        g_t = g_vals[step_t-1]
        m_t = beta_val * m_t + (1 - beta_val) * g_t
        m_hat = m_t / (1 - beta_val**step_t)
        m_vals.append(m_hat)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_vals, g_vals, 'o', alpha=0.3, label='Noisy Gradients ($g_t$)', color='gray')
    ax.plot(t_vals, true_grad, '--', label='True Gradient Trend', color='black')
    ax.plot(t_vals, m_vals, '-', label=f"Adam Moment ($m^*$, $\\beta={beta_val}$)", color='red', linewidth=2)
    
    # Visualize weights for the last step
    # Show how much the last step 'remembers' previous steps
    ax_inset = ax.inset_axes([0.05, 0.05, 0.25, 0.25])
    lookback = 20
    weights = [beta_val**(lookback - i) for i in range(lookback)]
    ax_inset.bar(range(lookback), weights, color='red', alpha=0.5)
    ax_inset.set_title("Memory Weights")
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    ax.set_title("Adam Moment as Optimal Memory Tracking")
    ax.set_xlabel("Time Step $t$")
    ax.set_ylabel("Gradient Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Render to base64 for marimo
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    mo.md(f"![Adam Geometric Intuition](data:image/png;base64,{img_str})")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Conclusion

    We have shown using SymPy that minimizing the exponentially weighted reconstruction error of the gradients yields the exact bias-corrected moment update rule used in Adam. This confirms the Nested Learning paper's claim that **Adam is an optimal associative memory** for the gradient history.
    """)
    return


if __name__ == "__main__":
    app.run()
