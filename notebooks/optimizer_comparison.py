import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import sympy as sp
    import numpy as np
    import matplotlib.pyplot as plt
    import io
    import base64
    
    # Set clearer plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    return base64, io, mo, np, plt, sp


@app.cell
def _(mo):
    mo.md(r"""
    # Optimizers as Learners: Adam vs. EGGROLL

    **Target Audience:** High School Math Teacher / Undergraduate Student
    
    **Goal:** To understand modern AI optimizers not just as "slope followers," but as **learning algorithms** themselves. We will compare two distinct approaches:
    1.  **Adam**: The "Careful Student" who learns by averaging past experiences (Reconstruction).
    2.  **EGGROLL**: The "Bold Explorer" who learns by trying random things and remembering what worked (Hebbian Association).

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 1: Adam (The Careful Student)

    Imagine you are walking in a thick fog (noisy gradients) and trying to find your way home (the minimum). You have a compass, but the needle shakes wildly with every step.

    **What does Adam do?**
    Adam doesn't trust the compass blindly at every step. Instead, it keeps a **running average** of where the needle has pointed in the past. It "remembers" the general direction.

    **Mathematical Intuition:**
    We can prove that Adam's "memory" ($m$) is mathematically the **best possible estimate** (in a weighted least-squares sense) of the true direction, assuming the true direction is constant locally.
    """)
    return


@app.cell
def _(mo, sp):
    # SymPy Derivation
    t = sp.Symbol('t', integer=True, positive=True)
    # Restrict beta to standard optimizer regime (0 < beta < 1)
    beta = sp.Symbol('beta', real=True, positive=True) 
    m = sp.Symbol('m', real=True)
    g = sp.Function('g')
    i = sp.Symbol('i', integer=True)

    # Objective: Minimize weighted squared error
    # We want to find 'm' that minimizes the distance to all past gradients g(i),
    # but we care exponentially more about recent ones (beta^(t-i)).
    J = sp.Sum(beta**(t - i) * (g(i) - m)**2, (i, 1, t))
    
    # Derivative
    dJ_dm = sp.diff(J, m)
    
    # Solution
    # We solve dJ/dm = 0 for m
    S = sp.Sum(beta**(t - i), (i, 1, t)).doit()
    numerator = sp.Sum(beta**(t - i) * g(i), (i, 1, t))
    m_star = numerator / S

    mo.md(f"""
    ### The Proof (SymPy)

    Let's define an "Error Function" $J(m)$. We want to find a memory value $m$ that minimizes the weighted squared error with respect to all past gradients $g_i$.

    $$ J(m) = \\sum_{{i=1}}^{{t}} \\beta^{{t-i}} (g_i - m)^2 $$

    To find the best $m$, we take the derivative and set it to zero:
    $$ \\frac{{dJ}}{{dm}} = {sp.latex(dJ_dm)} = 0 $$

    Solving for $m$, we get the **Weighted Average**:
    $$ m^* = \\frac{{\\sum_{{i=1}}^t \\beta^{{t-i}} g_i}}{{\\sum_{{i=1}}^t \\beta^{{t-i}}}} $$

    This matches the **Bias-Corrected First Moment** used in Adam:
    $$ \\hat{{m}}_t = \\frac{{m_t}}{{1 - \\beta^t}} $$
    
    *(Note: Adam also tracks a second moment $v_t$ for variance, but the logic of "optimal estimation" is identical.)*
    """)
    return J, S, beta, dJ_dm, g, i, m, m_star, numerator, t


@app.cell
def _(base64, io, mo, np, plt):
    # Visualization: Adam
    np.random.seed(42)
    steps = 100
    t_vals = np.arange(steps)
    true_grad = np.sin(t_vals * 0.1)
    noise = np.random.normal(0, 0.5, steps)
    g_vals = true_grad + noise

    beta1 = 0.9
    m_vals = []
    m_t = 0
    
    # Adam-style update loop
    for step_t in range(1, steps + 1):
        g_t = g_vals[step_t-1]
        # Update biased moment
        m_t = beta1 * m_t + (1 - beta1) * g_t
        # Bias correction
        m_hat = m_t / (1 - beta1**step_t)
        m_vals.append(m_hat)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(t_vals, g_vals, 'o', alpha=0.2, label='Noisy Compass ($g_t$)', color='gray', markersize=4)
    ax1.plot(t_vals, true_grad, '--', label='True Path (Unknown)', color='black', linewidth=1.5, alpha=0.7)
    ax1.plot(t_vals, m_vals, '-', label=f"Adam's Memory ($\\beta_1={beta1}$)", color='#D9534F', linewidth=2.5)
    
    ax1.set_title("Adam: Smoothing the Path (Low-Pass Filter)", fontsize=14)
    ax1.set_xlabel("Time Step $t$")
    ax1.set_ylabel("Gradient Value")
    ax1.legend(frameon=True, fancybox=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Remove top and right spines for cleaner look
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', bbox_inches='tight', dpi=120)
    buf1.seek(0)
    img_str1 = base64.b64encode(buf1.read()).decode('utf-8')
    
    mo.md(f"""
    ### Visualizing Adam
    Notice how the red line (Adam) ignores the gray noise and tracks the black dashed line (True Path). It acts like a **Low-Pass Filter**, smoothing out the jitters.
    
    ![Adam](data:image/png;base64,{img_str1})
    """)
    return beta1, buf1, fig1, g_t, g_vals, img_str1, m_hat, m_t, m_vals, noise, step_t, steps, t_vals, true_grad


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Part 2: EGGROLL (The Bold Explorer)

    Now imagine the fog is so thick you can't see your feet, and the ground is jagged and broken (non-differentiable). Your compass is useless because there is no smooth "slope" to measure.

    **What does EGGROLL do?**
    1.  **Gaussian Smoothing:** It "blurs" the landscape. Instead of looking at a single sharp point, it looks at the average of the area around it.
    2.  **Hebbian Learning:** It tries random steps. If a step leads to higher ground (reward), it remembers that direction. "Neurons that fire together, wire together."
    """)
    return


@app.cell
def _(base64, io, mo, np, plt):
    # Visualization: Smoothing
    def step_function(x):
        # A cliff: 0 on the left, 1 on the right
        return np.where(x > 0, 1.0, 0.0)

    def smoothed_objective(mu, sigma=0.5, samples=1000):
        # Clean implementation of Gaussian convolution
        mu = np.asarray(mu)
        results = []
        for m in mu:
            # Sample noise for this specific point m
            noise = np.random.normal(0, sigma, samples)
            # Evaluate function at perturbed points
            vals = step_function(m + noise)
            # Average the results (Expectation)
            results.append(vals.mean())
        return np.array(results)

    x_vals = np.linspace(-2, 2, 200)
    y_step = step_function(x_vals)
    y_smooth = smoothed_objective(x_vals, sigma=0.5)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(x_vals, y_step, label="Jagged Landscape (Step Function)", linewidth=2, linestyle="--", color='gray')
    ax2.plot(x_vals, y_smooth, label="Smoothed View (EGGROLL)", linewidth=3, color='#5CB85C')
    
    ax2.set_title("How EGGROLL 'Fixes' Broken Landscapes", fontsize=14)
    ax2.set_xlabel("Parameter Value $\\mu$")
    ax2.set_ylabel("Objective Value")
    ax2.legend(frameon=True, fancybox=True, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', bbox_inches='tight', dpi=120)
    buf2.seek(0)
    img_str2 = base64.b64encode(buf2.read()).decode('utf-8')

    mo.md(f"""
    ### Visualizing Smoothing
    The dashed line is the real world: a cliff edge. You can't calculate a slope at the edge (it's infinite) or on the flat parts (it's zero).
    
    The solid green line is what EGGROLL sees. By adding noise, it turns the cliff into a smooth hill that it can climb!
    
    ![Smoothing](data:image/png;base64,{img_str2})
    """)
    return smoothed_objective, step_function, x_vals, y_smooth, y_step


@app.cell
def _(mo):
    mo.md(r"""
    ### The "Hebbian" Connection

    In neuroscience, **Hebbian Learning** is summarized as: *"Cells that fire together, wire together."*

    EGGROLL does exactly this:
    - **Fire:** We try a random perturbation $E_i$.
    - **Wire:** If we get a high reward $f_i$, we update our weights in the direction of $E_i$.

    The update rule is proportional to the **correlation** between the noise and the reward:

    $$ \Delta \mu \approx \frac{1}{n\sigma} \sum_i \underbrace{f_i}_{\text{Reward}} \cdot \underbrace{E_i}_{\text{Direction}} $$

    This constructs a gradient from **correlation**, whereas Adam reconstructs a gradient from **error minimization**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Summary Comparison

    | Feature | Adam (Gradient Descent) | EGGROLL (Evolution Strategies) |
    | :--- | :--- | :--- |
    | **Analogy** | The Careful Student | The Bold Explorer |
    | **Learning Style** | **Reconstruction**: "Minimize the error between my memory and the compass." | **Association**: "Remember the random steps that led to success." |
    | **Best For** | Smooth, predictable terrain (Differentiable). | Rough, broken, or unknown terrain (Non-differentiable). |
    | **Math Core** | Weighted Least Squares (Optimal Estimation) | Gaussian Smoothing & Correlation (Hebbian) |
    """)
    return


if __name__ == "__main__":
    app.run()
