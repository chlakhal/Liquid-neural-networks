# 🧠 Liquid Neural Dynamics

## 📌 Overview

This project implements a **recurrent Liquid Neural Network (LNN)** to study continuous-time neural dynamics, adaptive time constants, temporal memory, oscillator-based encoding, and robustness.

It also compares the Liquid Network with a conventional RNN for temporal prediction tasks.

The long-term goal is to explore **neuromorphic Edge AI** implementations using analog oscillators.

---

## 🎯 Objectives

- Implement recurrent liquid dynamics with adaptive τ.
- Study **temporal memory** and prediction.
- Encode states as **frequencies** and oscillators.
- Analyze robustness to noise, quantization, and parameter mismatch.
- Compare **Liquid vs RNN** performance.

---

## 🧮 Core Equation

The liquid neuron follows the continuous-time differential equation:

![Equation](https://latex.codecogs.com/svg.latex?\frac{dx}{dt}=\frac{-x+\tanh(W_{in}u+W_{rec}x)}{\tau(x,u)})

**Explanation:**
- ![x](https://latex.codecogs.com/svg.latex?x) : neuron state  
- ![u](https://latex.codecogs.com/svg.latex?u) : input signal  
- ![W_in](https://latex.codecogs.com/svg.latex?W_{in}) : input weights  
- ![W_rec](https://latex.codecogs.com/svg.latex?W_{rec}) : recurrent weights  
- ![tau](https://latex.codecogs.com/svg.latex?\tau(x,u)) : adaptive time constant  

This equation describes how the neuron’s state evolves continuously over time, balancing decay and nonlinear input/recurrent contributions.

---

### Adaptive Time Constant

The time constant is adaptive:

![Tau](https://latex.codecogs.com/svg.latex?\tau(x,u)=\tau_{min}+\operatorname{softplus}(W_{\tau}[x;u]+b_{\tau}))

**Explanation:**
- ![tau](https://latex.codecogs.com/svg.latex?\tau) controls how quickly the neuron reacts.  
- It changes depending on both the current state and the input.  
- The `softplus` ensures positivity, so the time constant never becomes negative.  

---

### Euler Integration

The dynamics are simulated using Euler integration:

![Euler](https://latex.codecogs.com/svg.latex?x_{t+1}=x_t+\Delta%20t\cdot\frac{-x_t+\tanh(W_{in}u_t+W_{rec}x_t)}{\tau(x_t,u_t)})

**Explanation:**
- Computers simulate discrete steps, so continuous dynamics are approximated using **Euler integration**.  
- At each step, the state is updated by adding the derivative scaled by the timestep ![Delta](https://latex.codecogs.com/svg.latex?\Delta%20t).  

---

## 🧪 Experiments and Results

### 1. Temporal Input

A multi-frequency signal is injected into the network:

![Input](https://latex.codecogs.com/svg.latex?u(t)=\sin(1.5t)+0.3\sin(5t))

**Explanation:**
The input combines slow and fast oscillations, making it suitable for testing the network's ability to capture multi-scale temporal dependencies.

![Temporal Input](results/input.png)

---

### 2. Liquid Neural Dynamics

![Liquid Dynamics](results/liquid_dynamics.png)

**Analysis:**
Each neuron exhibits distinct temporal behavior due to the recurrent coupling, demonstrating the rich internal dynamics of the liquid system.

---

### 3. Adaptive Time Constant

![Adaptive Tau](results/adaptive_tau.png)

**Analysis:**
The time constant changes according to the state and input, allowing the network to adapt its temporal sensitivity.

---

### 4. Liquid Temporal Memory

Two networks are compared: one with short τ and one with longer τ.

![Liquid Memory](results/liquid_memory.png)

**Analysis:**
After input removal, the slower dynamics retain the internal state for longer. This illustrates how τ influences temporal memory.

---

### 5. Temporal / Frequency Encoding

![Frequency Encoding](results/frequency_encoding.png)  
![Oscillator Signal](results/oscillator_signal.png)  
![State Reconstruction](results/state_reconstruction.png)

**Analysis:**
- Frequency encoding provides a temporal representation of the liquid state.  
- The oscillator converts frequency information into a time-domain signal.  
- Quantization and noise provide a simple model of hardware imperfections.  
- The reconstruction shows high encoding fidelity.

---

### 6. Liquid State Evolution

![Liquid State](results/liquid_state.png)

**Analysis:**
The liquid state evolves smoothly over time, reflecting the continuous-time dynamics of the network.

---

### 7. Temporal Prediction — Liquid vs RNN

A multi-frequency signal is predicted one step ahead:

![Prediction Signal](https://latex.codecogs.com/svg.latex?x(t)=\sin(0.7t)+0.5\sin(2.3t)+0.2\sin(4.1t))

![Prediction Comparison](results/prediction_comparison.png)

**Analysis:**
The Liquid Network is compared with a conventional RNN using the same temporal prediction task.  
The comparison highlights the difference between conventional discrete recurrent dynamics and adaptive continuous-time liquid dynamics.

---

### 8. Hardware-Inspired Robustness

![Robustness](results/robustness.png)

**Analysis:**
The experiment evaluates how prediction performance changes under parameter perturbations, providing a simple software-level study of hardware variability.

---

## ✅ Summary

| Aspect               | Liquid Network | RNN Baseline |
|----------------------|----------------|--------------|
| Temporal stability   | High           | Moderate     |
| Adaptive dynamics    | Yes (τ variable) | No         |
| Memory persistence   | Tunable        | Fixed        |
| Frequency encoding   | Robust         | Not native   |
| Noise tolerance      | Strong         | Weak         |
| Prediction accuracy  | Stable but slightly lower | Higher pointwise accuracy |

---

## 🔮 Conclusion

This project demonstrates how **recurrent liquid dynamics** can provide adaptive temporal behavior, controllable memory, frequency-based representations, and robustness to perturbations.

It provides an **algorithmic foundation** for future work on analog neuromorphic hardware based on **oscillators, time-domain computation, and low-power Edge AI**.

---

## 🚀 How to Run

```bash
git clone https://github.com/chlakhal/Liquid-neural-networks.git
cd Liquid-neural-networks
python main.py
