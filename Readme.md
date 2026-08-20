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
After input removal, the slower dynamics retain the internal state for longer
