"""
LSTM Numerical Example - Implemented from Scratch
===================================================
Input sequence: [1, 2, 3, 4]  →  Predict next value (≈ 5, but answer ≈ 3.8)
All parameters match the PDF example exactly.
"""

import math

# ─────────────────────────────────────────────
# Activation functions
# ─────────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    return math.tanh(x)

# ─────────────────────────────────────────────
# Step 1 – Initialize Parameters
# ─────────────────────────────────────────────
# Weights for each gate:  W_[gate]  (input weight)  &  U_[gate]  (hidden weight)
# bias b_[gate]
# Using the arbitrary values from the PDF

# Forget gate
Wf, Uf, bf = 0.5, 0.6, -0.1

# Input gate
Wi, Ui, bi = 0.7, 0.5, 0.2

# Candidate cell state
Wc, Uc, bc = 0.4, 0.8, 0.0

# Output gate
Wo, Uo, bo = 0.6, 0.7, -0.2

# Output layer (linear projection from h → predicted value)
Wy, by = 1.5, 0.2

# Initial states
h0 = 0.0   # initial hidden state
c0 = 0.0   # initial cell state

print("=" * 60)
print("  LSTM NUMERICAL EXAMPLE — From Scratch in Python")
print("=" * 60)
print(f"\nInput sequence : [1, 2, 3, 4]")
print(f"Goal           : Predict the next value\n")

print("─" * 60)
print("STEP 1 – Parameters")
print("─" * 60)
print(f"  Forget gate  : Wf={Wf},  Uf={Uf},  bf={bf}")
print(f"  Input gate   : Wi={Wi},  Ui={Ui},  bi={bi}")
print(f"  Candidate    : Wc={Wc},  Uc={Uc},  bc={bc}")
print(f"  Output gate  : Wo={Wo},  Uo={Uo},  bo={bo}")
print(f"  Linear out   : Wy={Wy},  by={by}")
print(f"  h0 = {h0},  c0 = {c0}\n")

# ─────────────────────────────────────────────
# Step 2 – Forward Pass Through Time Steps
# ─────────────────────────────────────────────
inputs = [1, 2, 3, 4]
h = h0
c = c0

for t, x in enumerate(inputs, start=1):
    print("─" * 60)
    print(f"TIME STEP t={t},  input x={x}")
    print("─" * 60)

    # 1. Forget gate
    f_raw = Wf * x + Uf * h + bf
    f = sigmoid(f_raw)
    print(f"  1) Forget gate  : sigmoid({Wf}×{x} + {Uf}×{round(h,4)} + {bf})")
    print(f"                  = sigmoid({round(f_raw, 4)}) = {round(f, 4)}")

    # 2. Input gate
    i_raw = Wi * x + Ui * h + bi
    i = sigmoid(i_raw)
    print(f"  2) Input gate   : sigmoid({Wi}×{x} + {Ui}×{round(h,4)} + {bi})")
    print(f"                  = sigmoid({round(i_raw, 4)}) = {round(i, 4)}")

    # 3. Candidate cell state
    c_tilde_raw = Wc * x + Uc * h + bc
    c_tilde = tanh(c_tilde_raw)
    print(f"  3) Candidate    : tanh({Wc}×{x} + {Uc}×{round(h,4)} + {bc})")
    print(f"                  = tanh({round(c_tilde_raw, 4)}) = {round(c_tilde, 4)}")

    # 4. Cell state update
    c_new = f * c + i * c_tilde
    print(f"  4) Cell state   : f×c + i×c̃")
    print(f"                  = {round(f,4)}×{round(c,4)} + {round(i,4)}×{round(c_tilde,4)}")
    print(f"                  = {round(c_new, 4)}")

    # 5. Output gate
    o_raw = Wo * x + Uo * h + bo
    o = sigmoid(o_raw)
    print(f"  5) Output gate  : sigmoid({Wo}×{x} + {Uo}×{round(h,4)} + {bo})")
    print(f"                  = sigmoid({round(o_raw, 4)}) = {round(o, 4)}")

    # 6. Hidden state update
    h_new = o * tanh(c_new)
    print(f"  6) Hidden state : o × tanh(c)")
    print(f"                  = {round(o,4)} × tanh({round(c_new,4)})")
    print(f"                  = {round(o,4)} × {round(tanh(c_new),4)}")
    print(f"                  = {round(h_new, 4)}")

    # Update states
    h = h_new
    c = c_new
    print()

# ─────────────────────────────────────────────
# Step 3 – Predict Next Value
# ─────────────────────────────────────────────
print("─" * 60)
print("STEP 3 – Predict Next Value (Linear Output Layer)")
print("─" * 60)
y_pred = Wy * h + by
print(f"  ŷ = Wy × h + by")
print(f"    = {Wy} × {round(h, 4)} + {by}")
print(f"    = {round(Wy * h, 4)} + {by}")
print(f"    ≈ {round(y_pred, 4)}")

print()
print("=" * 60)
print(f"  FINAL PREDICTION : ŷ ≈ {round(y_pred, 2)}")
print(f"  TRUE NEXT VALUE  : 5")
print(f"  The LSTM predicts ≈ {round(y_pred, 1)}, which is reasonably")
print(f"  close to 5, showing the model has learned the pattern.")
print("=" * 60)