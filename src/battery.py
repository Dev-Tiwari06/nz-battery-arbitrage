"""
Battery arbitrage LP: the best charge/discharge schedule for one day of prices,
and the profit that schedule earns at the prices that actually happened.

The LP is written in matrix form and solved with scipy's HiGHS solver.
Variables, in order:  x = [c_1..c_48, d_1..d_48, s_1..s_48]
    c_t = charging power (MW), d_t = discharging power (MW), s_t = state of charge at end of period t (MWh)
"""
import numpy as np
from scipy.optimize import linprog

# Battery specification (matches Contact Energy's Glenbrook battery: 100 MW / 200 MWh)
BATTERY = dict(
    dt=0.5,          # trading period length (hours)
    p_max=100,       # max charge and discharge power (MW)
    s_min=20,        # minimum state of charge (MWh): 10% reserve
    s_max=200,       # maximum state of charge (MWh)
    s0=100,          # state of charge at the start AND end of every day (MWh)
    eta_c=0.94,      # charging efficiency
    eta_d=0.94,      # discharging efficiency (round trip = 0.94 x 0.94 = 0.88)
    deg_cost=5,      # degradation cost per MWh discharged ($/MWh)
)


def optimise_day(prices, b=BATTERY):
    """Solve the battery LP for one day. Returns (charge, discharge, soc), each an array of length T."""
    p = np.asarray(prices, dtype=float)
    T = len(p)
    I = np.eye(T)                                   # identity matrix
    shift = np.eye(T, k=-1)                         # 1s just below the diagonal: picks out s_{t-1}

    # Objective: maximise  dt * sum[(p_t - k) d_t - p_t c_t]    Note: linprog MINIMISES, so negate
    cost = np.concatenate([b["dt"] * p,                       # charging costs money
                           -b["dt"] * (p - b["deg_cost"]),    # discharging earns money (minus wear)
                           np.zeros(T)])                      # state of charge has no direct cost

    # Energy balance for every period:  s_t - s_{t-1} - dt*eta_c*c_t + (dt/eta_d)*d_t = 0
    # (for t = 1, s_0 is the fixed starting value s0, moved to the right-hand side)
    A_balance = np.hstack([-b["dt"] * b["eta_c"] * I,
                           (b["dt"] / b["eta_d"]) * I,
                           I - shift])
    b_balance = np.zeros(T)
    b_balance[0] = b["s0"]

    # End-of-day condition:  s_T = s0
    A_end = np.zeros((1, 3 * T))
    A_end[0, -1] = 1

    result = linprog(cost,
                     A_eq=np.vstack([A_balance, A_end]),
                     b_eq=np.append(b_balance, b["s0"]),
                     bounds=[(0, b["p_max"])] * (2 * T) + [(b["s_min"], b["s_max"])] * T,
                     method="highs")
    charge, discharge, soc = np.split(result.x, 3)
    return charge, discharge, soc


def settle(charge, discharge, actual_prices, b=BATTERY):
    """Profit ($) of a schedule at the prices that actually happened."""
    p = np.asarray(actual_prices, dtype=float)
    return b["dt"] * np.sum((p - b["deg_cost"]) * discharge - p * charge)


def day_profit(forecast, actual, b=BATTERY):
    """Plan the day on the FORECAST prices, then get paid at the ACTUAL prices."""
    charge, discharge, _ = optimise_day(forecast, b)
    return settle(charge, discharge, actual, b)