# Thermal-Aware Processor

## Problem

Modern processors are increasingly limited by thermal constraints.
Increasing transistor density and computational workloads can create
localized thermal hotspots, which can affect performance and reliability.

## Proposed Solution

This project aims to develop a thermally-aware processor architecture
that detects thermal hotspots and dynamically moves computation away
from overheated regions.

## Objectives

- Monitor processor core temperatures
- Detect thermal hotspots
- Identify cooler processing regions
- Dynamically migrate workloads
- Reduce localized thermal stress
- Evaluate the effect on processor performance

## Proposed Components

- Thermal Monitor
- Hotspot Detector
- Thermal-Aware Scheduler
- Task Migration Mechanism
- Processor Simulator

## Initial Prototype

The initial prototype will simulate a multi-core processor where each
core has a temperature and workload. When a core exceeds a defined
thermal threshold, the scheduler will migrate its workload to a
cooler processing region.

## Subsystems

### Power Estimation Subsystem (`src/power`)

The Power Estimation subsystem models CPU power dissipation dynamically and statically based on operating conditions:

- **Dynamic Power**: $P_{\text{dynamic}} = \alpha \cdot C_{\text{eff}} \cdot V^2 \cdot f$
  - $\alpha$: Switching activity factor derived from normalized workload ($[0.0, 1.0]$) via $\alpha = \alpha_{\text{base}} + \text{workload} \cdot \alpha_{\text{scale}}$.
  - $C_{\text{eff}}$: Effective switching capacitance in Farads (F).
  - $V$: Operating voltage in Volts (V).
  - $f$: Clock frequency in Hertz (Hz).
- **Static / Leakage Power**: $P_{\text{static}} = P_{\text{base}} \cdot f(T) \cdot g(V)$
  - $P_{\text{base}}$: Baseline static power in Watts (W).
  - $f(T)$: Thermal leakage scaling (`CONSTANT`, `LINEAR` via $1 + \beta \cdot (T - T_{\text{ref}})$, or `EXPONENTIAL` via $\exp(\beta \cdot (T - T_{\text{ref}}))$).
  - $g(V)$: Optional voltage-dependent leakage scaling ($V / V_{\text{ref}}$).
- **Total Power**: $P_{\text{total}} = P_{\text{dynamic}} + P_{\text{static}}$

#### Python API Usage

```python
from src.power import PowerConfig, PowerEstimator

config = PowerConfig(capacitance=1.5e-9, base_static_power=5.0)
estimator = PowerEstimator(config=config)

result = estimator.estimate(
    workload=0.8,       # 80% CPU load
    voltage=1.1,        # 1.1 V
    frequency=3.0e9,    # 3.0 GHz (Hz)
    temperature=65.0,   # 65 °C
)

print(f"Dynamic Power: {result.dynamic_power:.4f} W")
print(f"Static Power:  {result.static_power:.4f} W")
print(f"Total Power:   {result.total_power:.4f} W")
```

### Thermal RC Model Subsystem (`src/thermal`)

The Thermal Model subsystem converts dynamic/static power consumption into temperature evolution using a first-order equivalent thermal RC network:

- **Thermal Resistance ($R_\theta$)**: Relates power dissipation to equilibrium temperature rise:
  $$\Delta T = P \cdot R_\theta$$
  $$T_{\text{steady}} = T_{\text{ambient}} + P \cdot R_\theta$$
- **Thermal Capacitance ($C_{\text{th}}$)**: Models package thermal mass and thermal time constant $\tau$:
  $$C_{\text{th}} \frac{dT}{dt} = P(t) - \frac{T(t) - T_{\text{ambient}}}{R_\theta}$$
  $$\tau = R_\theta \cdot C_{\text{th}}$$
- **Transient Response**: Single-step and time-series simulation supporting constant or dynamic time-varying power profiles:
  $$T(t + \Delta t) = T_{\text{steady}} + (T(t) - T_{\text{steady}}) \cdot e^{-\Delta t / \tau}$$

#### Python API Usage

```python
from src.thermal import ThermalModel, ThermalConfig

model = ThermalModel(
    thermal_resistance=1.2,     # 1.2 °C/W
    thermal_capacitance=20.0,   # 20.0 J/°C (tau = 24.0s)
    ambient_temperature=25.0,   # 25.0 °C
    initial_temperature=25.0,
)

# 1. Direct steady-state calculation
t_steady = model.steady_state_temperature(power=25.0)  # 55.0 °C

# 2. Transient simulation over time
simulation = model.simulate(power=25.0, duration=120.0, timestep=1.0)
print(f"Final Temp: {simulation.final_temperature:.2f} °C")
```

## Team

### Owner
- Pranav Prasad


