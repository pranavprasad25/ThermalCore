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

## Team

### Owner
- Pranav Prasad

