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

### End-to-End Simulation Pipeline (`src/simulation`)

The ThermoShift End-to-End Simulation Subsystem orchestrates the complete simulation pipeline connecting workload specifications, CPU operating conditions, physical power estimation, and transient thermal RC modeling.

```
Workload Profile  ──>  Operating Conditions  ──>  Power Estimator  ──>  Power(t)  ──>  Thermal Model  ──>  Temperature(t)  ──>  Simulation Results
```

#### Architecture & Pipeline Execution Flow

For each discrete timestep $t_k = k \cdot \Delta t$:
1. **Workload Sampling**: Evaluates normalized utilization $w(t_k) \in [0.0, 1.0]$ from a time-varying `WorkloadProfile`.
2. **DVFS Mapping**: Converts workload into voltage $V$ and clock frequency $f$ using `LinearDVFSMapper`, `DiscreteDVFSMapper`, or `FixedOperatingConditionMapper`.
3. **Power Estimation**: Calls the existing `PowerEstimator` to compute dynamic power $P_{\text{dynamic}}$, static leakage power $P_{\text{static}}$ (scaled by current core temperature $T(t_k)$), and total power $P_{\text{total}}$.
4. **Thermal Integration**: Feeds total power into the existing `ThermalModel` to compute temperature evolution $T(t_k + \Delta t)$ over timestep $\Delta t$.
5. **State Recording**: Stores structured step result `SimulationStepResult` containing time, workload, operating conditions, power components, and junction temperature.

#### Workload Input Format (`src/workload`)

Workload utilization is represented as a normalized ratio $w \in [0.0, 1.0]$ ($0.0 = 0\%$, $1.0 = 100\%$). `WorkloadProfile` supports:
- **Constant Workload**: `WorkloadProfile.pattern_constant(workload=0.8, duration=10.0)`
- **Step Workload**: `WorkloadProfile.pattern_step(initial_workload=0.2, high_workload=0.8, final_workload=0.4, step_times=(2.0, 6.0), duration=10.0)`
- **Phase-Based Profiles**: `WorkloadProfile.from_phases([(duration1, workload1), (duration2, workload2)])`
- **Increasing / Decreasing Patterns**: `WorkloadProfile.pattern_increasing(...)`, `WorkloadProfile.pattern_decreasing(...)`
- **Mixed Workload**: `WorkloadProfile.pattern_mixed(duration=25.0)`
- **Arbitrary Time-Varying Functions**: `WorkloadProfile.from_function(func, duration, dt)`

#### Timestep & Integration Behavior

- Timestamps are evaluated at discrete points $t_k = k \cdot \Delta t$ for $k = 0, 1, \dots, N$ where $N = \text{round}(\text{duration} / \Delta t)$.
- Step workloads use right-continuous piecewise-constant evaluation, guaranteeing that step transitions take effect immediately at step boundary timestamps.
- Thermal integration preserves thermal mass inertia ($\tau = R_\theta \cdot C_{\text{th}}$), producing realistic transient temperature curves rather than immediate temperature jumps.

#### Python API Usage

```python
from src.workload import WorkloadProfile
from src.power import PowerConfig, PowerEstimator
from src.thermal import ThermalConfig, ThermalModel
from src.cpu import LinearDVFSMapper
from src.simulation import ThermoShiftSimulation

# 1. Define Workload Profile
timeline = [(0.0, 0.2), (2.0, 0.8), (6.0, 0.4)]
profile = WorkloadProfile.from_step(steps=timeline)

# 2. Configure Power and Thermal Subsystems
power_estimator = PowerEstimator(PowerConfig(base_static_power=5.0, capacitance=1.5e-9))
thermal_model = ThermalModel(ThermalConfig(ambient_temperature=25.0, thermal_resistance=1.2, thermal_capacitance=15.0))
dvfs_mapper = LinearDVFSMapper(min_voltage=0.8, max_voltage=1.25, min_frequency=1.0e9, max_frequency=3.5e9)

# 3. Instantiate and Run Simulation Engine
simulation = ThermoShiftSimulation(
    power_estimator=power_estimator,
    thermal_model=thermal_model,
    dvfs_mapper=dvfs_mapper,
)
results = simulation.run(workload_profile=profile, duration=10.0, timestep=1.0)

# 4. Access Results and Summary
summary = results.summary()
print(f"Peak Total Power: {summary.peak_total_power:.2f} W at t={summary.time_at_peak_power}s")
print(f"Peak Temperature: {summary.peak_temperature:.2f} °C at t={summary.time_at_peak_temperature}s")
print(f"Final Temperature: {results.final_temperature:.2f} °C")
```

#### Assumptions & Limitations

- **Single-Node Thermal Mass**: Models processor core/package thermal behavior using a first-order lump-parameter thermal RC circuit ($R_\theta, C_{\text{th}}$).
- **Core Frequency/Voltage Mapping**: Assumes immediate DVFS state switching without voltage regulator lock time delays.
- **Ambient Stability**: Assumes constant ambient environment temperature during the simulation run.

### Thermal Safety & Limits Subsystem (`src/safety`)

The Thermal Safety Subsystem monitors core junction temperatures, evaluates configured safety thresholds, classifies operational safety states, detects boundary crossing events, and performs time-series safety analysis on simulation results.

```
Temperature(t)  ──>  Thermal Safety Monitor  ──>  Compare Against Limits  ──>  Thermal Status  ──>  NORMAL / WARNING / CRITICAL / OVERHEATING
```

#### Thermal State Hierarchy & Safety Semantics

- **NORMAL** ($T < T_{\text{warning}}$): Safe operating temperature. `is_safe == True`.
- **WARNING** ($T_{\text{warning}} \le T < T_{\text{critical}}$): Temperature approaching elevated limits. `is_safe == False`, `is_warning == True`.
- **CRITICAL** ($T_{\text{critical}} \le T < T_{\text{maximum}}$): Critical thermal stress. Immediate thermal throttling or workload migration recommended. `is_safe == False`, `is_critical == True`.
- **OVERHEATING** ($T \ge T_{\text{maximum}}$): Hardware thermal limit reached/exceeded. Thermal emergency shutdown or severe throttling mandatory. `is_safe == False`, `is_overheating == True`.

*Strict Safety Rule*: `is_safe` evaluates to `True` **ONLY** when `status == ThermalStatus.NORMAL`.

#### Configurable Thresholds & Hysteresis

Configured via `ThermalSafetyConfig`:
- `warning_temperature`: Trigger boundary for WARNING state (default 70.0 °C).
- `critical_temperature`: Trigger boundary for CRITICAL state (default 85.0 °C).
- `maximum_temperature`: Trigger boundary for OVERHEATING state (default 95.0 °C).
- `hysteresis`: Temperature offset in °C required to de-escalate states, preventing rapid state flapping around threshold boundaries.

Enforced Constraint: $T_{\text{warning}} < T_{\text{critical}} < T_{\text{maximum}}$.

#### Threshold Crossing & Event Detection

The subsystem identifies transition events between consecutive evaluations:
- `WARNING_ENTERED`, `CRITICAL_ENTERED`, `OVERHEATING_ENTERED`
- `WARNING_CLEARED`, `CRITICAL_CLEARED`, `OVERHEATING_CLEARED`

#### Python API Usage

```python
from src.safety import ThermalSafetyConfig, ThermalSafetyMonitor, ThermalStatus
from src.simulation import ThermoShiftSimulation

# 1. Single-Temperature Safety Check
monitor = ThermalSafetyMonitor(warning_temperature=70.0, critical_temperature=85.0, maximum_temperature=95.0)
result = monitor.check_temperature(88.5)

print(f"Status: {result.status.value}")      # CRITICAL
print(f"Is Safe: {result.is_safe}")          # False
print(f"Is Critical: {result.is_critical}")  # True

# 2. Time-Series Simulation Analysis
simulation = ThermoShiftSimulation(...)
sim_results = simulation.run(...)

# Analyze simulation safety results
analysis = sim_results.analyze_safety(monitor)

summary = analysis.summary()
print(f"Maximum Temp: {summary['maximum_temperature']} °C")
print(f"Has Overheating: {summary['has_overheating']}")
print(f"Warning Timesteps: {summary['warning_timesteps']} ({summary['warning_duration_seconds']}s)")
print(f"Transitions Logged: {len(analysis.transitions)}")
```

## Team

### Owner
- Pranav Prasad


