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

## Team

### Owner
-Pranav Prasad
