# Energy Optimization in 5G HetNet using MARL

## Traffic Model

The traffic generator is fitted to real-world data and contains arrival rates for each time slot (of 20 mins) and each application category.

## Action Space

- Switch Antennas (-4, 0, +4)
- Switch sleep mode
  - 0: active
  - 1: SM1 (1ms activation delay)
  - 2: SM2 (10ms activation delay)
  - 3: SM3 (100ms activation delay)
- Switch connection mode
  - 0: disconnect all users
  - 1: keep current connections but refuse new connections
  - 2: accept new connections

## State Space

- Total power consumption
- User stats
- Actual and required sum rates
- States of base stations
  - Power consumption
  - Number of active antennas
  - Connection mode
  - Sleep mode
  - Next sleep mode
  - Remaining wakeup time
  - History traffic rates
  - Associated user stats

## Reward

- Weighted sum of drop rates in each app category
- Power consumption

## Notes

- The agents make a decision every 20ms.
- When a BS is in SM1 and a new user arrives, wake up automatically.
