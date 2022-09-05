# Energy Optimization in 5G HetNet using MARL

## Traffic Model

The traffic generator is fitted to real-world data and contains arrival rates for each time slot (of 20 mins) and each application category.

## Action Space

- Switch Antennas (-16, -4, 0, +4, +16)
- Switch sleep mode
  - 0: active
  - 1: SM1 (1ms activation delay)
  - 2: SM2 (10ms activation delay)
  - 3: SM3 (1s activation delay)
- Switch connection mode
  - 0: disconnect all users
  - 1: serve current users but refuse new connections
  - 2: accept new connections
  - 3: take over all users that are in its cell but not connected to it

## Observation Space

<!-- - Load prediction (by ES) for each app category in the next 10 secs -->
- Minimum sum rate at present for each app cat
- The state of the current BS
- States of all users in the coverage

## Reward

- Weighted sum of drop rates in each app cat
- Power consumption

## Notes

<!-- - Make a decision whenever users have been added or removed from the network -->
<!-- - When the BS is sleeping and a new user arrives, make a decision at once -->
<!-- - For a sleeping BS with waiting users, make a decision every 10 ms -->
<!-- - For a sleeping BS with no waiting users, make a decision every
  - 10ms, for SM1
  - 100ms, for SM2
  - 10s, for SM3 -->
- Make a decision every 10ms, except for SM3, where it is every 1s.
- When a BS is in SM1 and a new user arrives, wake up automatically.
