# Explore-Lite Setup

MavSentry uses m-explore-ros2 for autonomous exploration.

Upstream repository:
https://github.com/robo-friends/m-explore-ros2

Source revision:
326cf8a0b487c34246bb8f3326afbcd69576dc60

## Installation

1. Go to the ROS workspace source directory.
2. Clone the upstream m-explore-ros2 repository.
3. Checkout the source revision listed above.
4. Create the two COLCON_IGNORE files listed below.
5. Build the workspace using colcon.

## Local Configuration

The Raspberry Pi has two additional files:

map_merge/COLCON_IGNORE
map_merge/include/map_merge/COLCON_IGNORE

These prevent Colcon from building the corresponding directories.
