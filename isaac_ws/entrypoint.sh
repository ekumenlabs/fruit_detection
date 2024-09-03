#!/usr/bin/bash

SIM_SCRIPT_PATH="/root/isaac_ws/simulation_ws/scripts/launch_sim.py"
SDG_SCRIPT_PATH="/root/isaac_ws/simulation_ws/scripts/launch_sdg.py"

SCRIPT_PATH=$(case "${MODE}" in
    "SIM") echo "${SIM_SCRIPT_PATH}" ;;
    "SDG") echo "${SDG_SCRIPT_PATH}" ;;
    *) echo "${SIM_SCRIPT_PATH}" ;;
esac)

cd /isaac-sim

./python.sh ${SCRIPT_PATH}
