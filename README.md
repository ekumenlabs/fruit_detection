# Fruit detection

# Requisites

- [Docker](https://docs.docker.com/engine/install/ubuntu/)
- Ubuntu 20.04 / 22.04
- NVidia GPU GeForce RTX 3070 or higher.
- [NVidia GPU Driver](https://www.nvidia.com/en-us/drivers/unix/)
- [NVidia Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)

We recommend reading this [article](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html) from NVidia Omniverse which explains the basic configuration.

> **NOTE:** this project is disk savvy, make sure to have tens of GBs (~50GB) available of free disk space.


# Using the different docker components

## Architecture

Within the docker directory you'll find a docker compose file, the dockerfiles for each image and some custom configuration files.
The system relies on using profiles to select which set of services build and run depending on the workflow. The following sections explain how to deal with them.

## Profiles

The available profiles are:

- `detection`: loads the detection stack.
- `visualization`: loads RQt to visualize the input and output image processing.
- `test_camera`: loads the usb_cam driver that makes a connected webcam to publish. Useful when the Olive Camera is not available.
- `simulation`: loads the simulation NVidia Isaac Omniverse. 
- `dataset_gen`: generates a training dataset using NVidia Isaac Omniverse. 
> TBD

Compound profiles are:

- `test_real_pipeline`: loads `test_camera`,`visualization` and `detection`.
- `simulated_pipeline`: loads `simulation`,`visualization` and `detection`.

> TBD

Testing profiles are:

- `detection_test`: runs the tests for the detection stack.

## Build the images

To build all the docker images:

```bash
docker compose -f docker/docker-compose.yml --profile "*" build
```

## Run

To run the system you need to define which profile(s) to run. You can pile profiles by adding them one after the other to have a custom bring up of the system (e.g.`--profile detection --profile visualization`).

To load the test (camera) real system, you can:

```bash
docker compose -f docker/docker-compose.yml --profile test_real_pipeline up
```

To stop the system you can Ctrl-C or from another terminal call:

```bash
docker compose -f docker/docker-compose.yml --profile test_real_pipeline down
```

## Test

### Detection stack

```bash
docker compose -f docker/docker-compose.yml --profile detection_test build
```

## Dataset generation

It generates a dataset with 100 annotated pictures where the lighting conditions and the fruit pose is randomized.
To generate a new dataset:

```bash
docker compose -f docker/docker-compose.yml --profile dataset_gen up
```

And once it finishes (note the scene does not evolve anymore) check the generated folder under `isaac_ws/datasets/YYYYMMDDHHMMSS_out_fruit_sdg` where `YYYYMMDDHHMMSS` is the stamp of the dataset creation. 


# FAQs

1. How do I clean up all the docker resources?

Your good old friend `docker system prune` and the more agressive `docker system prune --all`. **Caution:** it will likely erase stuff you didn't want to erase as it is a blanket prune. Read the documentation for more information.