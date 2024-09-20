TODO: not setting the DATASET_PATH environment variable when composing the training profile prevents it from succeding, we need to decide how to solve this issue.

# Fruit detection

# Requisites

- [Docker](https://docs.docker.com/engine/install/ubuntu/)
- Ubuntu 20.04 / 22.04
- NVidia GPU GeForce RTX 3070 or higher.
- [NVidia GPU Driver](https://www.nvidia.com/en-us/drivers/unix/)
- [NVidia Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [Omniverse-launcher](https://www.nvidia.com/en-us/omniverse/download/)
- [Nucleus](https://docs.omniverse.nvidia.com/nucleus/latest/workstation/installation.html)

We recommend reading this [article](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html) from NVidia Omniverse which explains the basic configuration.

> **NOTE:** this project is disk savvy, make sure to have tens of GBs (~50GB) available of free disk space.

## Contributing

This projects uses pre-commit hooks for linting. To install and make sure they are run when committing:

```bash
python3 -m pip install -U pre-commit
pre-commit install
```

If you want to run the linters but still not ready to commit you can run:

```bash
pre-commit run --all-files
```

# Using the different docker components

## Architecture

Within the docker directory you'll find a docker compose file, the dockerfiles for each image and some custom configuration files.
The system relies on using profiles to select which set of services build and run depending on the workflow. The following sections explain how to deal with them.

## Profiles

The available profiles are:

- `training`: trains a fasterrcnn_resnet50_fpn model based on a synthetic dataset.
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

- `training_test`: runs the tests for the training stack.
- `detection_test`: runs the tests for the detection stack.

## Build the images

To build all the docker images:

```bash
docker compose -f docker/docker-compose.yml --profile "*" build
```

## Training

To train a model you need a NVidia Omniverse synthetic dataset. You first need to set up the following environment variable:
```
export DATASET_PATH=PATH/TO/TRAINING/DATA
```

Then you can run the training using the training profile:

```bash
docker compose -f docker/docker-compose.yml --profile training up
```

After the training ends, a `model.pth` file will be available inside `model`. Additionally, you will notice that the dataset files were organized in different folders based on their extension. To test the model you can run:

```bash
docker compose -f docker/docker-compose.yml --profile training_test up
```

This will evaluate every image in the `DATASET_PATH` and generate annotated images in the `model` folder.

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

### Running test_real_pipeline

For running this pipeline is needed to have a trained model (.pth file) on the `model` folder. By default, the detection service will try to load a file called `model.pth`, but this can be override by changing the `model_path` parameter from `detection_ws/src/detection/launch/detection.launch.py`.

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
The following .gif video shows pictures where the ground plane conditions color is randomized having a better dataset for the simulation.

![Dataset gen](./doc/dataset_gen.gif)

And once it finishes (note the scene does not evolve anymore) check the generated folder under `isaac_ws/datasets/YYYYMMDDHHMMSS_out_fruit_sdg` where `YYYYMMDDHHMMSS` is the stamp of the dataset creation.

# Contributing

Issues or PRs are always welcome! Please refer to [CONTRIBUTING](CONTRIBUTING.md) document.

# Code of Conduct

The free software code of conduct fosters an inclusive, respectful community for contributors by promoting collaboration and mutual respect. For more details, refer to the full document [Code of Conduct](CODE_OF_CONDUCT.md).

# FAQs

1. How do I clean up all the docker resources?

Your good old friend `docker system prune` and the more agressive `docker system prune --all`. **Caution:** it will likely erase stuff you didn't want to erase as it is a blanket prune. Read the documentation for more information.

2. Do you have problems with XWindows?

```bash
xhost +si:localuser:root
```
