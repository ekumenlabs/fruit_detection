# Fruit detection

# Using the different docker components

## Architecture

Within the docker directory you'll find a docker compose file, the dockerfiles for each image and some custom configuration files.
The system relies on using profiles to select which set of services build and run depending on the workflow. The following sections explain how to deal with them.

## Profiles

The available profiles are:

- `detection`: loads the detection stack.
- `visualization`: loads RQt to visualize the input and output image processing.
- `test_camera`: loads the usb_cam driver that makes a connected webcam to publish. Useful when the Olive Camera is not available.

Compound profiles are:

- `test_real_pipeline`: loads `test_camera`,`visualization` and `detection`.     
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