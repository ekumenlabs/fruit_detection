from datetime import datetime

import numpy as np

import carb

from isaacsim import SimulationApp

import yaml

yaml_file_path = "/root/isaac_ws/simulation_ws/scripts/config.yml"

with open(yaml_file_path, 'r') as file:
    try:
        config = yaml.safe_load(file)
        print(config)
    except yaml.YAMLError as exc:
        print(f"Error while reading YAML file: {exc}")

stamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
simulation_app = SimulationApp(launch_config=config["SIMULATION_APP_CONFIG"])


# Late import of runtime modules (the SimulationApp needs to be created before loading the modules)
import omni.replicator.core as rep
# Custom util functions for the example
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.utils import prims
from omni.isaac.core.utils.semantics import remove_all_semantics
from omni.isaac.core.utils.stage import get_current_stage, create_new_stage
from omni.isaac.nucleus import get_assets_root_path
import pxr
from pxr import Gf, UsdGeom

# Get server path
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not get nucleus server path, closing application.")
    simulation_app.close()

# Creates a new stage.
if not create_new_stage():
    carb.log_error(f"Could not create a new stage, closing the application.")
    simulation_app.close()
stage = get_current_stage()

# Disable capture on play (data generation will be triggered manually)
rep.orchestrator.set_capture_on_play(False)

# Clear any previous semantic data in the loaded stage
for prim in stage.Traverse():
    remove_all_semantics(prim, False)

# Create a ground plane
PhysicsContext()
ground_plane = GroundPlane(prim_path=config["GROUND_PLANE"]["prim_path"], size=config["GROUND_PLANE"]["size"], color=np.array(config["GROUND_PLANE"]["color"]))

# Spawn an apple in a random pose.
apple_prim = prims.create_prim(
    prim_path=config["SEMANTIC_OBJECTS"]["Apple"]["prim"],
    position=(-0.1, -0.05, 0.5),
    orientation=(1., 0., 0., 0.),
    scale=config["OBJECTS_SCALE"],
    usd_path=config["SEMANTIC_OBJECTS"]["Apple"]["url"],
    semantic_label=config["SEMANTIC_OBJECTS"]["Apple"]["class"],
)

# Spawn an avocado in a random pose.
avocado_prim = prims.create_prim(
    prim_path=config["SEMANTIC_OBJECTS"]["Avocado"]["prim"],
    position=(0, 0.1, 0.5),
    orientation=(1., 0., 0., 0.),
    scale=config["OBJECTS_SCALE"],
    usd_path=config["SEMANTIC_OBJECTS"]["Avocado"]["url"],
    semantic_label=config["SEMANTIC_OBJECTS"]["Avocado"]["class"],
)

# Spawn a lime in a random pose.
lime_prim = prims.create_prim(
    prim_path=config["SEMANTIC_OBJECTS"]["Lime"]["prim"],
    position=(0.1, -0.05, 0.5),
    orientation=(1., 0., 0., 0.),
    scale=config["OBJECTS_SCALE"],
    usd_path=config["SEMANTIC_OBJECTS"]["Lime"]["url"],
    semantic_label=config["SEMANTIC_OBJECTS"]["Lime"]["class"],
)

# Create the camera used for the acquisition.
sdg_camera = rep.create.camera(
    name=config["SDG_CAMERA"]["name"],
    position=config["SDG_CAMERA"]["pos"],
    rotation=config["SDG_CAMERA"]["rot"],
    focal_length=config["SDG_CAMERA"]["focal_length"],
    focus_distance=config["SDG_CAMERA"]["focus_distance"],
    f_stop=config["SDG_CAMERA"]["f_stop"],
    horizontal_aperture=config["SDG_CAMERA"]["horizontal_aperture"],
    clipping_range=Gf.Vec2f(*config["SDG_CAMERA"]["clipping_range"]),
    projection_type=config["SDG_CAMERA"]["projection_type"],
    count=1,
)
sdg_camera_render_product = rep.create.render_product(
    sdg_camera, (config["SDG_CAMERA"]["width"], config["SDG_CAMERA"]["height"]), name="SdgCameraView"
)
sdg_camera_render_product.hydra_texture.set_updates_enabled(False)

def register_move_objects():
    def move_objects():
        object_prims = rep.get.prims(semantics=[
            ("class", config["SEMANTIC_OBJECTS"]["Apple"]["class"]),
            ("class", config["SEMANTIC_OBJECTS"]["Avocado"]["class"]),
            ("class", config["SEMANTIC_OBJECTS"]["Lime"]["class"]),
        ])
        with object_prims:
            rep.modify.pose(
                position=rep.distribution.uniform(config["OBJECTS_POSE_CONFIG"]["min_pos"], config["OBJECTS_POSE_CONFIG"]["max_pos"]),
                rotation=rep.distribution.uniform(config["OBJECTS_POSE_CONFIG"]["min_rot"], config["OBJECTS_POSE_CONFIG"]["max_rot"])
            )
        return object_prims.node
    rep.randomizer.register(move_objects)

def register_lights_placement():
    def randomize_lights():
        lights = rep.create.light(
            light_type="distant",
            color=rep.distribution.uniform(config["LIGHT_CONFIG"]["min_color"], config["LIGHT_CONFIG"]["max_color"]),
            intensity=rep.distribution.uniform(config["LIGHT_CONFIG"]["min_intensity"], config["LIGHT_CONFIG"]["max_intensity"]),
            position=rep.distribution.uniform(config["LIGHT_CONFIG"]["min_pos"], config["LIGHT_CONFIG"]["max_pos"]),
            scale=1.,
            count=3,
        )
        return lights.node
    rep.randomizer.register(randomize_lights)
    
def register_groundplane_colors():
    def randomize_groundplane_colors():
        object_prims = rep.get.prims(path_pattern="/World/GroundPlane")
        with object_prims:
            rep.randomizer.color(colors=rep.distribution.uniform(config["COLOR_RANDOMIZER"]["min_color"], config["COLOR_RANDOMIZER"]["max_color"]))
        return object_prims.node
    rep.randomizer.register(randomize_groundplane_colors)

register_move_objects()
register_lights_placement()
register_groundplane_colors()


# Get the writer from the registry and initialize it with the given config parameters
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(**config["WRITER_CONFIG"])

# Attach writer to the render product
writer.attach(sdg_camera_render_product)

# Setup the randomizations to be triggered every frame
with rep.trigger.on_frame():
    rep.randomizer.move_objects()
    rep.randomizer.randomize_lights()
    rep.randomizer.randomize_groundplane_colors()

sdg_camera_render_product.hydra_texture.set_updates_enabled(True)

# Start the SDG
print("Running SDG for {} frames.".format(config["NUM_FRAMES"]))
for i in range(config["NUM_FRAMES"]):
    rep.orchestrator.step(delta_time=0.0)

# Cleanup writer and render products
writer.detach()
sdg_camera_render_product.destroy()

# Wait for the data to be written to disk
rep.orchestrator.wait_until_complete()

while simulation_app.is_running():
    simulation_app.update()
simulation_app.close()