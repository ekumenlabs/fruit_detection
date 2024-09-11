from datetime import datetime

import carb
from isaacsim import SimulationApp
import numpy as np

HEADLESS=False
SIMULATION_APP_CONFIG={
    "renderer": "RayTracedLighting",
    "headless": HEADLESS,
    "anti_aliasing": "FXAA",
}
SEMANTIC_OBJECTS={
    "Apple": {
        "url": "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Food/Fruit/Apple.usd",
        "class": "apple",
        "prim": "/World/Apple",
    },
    "Avocado": {
        "url": "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Food/Fruit/Avocado01.usd",
        "class": "avocado",
        "prim": "/World/Avocado",
    },
    "Lime": {
        "url": "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Food/Fruit/Lime01.usd",
        "class": "lime",
        "prim": "/World/Lime",
    },
}
OBJECTS_POSE_CONFIG={
    "min_pos": (-2., -2., 0.5),
    "max_pos": (2., 2., 0.5),
    "min_rot": (-180., -90., -180.),
    "max_rot": (180., 90, 180.),
}
OBJECTS_SCALE=(0.1, 0.1, 0.1)
LIGHT_CONFIG={
    "min_color": (0.5, 0.5, 0.5), # gray
    "max_color": (0.9, 0.9, 0.9), # almost white
    "min_distant_intensity": 500.,
    "max_distant_intensity": 900.,
    "min_sphere_intensity": 100000.,
    "max_sphere_intensity": 500000.,
    "min_cylinder_intensity": 100000.,
    "max_cylinder_intensity": 500000.,
    "min_pos": (-5., -5., 10.),
    "max_pos": (5., 5., 20.),
    "min_temperature": 2000., # warm light, indoor home - like light and some outdoor places
    "max_temperature": 7000., # cold light, indoor commercial light type
    "min_exposure": 0., # intensity = 2^exposure * intesity, setting to 0. disables the effect.
    "max_exposure": 0.,
}
SDG_CAMERA={
    "width": 640,
    "height": 480,
    "name": "sdg_camera",
    "pos": (0., 0., 5.),
    "rot": (0., -90., 0.),
    "focal_length": 2.1,
    "focus_distance": 5.5,
    "f_stop": 200,
    "horizontal_aperture": 5.856,
    "vertical_aperture": 3.276,
    "clipping_range": (0.01, 10000000),
    "projection_type": "pinhole",
}
stamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
WRITER_CONFG={
    "output_dir": f"/root/isaac_ws/datasets/{stamp_str}_out_fruit_sdg",
    "rgb": True,
    "bounding_box_2d_tight": True,
}
NUM_FRAMES=300

simulation_app = SimulationApp(launch_config=SIMULATION_APP_CONFIG)


# Late import of runtime modules (the SimulationApp needs to be created before loading the modules)
import omni.replicator.core as rep
# Custom util functions for the example
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.utils import prims
from omni.isaac.core.utils.semantics import remove_all_semantics
from omni.isaac.core.utils.stage import get_current_stage, create_new_stage
from omni.isaac.nucleus import get_assets_root_path

# Configure replicator settings.
rep.settings.carb_settings("/omni/replicator/RTSubframes", 3)

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
ground_plane = GroundPlane(prim_path="/World/GroundPlane", size=10, color=np.array([1., 1., 1.]))

# Spawn an apple in a random pose.
apple_prim = prims.create_prim(
    prim_path=SEMANTIC_OBJECTS["Apple"]["prim"],
    position=(-0.1, -0.05, 0.5),
    orientation=(1., 0., 0., 0.),
    scale=OBJECTS_SCALE,
    usd_path=SEMANTIC_OBJECTS["Apple"]["url"],
    semantic_label=SEMANTIC_OBJECTS["Apple"]["class"],
)

# Spawn an avocado in a random pose.
avocado_prim = prims.create_prim(
    prim_path=SEMANTIC_OBJECTS["Avocado"]["prim"],
    position=(0, 0.1, 0.5),
    orientation=(1., 0., 0., 0.),
    scale=OBJECTS_SCALE,
    usd_path=SEMANTIC_OBJECTS["Avocado"]["url"],
    semantic_label=SEMANTIC_OBJECTS["Avocado"]["class"],
)

# Spawn a lime in a random pose.
lime_prim = prims.create_prim(
    prim_path=SEMANTIC_OBJECTS["Lime"]["prim"],
    position=(0.1, -0.05, 0.5),
    orientation=(1., 0., 0., 0.),
    scale=OBJECTS_SCALE,
    usd_path=SEMANTIC_OBJECTS["Lime"]["url"],
    semantic_label=SEMANTIC_OBJECTS["Lime"]["class"],
)

# Create the camera used for the acquisition.
sdg_camera = rep.create.camera(
    name=SDG_CAMERA["name"],
    position=SDG_CAMERA["pos"],
    rotation=SDG_CAMERA["rot"],
    focal_length=SDG_CAMERA["focal_length"],
    focus_distance=SDG_CAMERA["focus_distance"],
    f_stop=SDG_CAMERA["f_stop"],
    horizontal_aperture=SDG_CAMERA["horizontal_aperture"],
    clipping_range=SDG_CAMERA["clipping_range"],
    projection_type=SDG_CAMERA["projection_type"],
    count=1,
)
sdg_camera_render_product = rep.create.render_product(
    sdg_camera, (SDG_CAMERA["width"], SDG_CAMERA["height"]), name="SdgCameraView"
)
sdg_camera_render_product.hydra_texture.set_updates_enabled(False)

def register_move_objects():
    def move_objects():
        object_prims = rep.get.prims(semantics=[
            ("class", SEMANTIC_OBJECTS["Apple"]["class"]),
            ("class", SEMANTIC_OBJECTS["Avocado"]["class"]),
            ("class", SEMANTIC_OBJECTS["Lime"]["class"]),
        ])
        with object_prims:
            rep.modify.pose(
                position=rep.distribution.uniform(OBJECTS_POSE_CONFIG["min_pos"], OBJECTS_POSE_CONFIG["max_pos"]),
                rotation=rep.distribution.uniform(OBJECTS_POSE_CONFIG["min_rot"], OBJECTS_POSE_CONFIG["max_rot"])
            )
        return object_prims.node
    rep.randomizer.register(move_objects)


def register_lights():
    def create_light_node(type: str):
        light = rep.create.light(
            light_type=type,
            color=rep.distribution.uniform(LIGHT_CONFIG["min_color"], LIGHT_CONFIG["max_color"]),
            intensity=rep.distribution.uniform(LIGHT_CONFIG[f"min_{type}_intensity"], LIGHT_CONFIG[f"max_{type}_intensity"]),
            position=rep.distribution.uniform(LIGHT_CONFIG["min_pos"], LIGHT_CONFIG["max_pos"]),
            temperature=rep.distribution.uniform(LIGHT_CONFIG["min_temperature"], LIGHT_CONFIG["max_temperature"]),
            exposure=rep.distribution.uniform(LIGHT_CONFIG["min_exposure"], LIGHT_CONFIG["max_exposure"]),
            scale=1.,
            count=1,
        )
        return light.node

    def randomize_distant_light():
        return create_light_node("distant")

    def randomize_cylinder_light():
        return create_light_node("cylinder")

    def randomize_sphere_light():
        return create_light_node("sphere")

    rep.randomizer.register(randomize_distant_light)
    rep.randomizer.register(randomize_cylinder_light)
    rep.randomizer.register(randomize_sphere_light)
    
def register_groundplane_colors():
    def randomize_groundplane_colors():
        object_prims = rep.get.prims(path_pattern="/World/GroundPlane")
        with object_prims:
            rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))
        return object_prims.node
    rep.randomizer.register(randomize_groundplane_colors)

register_move_objects()
register_lights()
register_groundplane_colors()


# Get the writer from the registry and initialize it with the given config parameters
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(**WRITER_CONFG)

# Attach writer to the render product
writer.attach(sdg_camera_render_product)

# Setup the randomizations to be triggered every frame
with rep.trigger.on_frame():
    rep.randomizer.randomize_distant_light()
    rep.randomizer.randomize_cylinder_light()
    rep.randomizer.randomize_sphere_light()
    rep.randomizer.move_objects()
    rep.randomizer.randomize_groundplane_colors()

sdg_camera_render_product.hydra_texture.set_updates_enabled(True)

# Start the SDG
print(f"Running SDG for {NUM_FRAMES} frames")
for i in range(NUM_FRAMES):
    rep.orchestrator.step(delta_time=0.0)

# Cleanup writer and render products
writer.detach()
sdg_camera_render_product.destroy()

# Wait for the data to be written to disk
rep.orchestrator.wait_until_complete()

while simulation_app.is_running():
    simulation_app.update()
simulation_app.close()