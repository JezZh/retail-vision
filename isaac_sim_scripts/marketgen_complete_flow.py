"""Track K footfall on the open ConvenienceStore01 stage.

Runs inside the live Isaac Sim 6.0.1 session. The open stage is edited in
place. This script never opens, reloads, resets, or saves the USD layer.

``isaacsim.replicator.agent.core`` setup_simulation() reopens the base stage
and would wipe the session (the cameras, the softened sun, and the navmesh
are not saved). Characters are spawned through that extension's payload,
behavior-agent, and IRACharacterAPI commands on the stage that is already
open. Their timed routes are issued on the behavior agents, because a shared
wander/patrol routine cannot express this store's clock.

The export is centimeters, Z-up. Walk speeds are meters per second converted
with metersPerUnit. Path targets are stage centimeters on the baked navmesh.
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import subprocess
from dataclasses import dataclass, field

import carb
import carb.settings
import omni.anim.behavior.core as bh
import omni.client
import omni.kit.app
import omni.kit.commands
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

OUTPUT_DIR = "isaac_sim_scripts"
VIDEO_DIR = "demo_videos"
STAGE_TOKEN = "ConvenienceStore01.usd"
WORLD_ROOT = "/World"
STORE_ROOT = "/Root"
FRAMEWORK = "/Root/SupermarketFramework"
CHARACTERS_ROOT = "/World/Characters"
MOTION_LIBRARY = "/World/Characters/HumanMotionLibrary"

CHECKOUT_CAMERA = "/World/CCTV_Checkout_Shelves"
ENTRANCE_CAMERA = "/World/CCTV_Entrance_Aisle"
ENTRANCE_END_CAMERA = "/World/CCTV_Entrance_Aisle_End"
BACK_CAMERA = "/World/CCTV_Back_Aisle"
LEFT_BACK_CAMERA = "/World/CCTV_Back_Aisle_Left"
CAMERAS = (
    CHECKOUT_CAMERA,
    ENTRANCE_CAMERA,
    BACK_CAMERA,
    LEFT_BACK_CAMERA,
    ENTRANCE_END_CAMERA,
)
VIDEO_FILES = {
    CHECKOUT_CAMERA: "cam_checkout.mp4",
    ENTRANCE_CAMERA: "cam_entrance.mp4",
    BACK_CAMERA: "cam_back_aisle.mp4",
    LEFT_BACK_CAMERA: "cam_back_aisle_left.mp4",
    ENTRANCE_END_CAMERA: "cam_entrance_end.mp4",
}

FOCAL_LENGTH_MM = 18.0
HORIZONTAL_APERTURE_MM = 20.955
VERTICAL_APERTURE_MM = 15.2908
F_STOP = 11.0
# Stage origin is the front-left corner. Y grows out the front door, so the
# store interior is negative Y. Units are centimeters.
# The eye sits in the back-right corner, just behind ShelfBox3 (that shelf
# ends at y=-1112) and in front of the back wall (y=-1200), 2.75 m up.
# rotateXYZ (-36, 0, -180) aims toward +Y but rolls the camera: its up axis
# points downward, so the picture is sideways into the shelf ends. The mount
# below keeps the 36° downward pitch, faces the front, and stays upright.
BACK_EYE_CM = (1120.0, -1155.0, 275.0)
# The right mount is 1.8 m in from the right wall. The main room's left
# boundary at the back is the partition at x=300, not the front left wall,
# because the back-left corner is the service notch. 0.5 m in from that
# partition, same height and same distance off the back wall.
LEFT_BACK_EYE_CM = (430.0, -1040.0, 275.0)
LEFT_BACK_YAW_DEG = 16.0
SUN_INTENSITY = 1193.6619873046875
CEILING_RECT_INTENSITY = 15000.0
CEILING_RECTS = (
    ("CeilingLightMesh_1_1", "RectLight"),
    ("CeilingLightMesh_1_2", "RectLight_01"),
    ("CeilingLightMesh_2_1", "RectLight_02"),
    ("CeilingLightMesh_2_2", "RectLight_03"),
    ("CeilingLightMesh_3_1", "RectLight_04"),
    ("CeilingLightMesh_3_2", "RectLight_05"),
)
BACK_FOCAL_LENGTH_MM = 16.0
BACK_HORIZONTAL_APERTURE_MM = 36.0
# Recordings are 16:9. A 36 mm horizontal gate needs 20.25 mm vertical or the
# picture stretches. The calibration named only the horizontal gate.
BACK_VERTICAL_APERTURE_MM = 20.25
BACK_FORWARD_DEPRESSION_DEG = 36.0
ENTRANCE_DEPRESSION_DEG = 48.0
# Far end of the front wall, past the door and the front shelf, just left of
# the front-right pillar. Same height and look-into-the-shop aim as the door camera.
ENTRANCE_END_EYE_CM = (1140.0, -18.0, 312.0)
ENTRANCE_END_FORWARD = (0.03, -0.668, -0.743)
# 0.20 m is the corridor minimum the take asks for. These bodies report
# get_radius() of about 23–26 cm, and a navmesh whose maximum is 20 cm
# accepts move_to without ever moving them. The maximum stays 26 cm.
AGENT_RADIUS_CM = 20.0
AGENT_MAX_RADIUS_CM = 26.0
SEPARATION_WEIGHT = 2.5
# Recast accepts a span only when its up-component is strictly greater than
# cos(maxSlope). At 0 degrees that cutoff is 1, so the flat floor bakes as an
# empty mesh. 0.05 degrees is the smallest value that still connected the
# aisle and the exterior exit, and it does not admit a ramp.
AGENT_MAX_SLOPE_DEG = 0.05
# Exterior parking point, meters (-6, 0, 0). Stage units are centimeters.
EXIT_NODE_CM = (-600.0, 0.0, 0.0)
NAVMESH_INCLUDE = "/World/NavMesh_StoreFloor"
NAVMESH_MIN_RADIUS = "/exts/omni.anim.navigation.core/navMesh/config/agentMinRadius"
NAVMESH_MAX_RADIUS = "/exts/omni.anim.navigation.core/navMesh/config/agentMaxRadius"
NAVMESH_MAX_SLOPE = "/exts/omni.anim.navigation.core/navMesh/config/agentMaxFloorSlope"

# The people pack shipped with Isaac Sim 6.0. Occupational meshes, not
# casual shoppers. original_* folders repeat the same bodies.
CHARACTER_ROOT_URL = "Isaac/People/Characters/"
MOTION_LIBRARY_URL = "Isaac/People/MotionLibrary/HumanMotionLibrary.usd"
SAMPLE_SEED = 601
CASHIER_FOLDER = "F_Business_02"
STOCK_FOLDER = "male_adult_construction_01_new"

WALK_MPS = 1.15
BRISK_MPS = 1.45
RECORD_SECONDS = 90.0
FPS = 30
VIDEO_SIZE = (1280, 720)
DWELL_SECONDS = 20.0
BEVERAGE_DWELL_SECONDS = 15.0

MILK_ALERT = "[TRACK_K_ALERT][SHELF] 5 items removed from Row_1_Tier_2. Current Stock: 0."
CHIPS_ALERT = "[TRACK_K_ALERT][SHELF] 10 items removed from Row_2_Tier_1. Current Stock: 1."
QUEUE_ALERT = "[TRACK_K_ALERT][QUEUE] Queue threshold broken. Length: 4. Est. wait time: 3.5 mins."

# Primary till is the register on SM_Counter_Front. There is no
# SM_Checkout_Counter_03a on this stage. The second till is further down
# the same counter (SM_CashRegister_Out2).
PRIMARY_REGISTER = "/Root/BP_CashierArea_01/SM_CashRegister_Out_StaticMeshComponent0"
BACKUP_REGISTER = "/Root/BP_CashierArea_01/SM_CashRegister_Out2_StaticMeshComponent0"
DAIRY_SHELF = "/Root/ShelfBox7"
SNACK_SHELF = "/Root/ShelfBox6"
BEVERAGE_CASE = "/Root/ShelfBox5"


def _stage() -> Usd.Stage:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open in Isaac Sim.")
    identifier = stage.GetRootLayer().identifier.replace("\\", "/")
    if STAGE_TOKEN not in identifier:
        raise RuntimeError(
            f"The open stage is {identifier}. Refusing to edit it. "
            f"Open {STAGE_TOKEN} and run this script again without reloading."
        )
    return stage


def _range(stage: Usd.Stage, prim: Usd.Prim) -> Gf.Range3d:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        True,
    )
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if bounds.IsEmpty():
        raise RuntimeError(f"No world bounds for {prim.GetPath()}")
    return bounds


def _mm_to_tenths(millimeters: float, meters_per_unit: float) -> float:
    tenth_mm = meters_per_unit * 100.0
    if tenth_mm <= 0.0:
        raise RuntimeError(f"Invalid metersPerUnit {meters_per_unit}")
    return millimeters / tenth_mm


def _normalize(vector: Gf.Vec3d) -> Gf.Vec3d:
    length = vector.GetLength()
    if length < 1e-6:
        raise RuntimeError("Cannot normalize a zero vector.")
    return vector / length


def _look_matrix(eye: Gf.Vec3d, target: Gf.Vec3d) -> Gf.Matrix4d:
    """Camera-to-world. USD cameras look down local -Z, with +Y up."""
    forward = _normalize(target - eye)
    up = Gf.Vec3d(0.0, 0.0, 1.0)
    z_axis = -forward
    x_axis = _normalize(Gf.Cross(forward, up))
    y_axis = _normalize(Gf.Cross(z_axis, x_axis))
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRow(0, Gf.Vec4d(x_axis[0], x_axis[1], x_axis[2], 0.0))
    matrix.SetRow(1, Gf.Vec4d(y_axis[0], y_axis[1], y_axis[2], 0.0))
    matrix.SetRow(2, Gf.Vec4d(z_axis[0], z_axis[1], z_axis[2], 0.0))
    matrix.SetRow(3, Gf.Vec4d(eye[0], eye[1], eye[2], 1.0))
    return matrix


def _camera_axis(prim: Usd.Prim, local_axis: Gf.Vec3d) -> Gf.Vec3d:
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return _normalize(matrix.TransformDir(local_axis))


def _camera_forward(prim: Usd.Prim) -> Gf.Vec3d:
    return _camera_axis(prim, Gf.Vec3d(0.0, 0.0, -1.0))


def _depression_of(forward: Gf.Vec3d) -> float:
    horizontal = math.hypot(forward[0], forward[1])
    return math.degrees(math.atan2(-forward[2], horizontal))


def _point_on_depression(eye: Gf.Vec3d, aim_xy: Gf.Vec2d, target_z: float, depression_deg: float) -> Gf.Vec3d:
    direction = _normalize(Gf.Vec3d(aim_xy[0] - eye[0], aim_xy[1] - eye[1], 0.0))
    drop = eye[2] - target_z
    if drop <= 0.0:
        raise RuntimeError("Camera is not above its subject.")
    horizontal = drop / math.tan(math.radians(depression_deg))
    return Gf.Vec3d(eye[0] + direction[0] * horizontal, eye[1] + direction[1] * horizontal, target_z)


def _ceiling_z(stage: Usd.Stage) -> float:
    framework = stage.GetPrimAtPath(FRAMEWORK)
    walls = [child for child in framework.GetChildren() if "Wall" in child.GetName()]
    if not walls:
        raise RuntimeError("No wall components to measure the ceiling.")
    return max(_range(stage, wall).GetMax()[2] for wall in walls)


def _camera_pose(stage: Usd.Stage, path: str) -> dict:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing camera {path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    eye = matrix.ExtractTranslation()
    forward = _camera_forward(prim)
    focal = UsdGeom.Camera(prim).GetFocalLengthAttr().Get()
    return {
        "eye": (float(eye[0]), float(eye[1]), float(eye[2])),
        "depression": _depression_of(forward),
        "focal": float(focal) if focal is not None else None,
    }


def _solid_at(stage: Usd.Stage, point: Gf.Vec3d) -> str | None:
    """Name a leaf mesh whose bounds contain the point.

    Group prims such as SupermarketFramework have a bounds box that fills the
    shop, including empty air. Only mesh leaves count as a collision.
    """
    root = stage.GetPrimAtPath(STORE_ROOT)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], True)

    def visit(prim: Usd.Prim) -> str | None:
        if prim.GetName() in ("Lighting", "Floor"):
            return None
        if prim.IsA(UsdGeom.Mesh):
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if box.IsEmpty():
                return None
            size = box.GetSize()
            if max(float(size[0]), float(size[1]), float(size[2])) > 2000.0:
                return None
            minimum, maximum = box.GetMin(), box.GetMax()
            if (
                minimum[0] <= point[0] <= maximum[0]
                and minimum[1] <= point[1] <= maximum[1]
                and minimum[2] <= point[2] <= maximum[2]
            ):
                return prim.GetName()
            return None
        for child in prim.GetChildren():
            hit = visit(child)
            if hit:
                return hit
        return None

    for child in root.GetChildren():
        hit = visit(child)
        if hit:
            return hit
    return None


def ensure_shop_lights(stage: Usd.Stage) -> dict:
    """Match export20: full authored sun, plus a 15000 rect light on each ceiling panel."""
    sun = stage.GetPrimAtPath("/Root/Lighting/DirectionalLight")
    if not sun or not sun.IsValid():
        raise RuntimeError("Missing /Root/Lighting/DirectionalLight")
    for name in ("inputs:intensity", "intensity"):
        attr = sun.GetAttribute(name)
        if attr and attr.IsValid() and attr.HasValue():
            attr.Set(SUN_INTENSITY)
    sun.SetCustomDataByKey("marketgen:originalIntensity", SUN_INTENSITY)
    made = []
    orient = Gf.Quatf(0.70710677, Gf.Vec3f(0.0, 0.0, 0.70710677))
    for mesh_name, light_name in CEILING_RECTS:
        parent = f"{FRAMEWORK}/{mesh_name}"
        if not stage.GetPrimAtPath(parent):
            raise RuntimeError(f"Missing ceiling fixture {parent}")
        path = f"{parent}/{light_name}"
        light = UsdLux.RectLight.Define(stage, path)
        light.CreateIntensityAttr(CEILING_RECT_INTENSITY)
        light.CreateWidthAttr(100.0)
        light.CreateHeightAttr(100.0)
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        light.CreateSpecularAttr(1.0)
        light.CreateDiffuseAttr(1.0)
        light.CreateExposureAttr(0.0)
        light.CreateNormalizeAttr(False)
        light.CreateEnableColorTemperatureAttr(False)
        xform = UsdGeom.Xformable(light.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(2.8421709430404007e-13, 1.2505552149377763e-12, -2.1572489773768666)
        )
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(orient)
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(0.9, 0.9, 1.0))
        made.append(path)
    return {"sun": SUN_INTENSITY, "rects": made}


def _place_aisle_camera(
    stage: Usd.Stage,
    meters_per_unit: float,
    path: str,
    eye_cm: tuple[float, float, float],
    role: str,
    yaw_deg: float = 0.0,
) -> dict:
    """Mount one upright aisle camera, 36° down toward the front door."""
    if not stage.GetPrimAtPath(WORLD_ROOT):
        UsdGeom.Xform.Define(stage, WORLD_ROOT)
    eye = Gf.Vec3d(*eye_cm)
    blocker = _solid_at(stage, eye)
    if blocker:
        raise RuntimeError(f"{path} mount is inside {blocker} at {eye}.")
    pitch = math.radians(BACK_FORWARD_DEPRESSION_DEG)
    yaw = math.radians(yaw_deg)
    forward = _normalize(
        Gf.Vec3d(
            math.sin(yaw) * math.cos(pitch),
            math.cos(yaw) * math.cos(pitch),
            -math.sin(pitch),
        )
    )
    matrix = _look_matrix(eye, eye + forward)
    for step_cm in (20.0, 50.0):
        sample = eye + forward * step_cm
        lens = _solid_at(stage, sample)
        if lens:
            raise RuntimeError(f"{path} lens looks into {lens} at {sample}.")
    existing = stage.GetPrimAtPath(path)
    if existing and existing.IsValid():
        omni.kit.commands.execute("DeletePrims", paths=[path])
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    focal = _mm_to_tenths(BACK_FOCAL_LENGTH_MM, meters_per_unit)
    horizontal = _mm_to_tenths(BACK_HORIZONTAL_APERTURE_MM, meters_per_unit)
    vertical = _mm_to_tenths(BACK_VERTICAL_APERTURE_MM, meters_per_unit)
    camera.CreateFocalLengthAttr(focal)
    camera.CreateHorizontalApertureAttr(horizontal)
    camera.CreateVerticalApertureAttr(vertical)
    camera.CreateFStopAttr(F_STOP)
    camera.CreateFocusDistanceAttr(800.0)
    near = 0.05 / meters_per_unit
    far = 200.0 / meters_per_unit
    camera.CreateClippingRangeAttr(Gf.Vec2f(near, far))
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(matrix)
    forward = _camera_forward(camera.GetPrim())
    up = _camera_axis(camera.GetPrim(), Gf.Vec3d(0.0, 1.0, 0.0))
    depression = _depression_of(forward)
    if abs(depression - BACK_FORWARD_DEPRESSION_DEG) > 1.0:
        raise RuntimeError(f"{path} depression is {depression:.2f}, expected {BACK_FORWARD_DEPRESSION_DEG}.")
    if forward[1] <= 0.7 or up[2] <= 0.5:
        raise RuntimeError(f"{path} is rolled or not facing the front. Forward {forward}, up {up}.")
    read_focal = camera.GetFocalLengthAttr().Get()
    read_horizontal = camera.GetHorizontalApertureAttr().Get()
    if read_focal is None or abs(float(read_focal) - focal) > 0.05:
        raise RuntimeError(f"{path} focal length read back {read_focal}, expected {focal}.")
    if read_horizontal is None or abs(float(read_horizontal) - horizontal) > 0.05:
        raise RuntimeError(f"{path} horizontal aperture read back {read_horizontal}, expected {horizontal}.")
    target = eye + forward * 400.0
    camera.GetPrim().SetCustomDataByKey("marketgen:role", role)
    return {
        "path": path,
        "eye_cm": [round(eye[0], 1), round(eye[1], 1), round(eye[2], 1)],
        "forward": [round(float(v), 3) for v in forward],
        "up": [round(float(v), 3) for v in up],
        "target_cm": [round(target[0], 1), round(target[1], 1), round(target[2], 1)],
        "depression_deg": round(depression, 2),
        "focal_length_mm": float(read_focal),
        "horizontal_aperture_mm": float(read_horizontal),
    }


def place_back_camera(stage: Usd.Stage, meters_per_unit: float) -> dict:
    """Mount the back-right aisle camera, looking toward the front door."""
    return _place_aisle_camera(
        stage,
        meters_per_unit,
        BACK_CAMERA,
        BACK_EYE_CM,
        "Back-right aisle CCTV at 11.2 m, -11.55 m, 2.75 m. Upright, 36 degrees down the aisle. 16 mm on a 36 mm gate.",
    )


def place_left_back_camera(stage: Usd.Stage, meters_per_unit: float) -> dict:
    """Mount the same aisle camera on the left side of the main room."""
    return _place_aisle_camera(
        stage,
        meters_per_unit,
        LEFT_BACK_CAMERA,
        LEFT_BACK_EYE_CM,
        "Back-left aisle CCTV at 4.3 m, -10.4 m, 2.75 m. Upright, 36 degrees down, yawed 16 degrees toward the gondolas. 16 mm on a 36 mm gate.",
        LEFT_BACK_YAW_DEG,
    )


def place_entrance_end_camera(stage: Usd.Stage, meters_per_unit: float) -> dict:
    """Mount the entrance camera's twin at the far end of the front wall."""
    eye = Gf.Vec3d(*ENTRANCE_END_EYE_CM)
    blocker = _solid_at(stage, eye)
    if blocker:
        raise RuntimeError(f"{ENTRANCE_END_CAMERA} mount is inside {blocker} at {eye}.")
    forward = _normalize(Gf.Vec3d(*ENTRANCE_END_FORWARD))
    for step_cm in (30.0, 80.0):
        sample = eye + forward * step_cm
        lens = _solid_at(stage, sample)
        if lens:
            raise RuntimeError(f"{ENTRANCE_END_CAMERA} lens looks into {lens} at {sample}.")
    source = UsdGeom.Camera(stage.GetPrimAtPath(ENTRANCE_CAMERA))
    camera = UsdGeom.Camera.Define(stage, ENTRANCE_END_CAMERA)
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    camera.CreateFocalLengthAttr(float(source.GetFocalLengthAttr().Get() or 10.0))
    camera.CreateHorizontalApertureAttr(float(source.GetHorizontalApertureAttr().Get() or HORIZONTAL_APERTURE_MM))
    camera.CreateVerticalApertureAttr(float(source.GetVerticalApertureAttr().Get() or VERTICAL_APERTURE_MM))
    camera.CreateFStopAttr(float(source.GetFStopAttr().Get() or F_STOP))
    camera.CreateFocusDistanceAttr(float(source.GetFocusDistanceAttr().Get() or 800.0))
    clip = source.GetClippingRangeAttr().Get()
    if clip is None:
        clip = Gf.Vec2f(0.05 / meters_per_unit, 200.0 / meters_per_unit)
    camera.CreateClippingRangeAttr(clip)
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(_look_matrix(eye, eye + forward))
    posed_forward = _camera_forward(camera.GetPrim())
    depression = _depression_of(posed_forward)
    if abs(depression - ENTRANCE_DEPRESSION_DEG) > 1.0 or posed_forward[1] > -0.5:
        raise RuntimeError(f"{ENTRANCE_END_CAMERA} is not looking into the shop. Forward {posed_forward}.")
    camera.GetPrim().SetCustomDataByKey(
        "marketgen:role",
        "Front-wall CCTV at the right end of the entrance wall. Same 48 degree look into the shop as the door camera.",
    )
    return {
        "path": ENTRANCE_END_CAMERA,
        "eye_cm": [round(float(v), 1) for v in eye],
        "depression_deg": round(depression, 2),
        "focal_length_mm": float(camera.GetFocalLengthAttr().Get()),
    }


def tilt_entrance_camera(stage: Usd.Stage) -> dict:
    """Add 12° of downward tilt. The eye, yaw, and 18 mm lens stay put."""
    prim = stage.GetPrimAtPath(ENTRANCE_CAMERA)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing {ENTRANCE_CAMERA}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    eye = Gf.Vec3d(matrix.ExtractTranslation())
    forward = _camera_forward(prim)
    horizontal = math.hypot(forward[0], forward[1])
    if horizontal < 1e-6:
        raise RuntimeError("Entrance camera has no horizontal aim.")
    depression = math.radians(ENTRANCE_DEPRESSION_DEG)
    aimed = _normalize(
        Gf.Vec3d(
            forward[0] / horizontal * math.cos(depression),
            forward[1] / horizontal * math.cos(depression),
            -math.sin(depression),
        )
    )
    target = eye + aimed * 500.0
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(_look_matrix(eye, target))
    posed = _camera_pose(stage, ENTRANCE_CAMERA)
    moved = math.dist(posed["eye"], (eye[0], eye[1], eye[2]))
    if moved > 0.5:
        raise RuntimeError(f"Entrance camera eye moved {moved:.2f} cm.")
    if abs(posed["depression"] - ENTRANCE_DEPRESSION_DEG) > 0.6:
        raise RuntimeError(f"Entrance depression is {posed['depression']:.2f}, expected {ENTRANCE_DEPRESSION_DEG}.")
    if posed["focal"] is None or abs(posed["focal"] - FOCAL_LENGTH_MM) > 0.1:
        raise RuntimeError(f"Entrance focal length changed to {posed['focal']}.")
    return {
        "path": ENTRANCE_CAMERA,
        "eye_cm": [round(v, 1) for v in posed["eye"]],
        "depression_deg": round(posed["depression"], 2),
        "focal_length_mm": posed["focal"],
    }


def _fit_nav_volume(prim: Usd.Prim, minimum: Gf.Vec3d, maximum: Gf.Vec3d, meters_per_unit: float) -> None:
    half = 0.5 / meters_per_unit
    extent = UsdGeom.Boundable(prim).CreateExtentAttr()
    extent.Set([Gf.Vec3f(-half, -half, -half), Gf.Vec3f(half, half, half)])
    local_size = 2.0 * half
    size = maximum - minimum
    center = (minimum + maximum) * 0.5
    scale = Gf.Vec3d(size[0] / local_size, size[1] / local_size, size[2] / local_size)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(center[0], center[1], center[2]))
    xform.AddScaleOp().Set(scale)


def extend_walkable_volume(stage: Usd.Stage, meters_per_unit: float) -> dict:
    """Keep the notch exclude and pull the include volume out to the exterior exit."""
    prim = stage.GetPrimAtPath(NAVMESH_INCLUDE)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing {NAVMESH_INCLUDE}. Run step1 on this open stage first.")
    # x=-720 covers the exterior parking point at x=-600. y=80 crosses the door.
    # The service-notch exclude volume is left exactly as step1 baked it.
    minimum = Gf.Vec3d(-720.0, -1205.0, -20.0)
    maximum = Gf.Vec3d(1305.0, 80.0, 220.0)
    _fit_nav_volume(prim, minimum, maximum, meters_per_unit)
    return {
        "include": NAVMESH_INCLUDE,
        "min_cm": [minimum[0], minimum[1], minimum[2]],
        "max_cm": [maximum[0], maximum[1], maximum[2]],
    }


def set_tight_agent() -> dict:
    """Minimum agent radius 0.20 m. Maximum stays 0.26 m so these bodies can path."""
    settings = carb.settings.get_settings()
    for path in (NAVMESH_MAX_RADIUS, "/persistent" + NAVMESH_MAX_RADIUS):
        settings.set(path, AGENT_MAX_RADIUS_CM)
    for path in (NAVMESH_MIN_RADIUS, "/persistent" + NAVMESH_MIN_RADIUS):
        settings.set(path, AGENT_RADIUS_CM)
    for path in (NAVMESH_MAX_SLOPE, "/persistent" + NAVMESH_MAX_SLOPE):
        settings.set(path, AGENT_MAX_SLOPE_DEG)
    report = {
        "min_cm": float(settings.get(NAVMESH_MIN_RADIUS)),
        "max_cm": float(settings.get(NAVMESH_MAX_RADIUS)),
        "slope_deg": float(settings.get(NAVMESH_MAX_SLOPE)),
    }
    if abs(report["min_cm"] - AGENT_RADIUS_CM) > 0.01 or abs(report["max_cm"] - AGENT_MAX_RADIUS_CM) > 0.01:
        raise RuntimeError(f"Agent radius did not stick: {report}")
    if abs(report["slope_deg"] - AGENT_MAX_SLOPE_DEG) > 0.01:
        raise RuntimeError(f"Agent max slope did not stick: {report}")
    return report


async def bake_navmesh() -> dict:
    import omni.anim.navigation.core as nav

    carb.settings.get_settings().set("/exts/omni.anim.navigation.core/navMesh/bakeWaitTimeoutSeconds", 0.0)
    carb.settings.get_settings().set("/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
    app = omni.kit.app.get_app()
    await app.next_update_async()
    await app.next_update_async()
    interface = nav.acquire_interface()
    baked = bool(interface.start_navmesh_baking_and_wait())
    navmesh = interface.get_navmesh()
    if navmesh is None:
        raise RuntimeError("NavMesh bake did not produce a mesh.")
    return {"baked": baked}


def _probe_nav(navmesh, point: tuple[float, float, float]) -> dict | None:
    result = navmesh.query_closest_point(target=point)
    if not result:
        return None
    hit = result[0]
    if hit is None:
        return None
    delta = Gf.Vec3d(float(hit[0]) - point[0], float(hit[1]) - point[1], float(hit[2]) - point[2])
    return {
        "hit": [round(float(hit[0]), 1), round(float(hit[1]), 1), round(float(hit[2]), 1)],
        "distance_cm": round(delta.GetLength(), 1),
    }


def verify_tight_navmesh() -> dict:
    import omni.anim.navigation.core as nav

    navmesh = nav.acquire_interface().get_navmesh()
    if navmesh is None:
        raise RuntimeError("NavMesh is missing after the bake.")
    probes = {
        "primary_aisle": _probe_nav(navmesh, (700.0, -457.0, 30.0)),
        "dairy_aisle": _probe_nav(navmesh, (760.0, -430.0, 30.0)),
        "snack_aisle": _probe_nav(navmesh, (760.0, -740.0, 30.0)),
        "entry_lane": _probe_nav(navmesh, (420.0, -180.0, 30.0)),
        "front_left_floor": _probe_nav(navmesh, (80.0, -200.0, 30.0)),
        "service_notch": _probe_nav(navmesh, (150.0, -1000.0, 30.0)),
        "exit_node": _probe_nav(navmesh, (EXIT_NODE_CM[0], EXIT_NODE_CM[1], 30.0)),
    }
    for name in ("primary_aisle", "dairy_aisle", "snack_aisle", "entry_lane", "front_left_floor"):
        hit = probes[name]
        if hit is None or hit["distance_cm"] > 80.0:
            raise RuntimeError(
                f"{name} is not walkable at slope {AGENT_MAX_SLOPE_DEG} deg "
                f"and radius {AGENT_RADIUS_CM:.0f}/{AGENT_MAX_RADIUS_CM:.0f} cm: {hit}"
            )
    notch = probes["service_notch"]
    if notch is not None and notch["distance_cm"] < 50.0:
        raise RuntimeError(f"Service notch was baked as walkable: {notch}")
    exit_hit = probes["exit_node"]
    if exit_hit is None or exit_hit["distance_cm"] > 120.0:
        raise RuntimeError(f"Exit node {EXIT_NODE_CM} is not on the navmesh: {exit_hit}")
    start = navmesh.query_closest_point(target=(420.0, -180.0, 20.0))
    end = navmesh.query_closest_point(target=(780.0, -457.0, 20.0))
    door = navmesh.query_closest_point(target=(390.0, -40.0, 20.0))
    outside = navmesh.query_closest_point(target=(EXIT_NODE_CM[0], EXIT_NODE_CM[1], 20.0))
    path = None
    exit_path = None
    if start and end and start[0] is not None and end[0] is not None:
        path = navmesh.query_shortest_path(start_pos=start[0], end_pos=end[0])
    if door and outside and door[0] is not None and outside[0] is not None:
        exit_path = navmesh.query_shortest_path(start_pos=door[0], end_pos=outside[0])

    def _length(found) -> float | None:
        if found is None or not hasattr(found, "length"):
            return None
        return round(float(found.length()), 1)

    aisle_cm = _length(path)
    exit_cm = _length(exit_path)
    if aisle_cm is None:
        raise RuntimeError("No navmesh path from the entry lane to the primary aisle.")
    if exit_cm is None:
        raise RuntimeError("No navmesh path from the front door to the exterior exit node.")
    # Behavior move_to uses each body's own radius. Confirm the largest body can path.
    from omni.anim.navigation.core import NavAgentDesc

    body = NavAgentDesc(radius=AGENT_MAX_RADIUS_CM, height=200.0, collision_gap=0.0)
    body_door = navmesh.query_closest_point(target=(390.0, -40.0, 20.0), agent=body)
    body_aisle = navmesh.query_closest_point(target=(760.0, -430.0, 20.0), agent=body)
    body_exit = navmesh.query_closest_point(target=(EXIT_NODE_CM[0], EXIT_NODE_CM[1], 20.0), agent=body)
    body_path = None
    if body_door and body_aisle and body_door[0] is not None and body_aisle[0] is not None:
        body_path = navmesh.query_shortest_path(start_pos=body_door[0], end_pos=body_aisle[0], agent=body)
    body_exit_path = None
    if body_door and body_exit and body_door[0] is not None and body_exit[0] is not None:
        body_exit_path = navmesh.query_shortest_path(start_pos=body_door[0], end_pos=body_exit[0], agent=body)
    if _length(body_path) is None or _length(body_exit_path) is None:
        raise RuntimeError(
            f"No navmesh path for a {AGENT_MAX_RADIUS_CM:.0f} cm body "
            f"from the door to the aisle or the exit."
        )
    return {"probes": probes, "entry_to_aisle_cm": aisle_cm, "door_to_exit_cm": exit_cm}


def _enable(ext_name: str) -> None:
    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled(ext_name):
        manager.set_extension_enabled_immediate(ext_name, True)


def _resolve_people_root() -> str:
    from omni.metropolis.utils.isaac_sim_util import resolve_asset_path

    url = resolve_asset_path(CHARACTER_ROOT_URL)
    if not url.endswith("/"):
        url += "/"
    return url


def _resolve_motion_library() -> str:
    from omni.metropolis.utils.isaac_sim_util import resolve_asset_path

    return resolve_asset_path(MOTION_LIBRARY_URL)


def _list_character_folders(root_url: str) -> list[str]:
    result, entries = omni.client.list(root_url)
    if result != omni.client.Result.OK:
        raise RuntimeError(f"Cannot list character assets at {root_url}: {result}")
    folders = []
    for entry in entries:
        name = entry.relative_path.strip("/")
        if not name or name.startswith(".") or name.startswith("original_"):
            continue
        if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            folders.append(name)
    if len(folders) < 10:
        raise RuntimeError(f"Need 10 unique character folders, found {folders}")
    return folders


def _usd_in_folder(root_url: str, folder: str) -> str:
    folder_url = root_url + folder + "/"
    result, entries = omni.client.list(folder_url)
    if result != omni.client.Result.OK:
        raise RuntimeError(f"Cannot list {folder_url}: {result}")
    for entry in entries:
        if entry.relative_path.lower().endswith((".usd", ".usda")):
            return folder_url + entry.relative_path
    raise RuntimeError(f"No USD in {folder_url}")


def assign_cast(root_url: str) -> list[tuple[str, str, str]]:
    """Return (role, folder, usd url). Shoppers are a seeded shuffle of the rest."""
    folders = _list_character_folders(root_url)
    if CASHIER_FOLDER not in folders or STOCK_FOLDER not in folders:
        raise RuntimeError(f"Missing cashier or stock mesh in {folders}")
    remaining = [name for name in folders if name not in (CASHIER_FOLDER, STOCK_FOLDER)]
    rng = random.Random(SAMPLE_SEED)
    rng.shuffle(remaining)
    shoppers = remaining[:8]
    if len(shoppers) < 8:
        raise RuntimeError("Not enough unique meshes for 8 shoppers.")
    cast = [("Staff_01", CASHIER_FOLDER), ("Staff_02", STOCK_FOLDER)]
    cast += [(f"Shopper_{index:02d}", folder) for index, folder in enumerate(shoppers, start=1)]
    return [(role, folder, _usd_in_folder(root_url, folder)) for role, folder in cast]


async def _wait_frames(count: int) -> None:
    app = omni.kit.app.get_app()
    for _ in range(count):
        await app.next_update_async()


async def _payload(stage: Usd.Stage, prim_path: str, asset_url: str) -> Usd.Prim:
    """Load a payload. CreatePayloadCommand renames the prim if the path is taken."""
    occupant = stage.GetPrimAtPath(prim_path)
    if occupant and occupant.IsValid():
        omni.kit.commands.execute("DeletePrims", paths=[prim_path])
        await _wait_frames(2)
    _ok, created = omni.kit.commands.execute(
        "CreatePayloadCommand",
        usd_context=omni.usd.get_context(),
        path_to=prim_path,
        asset_path=asset_url,
        prim_path=None,
        instanceable=False,
        select_prim=False,
    )
    created_path = str(created) if created else prim_path
    prim = None
    for _ in range(400):
        await omni.kit.app.get_app().next_update_async()
        prim = stage.GetPrimAtPath(created_path)
        if not (prim and prim.IsValid() and prim.GetChildren()):
            continue
        if "HumanMotionLibrary" in created_path:
            return prim
        if _skelroot_or_none(stage, prim) is not None:
            return prim
    names = [child.GetName() for child in prim.GetChildren()] if prim and prim.IsValid() else []
    raise RuntimeError(f"Payload at {created_path} did not compose. children={names} url={asset_url}")


def _skelroot_or_none(stage: Usd.Stage, prim: Usd.Prim):
    from omni.metropolis.pipeline.usd_util import get_descendant_prims_by_type

    if not prim or not prim.IsValid():
        return None
    found = get_descendant_prims_by_type("SkelRoot", prim, None, stage)
    return found[0] if found else None


def _skelroot(stage: Usd.Stage, prim: Usd.Prim) -> Usd.Prim:
    found = _skelroot_or_none(stage, prim)
    if found is None:
        raise RuntimeError(f"No SkelRoot under {prim.GetPath()}")
    return found


def _ensure_translate_scale(prim: Usd.Prim) -> None:
    xform = UsdGeom.Xformable(prim)
    ops = {op.GetOpName() for op in xform.GetOrderedXformOps()}
    if "xformOp:translate" not in ops:
        xform.AddTranslateOp()
    if "xformOp:orient" not in ops:
        xform.AddOrientOp()
    if "xformOp:scale" not in ops:
        xform.AddScaleOp()


def _set_translate(prim: Usd.Prim, position: Gf.Vec3d) -> None:
    _ensure_translate_scale(prim)
    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(position))


def _world_height(prim: Usd.Prim) -> float:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], True)
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if bounds.IsEmpty():
        return 0.0
    return float(bounds.GetSize()[2])


def _fit_human_height(prim: Usd.Prim, target_cm: float = 172.0) -> float:
    """People assets are authored in meters. A cm stage leaves them 1.8 cm tall."""
    _ensure_translate_scale(prim)
    height = _world_height(prim)
    if height <= 1.0:
        return 0.0
    if 150.0 <= height <= 210.0:
        return height
    scale = prim.GetAttribute("xformOp:scale").Get()
    current = Gf.Vec3d(scale) if scale is not None else Gf.Vec3d(1.0, 1.0, 1.0)
    factor = target_cm / height
    prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(current[0] * factor, current[1] * factor, current[2] * factor))
    return _world_height(prim)


@dataclass
class Person:
    role: str
    folder: str
    payload_path: str
    skel_path: str
    agent: object
    height_cm: float
    entry_s: float = 0.0
    shopper_index: int = 0
    phase: str = "parked"
    phase_until: float = -1.0
    goal: tuple[float, float, float] | None = None
    face: tuple[float, float, float] = (0.0, -1.0, 0.0)
    focus: tuple[float, float, float] | None = None
    hidden: bool = False
    gone: bool = False
    notes: list[str] = field(default_factory=list)
    clip: str = ""


def _vec3(point: tuple[float, float, float]) -> carb.Float3:
    return carb.Float3(float(point[0]), float(point[1]), float(point[2]))


def _cancel(agent) -> None:
    task_id = agent.get_action_task_id()
    if task_id:
        agent.cancel_task(task_id)


def _position(agent) -> tuple[float, float, float]:
    pos = agent.get_world_translation()
    return float(pos[0]), float(pos[1]), float(pos[2])


def _close(agent, goal: tuple[float, float, float], radius_cm: float = 55.0) -> bool:
    pos = _position(agent)
    return math.hypot(pos[0] - goal[0], pos[1] - goal[1]) <= radius_cm


def _speed(agent, meters_per_unit: float, meters_per_second: float) -> None:
    agent.set_speed(meters_per_second / meters_per_unit)


def _matching_action(agent, preferences: tuple[str, ...]) -> str | None:
    names = {str(name) for name in agent.get_custom_action_names()}
    for name in preferences:
        if name in names:
            return name
    lowered = {name.lower(): name for name in names}
    for name in preferences:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _resolve_clip(person: Person, preferences: tuple[str, ...]) -> str:
    """Use a listed clip only. Calling custom_action on an unknown name crashes Kit.

    This motion library has Idle / IdleTilt as assets, but they are not registered
    as custom actions, and there is no typing or look_around clip. The body loop
    then falls through to idle() and look_at().
    """
    if person.clip:
        return person.clip
    listed = _matching_action(person.agent, preferences)
    person.clip = listed or "idle"
    return person.clip


def _play(person: Person, preferences: tuple[str, ...], duration: float, face: tuple[float, float, float] | None = None) -> str:
    _cancel(person.agent)
    clip = _resolve_clip(person, preferences)
    if clip and clip != "idle":
        task_id = person.agent.custom_action(clip, duration=duration)
        if task_id != bh.BEHAVIOR_TASK_ID_INVALID:
            return clip
    facing = _vec3(face or person.face)
    if person.focus is not None and duration > 0.0:
        person.agent.look_at(_vec3(person.focus), duration=duration)
        return "look_at"
    person.agent.idle(facing=facing)
    return "idle"


def _hand_task_ok(task_id) -> bool:
    return task_id is not None and task_id != bh.BEHAVIOR_TASK_ID_INVALID


def _reach_point(person: Person, distance_cm: float = 45.0, height_cm: float = 125.0) -> tuple[float, float, float]:
    """A shelf point one arm's length along the facing direction, not a walk target."""
    pos = _position(person.agent)
    return (
        pos[0] + person.face[0] * distance_cm,
        pos[1] + person.face[1] * distance_cm,
        height_cm,
    )


def _reach_shelf(person: Person, target: tuple[float, float, float], duration: float) -> str:
    """Hand extension toward a product. The named reach USD is not in this build."""
    try:
        task_id = person.agent.reach_hand(
            hand_usage=bh.BehaviorHandUsage.RIGHT_HAND,
            target=_vec3(target),
            duration=float(duration),
        )
    except Exception as exc:
        print(f"REACH_HAND_FAIL {person.role} {exc}")
        return ""
    if not _hand_task_ok(task_id):
        return ""
    return "reach_hand"


def _tap_keys(person: Person, t: float) -> str:
    """Alternate fist and point about three times a second. The typing USD is not in this build."""
    slot = int(t / 0.35)
    if getattr(person, "hand_slot", -1) == slot:
        return "pose_hand"
    person.hand_slot = slot
    hand = bh.BehaviorHandUsage.RIGHT_HAND if slot % 2 == 0 else bh.BehaviorHandUsage.LEFT_HAND
    preset = bh.BehaviorHandPosePreset.FIST if slot % 2 == 0 else bh.BehaviorHandPosePreset.POINT
    try:
        task_id = person.agent.pose_hand(hand_usage=hand, preset=preset, duration=0.4)
    except Exception as exc:
        print(f"POSE_HAND_FAIL {person.role} {exc}")
        return ""
    if not _hand_task_ok(task_id):
        return ""
    return "pose_hand"


async def _rebind_or_online(people: list[Person]) -> None:
    """Refresh agent handles. Only stop and play again when one is actually missing."""
    iface = bh.acquire_interface()
    missing = [person for person in people if not person.gone and iface.get_agent(person.skel_path) is None]
    if missing:
        await _agents_online(people)
        return
    for person in people:
        if not person.gone:
            person.agent = iface.get_agent(person.skel_path)


async def _agents_online(people: list[Person], frames: int = 180) -> None:
    """Behavior agents register only after a fresh API resync and while the timeline plays.

    ApplyBehaviorAgentAPICommand does nothing if the API is already on the prim, and
    the runtime ignores that prim until BehaviorAgentAPI.Apply is written again.
    """
    import BehaviorSchema
    import omni.timeline

    stage = omni.usd.get_context().get_stage()
    for person in people:
        if person.gone:
            continue
        # A hidden payload is not registered on the next play. Show it first.
        payload = stage.GetPrimAtPath(person.payload_path)
        if payload and payload.IsValid():
            UsdGeom.Imageable(payload).MakeVisible()
            person.hidden = False
        prim = stage.GetPrimAtPath(person.skel_path)
        if prim and prim.IsValid():
            BehaviorSchema.BehaviorAgentAPI.Apply(prim)
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    await _wait_frames(2)
    timeline.play()
    iface = bh.acquire_interface()
    for _ in range(frames):
        missing = [person for person in people if not person.gone and iface.get_agent(person.skel_path) is None]
        if not missing:
            for person in people:
                if not person.gone:
                    person.agent = iface.get_agent(person.skel_path)
            return
        await omni.kit.app.get_app().next_update_async()
        if not timeline.is_playing():
            timeline.play()
    missing = [person.role for person in people if not person.gone and iface.get_agent(person.skel_path) is None]
    raise RuntimeError(f"Behavior agents missing after timeline play: {missing}")


def _walk(person: Person, goal: tuple[float, float, float], meters_per_unit: float, speed: float) -> None:
    _cancel(person.agent)
    _speed(person.agent, meters_per_unit, speed)
    task_id = person.agent.move_to(_vec3(goal), auto_brake=True)
    if task_id == bh.BEHAVIOR_TASK_ID_INVALID:
        person.agent.teleport(_vec3(goal), _vec3(person.face))
    person.goal = goal


def _show(stage: Usd.Stage, person: Person, visible: bool) -> None:
    imageable = UsdGeom.Imageable(stage.GetPrimAtPath(person.payload_path))
    if visible:
        imageable.MakeVisible()
        person.hidden = False
    else:
        imageable.MakeInvisible()
        person.hidden = True


def _clear(stage: Usd.Stage, person: Person) -> None:
    prim = stage.GetPrimAtPath(person.payload_path)
    if prim and prim.IsValid():
        prim.SetActive(False)
    person.gone = True
    person.phase = "cleared"


def _snap(point: tuple[float, float, float]) -> tuple[float, float, float]:
    import omni.anim.navigation.core as nav

    from omni.anim.navigation.core import NavAgentDesc

    mesh = nav.acquire_interface().get_navmesh()
    if mesh is None:
        raise RuntimeError("NavMesh is missing. Run step1 on this stage before the footfall.")
    body = NavAgentDesc(radius=AGENT_MAX_RADIUS_CM, height=200.0, collision_gap=0.0)
    closest, _island = mesh.query_closest_point(target=_vec3((point[0], point[1], 30.0)), agent=body)
    if closest is None:
        raise RuntimeError(f"No navmesh near {point}")
    return (float(closest[0]), float(closest[1]), float(closest[2]))


def _entry_seconds(index: int) -> float:
    """Pairs enter every 10 seconds: 1-2, 3-4, 5-6, 7-8."""
    if index <= 0:
        return 0.0
    return float(((index - 1) // 2) * 10)


def configure_spacing(people: list[Person]) -> dict:
    """Ask for separation weight 2.5. This Kit build has no such navmesh knob.

    Auto avoidance is the spacing control that does exist. It is enabled on
    every live agent. A real setter whose name contains 'separ' is called
    with 2.5 when one is present.
    """
    settings = carb.settings.get_settings()
    settings.set("/exts/omni.anim.behavior.core/enableAutoAvoidance", True)
    sample = next((person.agent for person in people if person.agent is not None), None)
    agent_names = []
    if sample is not None:
        agent_names = [name for name in dir(sample) if "separ" in name.lower() or "weight" in name.lower()]
    module_names = [name for name in dir(bh) if "separ" in name.lower()]
    setter = None
    if sample is not None:
        for name in agent_names:
            if not name.startswith("set_"):
                continue
            setter = name
            break
    applied = 0
    for person in people:
        agent = person.agent
        if agent is None:
            continue
        if hasattr(agent, "set_auto_avoidance_enabled"):
            agent.set_auto_avoidance_enabled(True)
        if setter is not None:
            getattr(agent, setter)(SEPARATION_WEIGHT)
            applied += 1
    if setter is None:
        print(
            f"SEPARATION_WEIGHT {SEPARATION_WEIGHT} has no setter on this Kit build "
            f"(agent={agent_names} module={module_names}). "
            "Auto avoidance is enabled. NavMesh minimum radius is 0.20 m and "
            f"maximum stays {AGENT_MAX_RADIUS_CM:.0f} cm so the bodies can walk."
        )
    else:
        print(f"SEPARATION_WEIGHT set via {setter}({SEPARATION_WEIGHT}) on {applied} agents")
    return {"setter": setter, "applied": applied, "avoidance": True}


async def spawn_cast(stage: Usd.Stage, cast: list[tuple[str, str, str]], motion_url: str) -> list[Person]:
    import NavSchema
    from isaacsim.replicator.agent.core.scene_assembly.character_loader import USDSchemaUtil
    from isaacsim.replicator.agent.core.configuration.models.character import IRACharacterGroup

    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        timeline.stop()
        await _wait_frames(8)
    existing = stage.GetPrimAtPath(CHARACTERS_ROOT)
    if existing and existing.IsValid() and list(existing.GetChildren()):
        # Deleting behavior agents and spawning them again crashes Kit on the
        # next timeline resume. A fresh Isaac Sim process is the safe retry.
        raise RuntimeError(
            "Characters are already on the stage. Refusing to delete them. "
            "Reuse the cast, or relaunch Isaac Sim before spawning again."
        )
    if not stage.GetPrimAtPath(CHARACTERS_ROOT):
        UsdGeom.Xform.Define(stage, CHARACTERS_ROOT)
    motion = await _payload(stage, MOTION_LIBRARY, motion_url)
    motion_path = str(motion.GetPath())

    people: list[Person] = []
    for role, folder, asset_url in cast:
        payload_path = f"{CHARACTERS_ROOT}/{role}"
        payload = await _payload(stage, payload_path, asset_url)
        payload_path = str(payload.GetPath())
        skel = _skelroot(stage, payload)
        omni.kit.commands.execute(
            "ApplyBehaviorAgentAPICommand",
            skelroot_prim_paths=[skel.GetPath()],
            motion_library_prim_path=motion_path,
            motion_library_skeleton_rig="Human",
        )
        await _wait_frames(2)
        if not skel.HasAPI(NavSchema.NavMeshExcludeAPI):
            omni.kit.commands.execute(
                "ApplyNavMeshAPICommand",
                prim_path=skel.GetPath(),
                api=NavSchema.NavMeshExcludeAPI,
            )
        group = IRACharacterGroup.model_construct(
            name=role,
            num=1,
            asset_path="",
            routines=[],
            triggers=[],
            semantic_labels=[("class", "character")],
        )
        USDSchemaUtil.create_ira_character(stage, skel, SAMPLE_SEED, role, group)
        _ensure_translate_scale(payload)
        index = 0
        entry = 0.0
        if role.startswith("Shopper_"):
            index = int(role.split("_")[1])
            entry = _entry_seconds(index)
        people.append(
            Person(
                role=role,
                folder=folder,
                payload_path=payload_path,
                skel_path=str(skel.GetPath()),
                agent=None,
                height_cm=0.0,
                entry_s=entry,
                shopper_index=index,
            )
        )
    await _wait_frames(8)
    for person in people:
        payload = stage.GetPrimAtPath(person.payload_path)
        height = 0.0
        for _ in range(40):
            height = _fit_human_height(payload)
            if height >= 150.0:
                break
            await omni.kit.app.get_app().next_update_async()
        person.height_cm = round(height, 1)
        if person.height_cm < 140.0:
            raise RuntimeError(f"{person.role} height is {person.height_cm} cm after unit fit.")
    await _agents_online(people)
    return people


def _marks() -> dict:
    """Stand points on the rebaked mesh. Customer side of the counter is +X of the tills."""
    # Shopper 1 holds the head of counter 1. Shoppers 2-5 line up behind that spot.
    queue = [
        (400.0, -345.0, 1.8),
        (400.0, -435.0, 1.8),
        (400.0, -525.0, 1.8),
        (400.0, -615.0, 1.8),
        (400.0, -705.0, 1.8),
    ]
    backup = [(400.0, -590.0, 1.8), (400.0, -680.0, 1.8)]
    door = [(340.0, -90.0, 1.8), (430.0, -90.0, 1.8)]
    # Outside the front wall, behind the entrance cameras, until each pair's entry.
    park = [(180.0, 60.0, 1.8), (280.0, 60.0, 1.8), (520.0, 60.0, 1.8), (700.0, 60.0, 1.8)]
    # Dairy is the front gondola (milk sits near y=-353). Snacks are the rear gondola.
    browse = [
        (620.0, -430.0, 1.8),
        (760.0, -430.0, 1.8),
        (900.0, -430.0, 1.8),
        (620.0, -740.0, 1.8),
        (760.0, -740.0, 1.8),
        (900.0, -740.0, 1.8),
        (1160.0, -280.0, 1.8),
        (1160.0, -420.0, 1.8),
    ]
    exit_point = _snap((EXIT_NODE_CM[0], EXIT_NODE_CM[1], 30.0))
    exit_gap = math.hypot(exit_point[0] - EXIT_NODE_CM[0], exit_point[1] - EXIT_NODE_CM[1])
    if exit_gap > 120.0:
        raise RuntimeError(f"Exit snap {exit_point} is {exit_gap:.0f} cm from {EXIT_NODE_CM}.")
    return {
        "staff1": _snap((200.0, -345.0, 1.8)),
        "staff2": _snap((250.0, -740.0, 1.8)),
        "staff2_register": _snap((200.0, -590.0, 1.8)),
        "counter1": _snap((400.0, -345.0, 1.8)),
        "queue": [_snap(point) for point in queue],
        "backup": [_snap(point) for point in backup],
        "door": [_snap(point) for point in door],
        "park": [_snap(point) for point in park],
        "browse": [_snap(point) for point in browse],
        "exit": exit_point,
    }


def _focus_for(index: int, stand: tuple[float, float, float]) -> tuple[float, float, float]:
    if index <= 3:
        return (stand[0], stand[1] + 70.0, 125.0)
    if index <= 6:
        return (stand[0], stand[1] - 70.0, 40.0)
    return (stand[0] + 70.0, stand[1], 140.0)


def _face_for(index: int) -> tuple[float, float, float]:
    if index <= 3:
        return (0.0, 1.0, 0.0)
    if index <= 6:
        return (0.0, -1.0, 0.0)
    return (1.0, 0.0, 0.0)


def _dwell_for(index: int) -> float:
    if index <= 6:
        return DWELL_SECONDS
    return BEVERAGE_DWELL_SECONDS


def place_people(stage: Usd.Stage, people: list[Person], marks: dict) -> None:
    for person in people:
        person.goal = None
        person.phase_until = -1.0
        person.notes = []
        person.gone = False
        person.hidden = False
        person.hand_slot = -1
        person.clip = ""
        person.reach_sent = False
    by_role = {person.role: person for person in people}
    staff1 = by_role["Staff_01"]
    staff1.face = (1.0, 0.0, 0.0)
    staff1.focus = (280.0, -345.0, 115.0)
    staff1.agent.teleport(_vec3(marks["staff1"]), _vec3(staff1.face))
    staff1.phase = "billing"

    staff2 = by_role["Staff_02"]
    staff2.face = (0.0, -1.0, 0.0)
    staff2.focus = (200.0, -800.0, 140.0)
    staff2.agent.teleport(_vec3(marks["staff2"]), _vec3(staff2.face))
    staff2.phase = "idle"

    for person in people:
        if person.shopper_index <= 0:
            continue
        person.face = _face_for(person.shopper_index)
        slot = (person.shopper_index - 1) % 2
        if person.entry_s <= 0.0:
            stand = marks["door"][slot]
            person.phase = "entering"
            _show(stage, person, True)
        else:
            stand = marks["park"][(person.shopper_index - 1) % len(marks["park"])]
            person.phase = "parked"
            _show(stage, person, False)
        person.agent.teleport(_vec3(stand), _vec3((0.0, -1.0, 0.0)))


def _products_in_box(stage: Usd.Stage, prefixes: tuple[str, ...], box: tuple[float, ...]) -> list[tuple]:
    xmin, xmax, ymin, ymax, zmin, zmax = box
    root = stage.GetPrimAtPath(STORE_ROOT)
    found = []
    for child in root.GetChildren():
        if not child.IsActive():
            continue
        name = child.GetName()
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        translate = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        x, y, z = float(translate[0]), float(translate[1]), float(translate[2])
        if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
            found.append((x, y, z, str(child.GetPath())))
    return found


def _tight_group(found: list[tuple], anchor: tuple[float, float, float], count: int, span: tuple[float, float, float]) -> list[str]:
    ranked = sorted(
        found,
        key=lambda item: (item[0] - anchor[0]) ** 2 + (item[1] - anchor[1]) ** 2 + (item[2] - anchor[2]) ** 2,
    )
    chosen = []
    for item in ranked:
        if not chosen:
            chosen.append(item)
        else:
            origin = chosen[0]
            if (
                abs(item[0] - origin[0]) <= span[0]
                and abs(item[1] - origin[1]) <= span[1]
                and abs(item[2] - origin[2]) <= span[2]
            ):
                chosen.append(item)
        if len(chosen) == count:
            break
    return [path for _x, _y, _z, path in chosen]


def select_milk_prims(stage: Usd.Stage) -> list[str]:
    """Five milk cartons on tier 2 of the front gondola, in front of shoppers 1-3."""
    found = _products_in_box(
        stage,
        ("SM_Milk",),
        (540.0, 900.0, -390.0, -310.0, 95.0, 160.0),
    )
    cartons = [item for item in found if "Bottle" not in item[3] and "bottle" not in item[3]]
    pool = cartons if len(cartons) >= 5 else found
    chosen = _tight_group(pool, (640.0, -353.0, 120.0), 5, (90.0, 40.0, 60.0))
    if len(chosen) < 5:
        raise RuntimeError(f"Only {len(chosen)} milk cartons on Shelf 1 tier 2 (scanned {len(found)}).")
    return chosen


def select_chip_prims(stage: Usd.Stage) -> list[str]:
    """Ten chip bags on the low aisle-facing row of the rear gondola."""
    found = _products_in_box(
        stage,
        ("SM_Packet2_3_Chips",),
        (540.0, 940.0, -870.0, -800.0, 15.0, 55.0),
    )
    chosen = _tight_group(found, (640.0, -830.0, 32.0), 10, (160.0, 50.0, 30.0))
    if len(chosen) < 10:
        raise RuntimeError(f"Only {len(chosen)} chip bags on Shelf 2 tier 1 (scanned {len(found)}).")
    return chosen


def select_gap_prims(stage: Usd.Stage) -> list[str]:
    """Milk then chips. Kept so an older caller still hides the Track K rows."""
    return select_milk_prims(stage) + select_chip_prims(stage)


class StoreDirector:
    """Track K clock. Each command is issued once."""

    def __init__(
        self,
        stage: Usd.Stage,
        people: list[Person],
        marks: dict,
        meters_per_unit: float,
        gap_paths: list[str] | None = None,
        milk_paths: list[str] | None = None,
        chip_paths: list[str] | None = None,
    ):
        self.stage = stage
        self.people = people
        self.by_role = {person.role: person for person in people}
        self.marks = marks
        self.mpu = meters_per_unit
        if milk_paths is None and chip_paths is None:
            paths = list(gap_paths or [])
            self.milk_paths = paths[:5]
            self.chip_paths = paths[5:]
        else:
            self.milk_paths = list(milk_paths or [])
            self.chip_paths = list(chip_paths or [])
        self.gap_paths = self.milk_paths + self.chip_paths
        self.milk_done = False
        self.chips_done = False
        self.queue_done = False
        self.split_done = False
        self.look_action = ""
        self.bill_action = ""

    def rebind(self) -> int:
        iface = bh.acquire_interface()
        bound = 0
        for person in self.people:
            if person.gone:
                continue
            agent = iface.get_agent(person.skel_path)
            if agent is not None:
                person.agent = agent
                bound += 1
        return bound

    def prime(self) -> None:
        self.rebind()
        staff1 = self.by_role["Staff_01"]
        staff2 = self.by_role["Staff_02"]
        staff1.hand_slot = -1
        staff1.phase_until = -1.0
        staff2.phase = "idle"
        staff2.phase_until = -1.0
        self._hold_billing(staff1, 0.0)
        self._hold_idle(staff2, 0.0)
        self.bill_action = "pose_hand"
        self.look_action = "idle"
        configure_spacing(self.people)
        print(f"CAST_ACTION billing={self.bill_action} stock={self.look_action}")

    def update(self, t: float) -> None:
        self.rebind()
        if t >= 25.0 and not self.milk_done:
            self._hide_products(self.milk_paths, MILK_ALERT, 5)
            self.milk_done = True
        if t >= 40.0 and not self.chips_done:
            self._hide_products(self.chip_paths, CHIPS_ALERT, 10)
            self.chips_done = True
        for person in self.people:
            if person.gone or person.agent is None:
                continue
            if person.role == "Staff_01":
                self._hold_billing(person, t)
            elif person.role == "Staff_02":
                self._update_staff2(person, t)
            elif person.shopper_index > 0:
                self._update_shopper(person, t)
        if t >= 50.0 and not self.queue_done:
            _trace(QUEUE_ALERT)
            _trace(f"QUEUE_MEASURED length={self._queue_count()} phases={self._queue_phases()}")
            self.queue_done = True
            self.split_done = True

    def _hold_billing(self, person: Person, t: float) -> None:
        if person.phase_until <= t:
            person.agent.idle(facing=_vec3(person.face))
            person.phase_until = t + 6.0
            person.hand_slot = -1
        _tap_keys(person, t)

    def _hold_idle(self, person: Person, t: float) -> None:
        if person.phase_until <= t:
            person.agent.idle(facing=_vec3(person.face))
            person.phase_until = t + 6.0

    def _update_staff2(self, person: Person, t: float) -> None:
        if t < 50.0:
            if person.phase != "idle":
                person.phase = "idle"
                print("STAFF_02 idle in back room")
            self._hold_idle(person, t)
            return
        if person.phase == "backup_till":
            self._hold_billing(person, t)
            return
        if person.phase != "to_register":
            person.phase = "to_register"
            person.face = (1.0, 0.0, 0.0)
            _walk(person, self.marks["staff2_register"], self.mpu, BRISK_MPS)
            print("STAFF_02 dispatch to counter 2")
            return
        if _close(person.agent, self.marks["staff2_register"], 70.0):
            person.phase = "backup_till"
            person.phase_until = t
            person.hand_slot = -1
            person.focus = (360.0, self.marks["staff2_register"][1], 115.0)
            person.agent.idle(facing=_vec3(person.face))
            _tap_keys(person, t)
            print(f"STAFF_02 typing at counter 2 t={t:.1f}")

    def _update_shopper(self, person: Person, t: float) -> None:
        index = person.shopper_index
        if person.phase == "parked" and t + 1e-4 >= person.entry_s:
            self._enter(person, t)
            return
        if person.phase == "entering" and person.goal is None and t + 1e-4 >= person.entry_s:
            self._send_browse(person)
            return
        if person.phase == "to_shelf" and person.goal and _close(person.agent, person.goal, 70.0):
            self._begin_dwell(person, t)
        if person.phase in ("dwell", "hold", "to_shelf"):
            self._maybe_reach(person, t)
        if person.phase == "dwell" and t >= person.phase_until:
            person.phase = "hold"
            person.phase_until = t
            person.agent.idle(facing=_vec3(person.face))
        if person.phase == "hold" and person.phase_until <= t:
            person.agent.idle(facing=_vec3(person.face))
            person.phase_until = t + 6.0
        if index == 1 and t >= 33.0 and person.phase in ("to_shelf", "dwell", "hold"):
            self._send_queue(person, self.marks["counter1"], "to_counter")
            return
        if index in (2, 3) and t >= 41.0 and person.phase in ("to_shelf", "dwell", "hold", "parked", "entering"):
            self._send_queue(person, self.marks["queue"][index - 1], "to_counter")
            return
        if index in (4, 5) and t >= 40.0 and person.phase in ("to_shelf", "dwell", "hold", "parked", "entering"):
            self._send_queue(person, self.marks["queue"][index - 1], "to_counter")
            return
        if person.phase == "to_counter" and person.goal and _close(person.agent, person.goal, 75.0):
            person.phase = "billing" if index == 1 else "queue"
            person.face = (-1.0, 0.0, 0.0)
            person.agent.idle(facing=_vec3(person.face))
            print(f"{person.role} at counter 1 t={t:.1f}")
            return
        if index in (4, 5) and t >= 50.0 and person.phase in ("queue", "to_counter") and "split" not in person.notes:
            person.notes.append("split")
            person.phase = "to_backup"
            person.face = (-1.0, 0.0, 0.0)
            goal = self.marks["backup"][index - 4]
            _walk(person, goal, self.mpu, WALK_MPS)
            print(f"{person.role} step to counter 2")
            return
        if person.phase == "to_backup" and person.goal and _close(person.agent, person.goal, 75.0):
            person.phase = "backup"
            person.agent.idle(facing=_vec3(person.face))
            print(f"{person.role} at counter 2 t={t:.1f}")
            return
        exit_at = {1: 45.0, 2: 60.0, 3: 70.0, 4: 65.0, 5: 75.0}.get(index)
        checkout = ("queue", "billing", "backup", "to_counter", "to_backup")
        if exit_at is not None and t >= exit_at and person.phase in checkout:
            self._send_exit(person, t)
            return
        if person.phase == "exit":
            pos = _position(person.agent)
            if pos[0] <= EXIT_NODE_CM[0] + 40.0 and abs(pos[1] - EXIT_NODE_CM[1]) <= 180.0:
                _clear(self.stage, person)
                print(f"{person.role} cleared past parking ({pos[0]:.0f},{pos[1]:.0f})")

    def _enter(self, person: Person, t: float) -> None:
        _show(self.stage, person, True)
        slot = (person.shopper_index - 1) % 2
        start = self.marks["door"][slot]
        person.agent.teleport(_vec3(start), _vec3((0.0, -1.0, 0.0)))
        self._send_browse(person)
        print(f"{person.role} enter t={t:.1f}s mesh={person.folder}")

    def _begin_dwell(self, person: Person, t: float) -> None:
        dwell = _dwell_for(person.shopper_index)
        person.phase = "dwell"
        person.phase_until = t + dwell
        _cancel(person.agent)
        person.agent.idle(facing=_vec3(person.face))
        print(f"{person.role} dwell {dwell:.0f}s from t={t:.1f}")

    def _maybe_reach(self, person: Person, t: float) -> None:
        if person.reach_sent or person.shopper_index > 6:
            return
        due = 15.0 if person.shopper_index <= 3 else 25.0
        at_shelf = person.phase in ("dwell", "hold") or (
            person.phase == "to_shelf" and person.goal is not None and _close(person.agent, person.goal, 80.0)
        )
        if t < due or not at_shelf:
            return
        used = _reach_shelf(person, _reach_point(person), 3.0)
        person.reach_sent = True
        print(f"{person.role} reach t={t:.1f} action={used or 'idle'}")

    def _send_browse(self, person: Person) -> None:
        goal = self.marks["browse"][person.shopper_index - 1]
        person.face = _face_for(person.shopper_index)
        person.focus = _focus_for(person.shopper_index, goal)
        person.phase = "to_shelf"
        _walk(person, goal, self.mpu, WALK_MPS)

    def _send_queue(self, person: Person, goal: tuple[float, float, float], phase: str) -> None:
        _show(self.stage, person, True)
        person.phase = phase
        person.face = (-1.0, 0.0, 0.0)
        person.focus = (300.0, goal[1], 115.0)
        _walk(person, goal, self.mpu, BRISK_MPS)
        print(f"{person.role} -> {phase}")

    def _send_exit(self, person: Person, t: float) -> None:
        goal = self.marks["exit"]
        person.phase = "exit"
        person.face = (0.0, 1.0, 0.0)
        _walk(person, goal, self.mpu, BRISK_MPS)
        print(f"{person.role} exit t={t:.1f} via parking {tuple(round(v, 1) for v in goal)}")

    def _hide_products(self, paths: list[str], alert: str, expected: int) -> None:
        hidden = []
        for path in paths:
            prim = self.stage.GetPrimAtPath(path)
            if prim and prim.IsValid() and prim.IsActive():
                prim.SetActive(False)
                hidden.append(path)
        if len(hidden) != expected:
            raise RuntimeError(f"Hid {len(hidden)} of {expected} products for {alert}")
        _trace(alert)
        _trace("SHELF_HID " + " ".join(path.rsplit("/", 1)[-1] for path in hidden))

    def _queue_count(self) -> int:
        count = 0
        for index in (2, 3, 4, 5):
            person = self.by_role[f"Shopper_{index:02d}"]
            if person.gone or person.agent is None:
                continue
            if person.phase in ("queue", "to_counter", "billing", "to_backup", "backup"):
                count += 1
        return count

    def _queue_phases(self) -> str:
        parts = []
        for index in (2, 3, 4, 5):
            person = self.by_role[f"Shopper_{index:02d}"]
            parts.append(f"{index}:{person.phase}")
        return ",".join(parts)

def _pause_timeline() -> None:
    import omni.timeline

    omni.timeline.get_timeline_interface().pause()


def _configure_timeline() -> None:
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    timeline.set_time_codes_per_second(float(FPS))
    timeline.set_start_time(0.0)
    timeline.set_end_time(RECORD_SECONDS + 30.0)
    timeline.set_current_time(0.0)
    timeline.commit()
    carb.settings.get_settings().set("/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)


def _open_encoders() -> list[tuple[str, subprocess.Popen]]:
    os.makedirs(VIDEO_DIR, exist_ok=True)
    width, height = VIDEO_SIZE
    encoders = []
    for camera in CAMERAS:
        filename = VIDEO_FILES[camera]
        path = os.path.join(VIDEO_DIR, filename)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            path,
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        encoders.append((path, process))
    return encoders


def _trace(line: str) -> None:
    print(line, flush=True)
    try:
        with open("/tmp/trackk_progress.log", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def _write_rgb(process: subprocess.Popen, array) -> None:
    import numpy as np

    frame = np.asarray(array)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise RuntimeError(f"RGB frame has shape {frame.shape}")
    rgb = np.ascontiguousarray(frame[:, :, :3])
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    expected = VIDEO_SIZE[0] * VIDEO_SIZE[1] * 3
    raw = rgb.tobytes()
    if len(raw) != expected:
        raise RuntimeError(f"Frame is {len(raw)} bytes, expected {expected} for {VIDEO_SIZE}.")
    process.stdin.write(raw)


def _close_encoders(encoders: list[tuple[str, subprocess.Popen]]) -> list[dict]:
    report = []
    for path, process in encoders:
        if process.stdin:
            process.stdin.close()
        code = process.wait(timeout=180)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        if code != 0 or size < 100000:
            raise RuntimeError(f"Encoder failed for {path}: code {code}, size {size}")
        report.append({"path": path, "bytes": size})
    return report


async def record(director: StoreDirector, frames: int) -> list[dict]:
    import omni.replicator.core as rep

    _configure_timeline()
    width, height = VIDEO_SIZE
    products = [rep.create.render_product(camera, (width, height)) for camera in CAMERAS]
    annotators = []
    for product in products:
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
        annotators.append(annotator)
    await _wait_frames(3)
    # stop() inside _configure_timeline drops the runtime agents. Bring them
    # back, then put every person on the navmesh mark again.
    await _agents_online(director.people)
    place_people(director.stage, director.people, director.marks)
    director.prime()
    for _ in range(8):
        await rep.orchestrator.step_async(pause_timeline=False)
    await _rebind_or_online(director.people)
    place_people(director.stage, director.people, director.marks)
    director.milk_done = False
    director.chips_done = False
    director.queue_done = False
    director.split_done = False
    director.prime()
    encoders = _open_encoders()
    try:
        written = 0
        empty_run = 0
        _trace(f"RECORD_ARMED frames={frames} cameras={len(CAMERAS)}")
        while written < frames:
            t = written / float(FPS)
            director.update(t)
            await rep.orchestrator.step_async(pause_timeline=False)
            frames_ok = True
            grabbed = []
            for annotator in annotators:
                data = annotator.get_data()
                if data is None or len(data) == 0:
                    frames_ok = False
                    break
                grabbed.append(data)
            if not frames_ok:
                empty_run += 1
                if empty_run > 20:
                    raise RuntimeError(f"Camera produced an empty frame at t={t:.2f}")
                continue
            empty_run = 0
            for data, (_path, process) in zip(grabbed, encoders):
                _write_rgb(process, data)
            if written % (FPS * 5) == 0:
                spots = []
                for person in director.people:
                    if person.gone or person.shopper_index <= 0 or person.agent is None:
                        continue
                    pos = _position(person.agent)
                    spots.append(f"{person.role}:{person.phase}({pos[0]:.0f},{pos[1]:.0f})")
                _trace(f"RECORD t={t:.1f}s frame={written} {' '.join(spots)}")
            written += 1
    finally:
        videos = _close_encoders(encoders)
    return videos


async def prepare() -> dict:
    _enable("isaacsim.replicator.agent.core")
    _enable("omni.anim.behavior.core")
    _enable("omni.anim.navigation.core")
    _enable("omni.replicator.core")
    stage = _stage()
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise RuntimeError("Expected a Z-up store.")
    for path in (CHECKOUT_CAMERA, ENTRANCE_CAMERA):
        if not stage.GetPrimAtPath(path):
            raise RuntimeError(f"Missing {path}. Run step1 on this open stage first.")
    lights = ensure_shop_lights(stage)
    camera = place_back_camera(stage, meters_per_unit)
    left_camera = place_left_back_camera(stage, meters_per_unit)
    entrance_end = place_entrance_end_camera(stage, meters_per_unit)
    root_url = _resolve_people_root()
    cast = assign_cast(root_url)
    people = await spawn_cast(stage, cast, _resolve_motion_library())
    marks = _marks()
    place_people(stage, people, marks)
    gap_paths = select_gap_prims(stage)
    director = StoreDirector(stage, people, marks, meters_per_unit, gap_paths)
    print("PHASE1_READY")
    print(f"CAMERA {camera['path']} eye={camera['eye_cm']} depression={camera['depression_deg']} focal={camera['focal_length_mm']}mm")
    print(f"CAMERA {left_camera['path']} eye={left_camera['eye_cm']} depression={left_camera['depression_deg']} focal={left_camera['focal_length_mm']}mm")
    print(f"CAMERA {entrance_end['path']} eye={entrance_end['eye_cm']} depression={entrance_end['depression_deg']} focal={entrance_end['focal_length_mm']}mm")
    print(f"LIGHTS sun={lights['sun']} rects={len(lights['rects'])}")
    for person in people:
        print(f"ACTOR {person.role} mesh={person.folder} height={person.height_cm}cm entry={person.entry_s}")
    print(f"SHELF_CLUSTER {len(gap_paths)} prims e.g. {gap_paths[0]}")
    return {
        "stage": stage,
        "camera": camera,
        "people": people,
        "director": director,
        "meters_per_unit": meters_per_unit,
    }


def bind_cast_from_stage(stage: Usd.Stage) -> list[Person]:
    """Rebuild the director cast from characters already on the open stage."""
    people: list[Person] = []
    roles = ["Staff_01", "Staff_02"] + [f"Shopper_{index:02d}" for index in range(1, 9)]
    for role in roles:
        payload = stage.GetPrimAtPath(f"{CHARACTERS_ROOT}/{role}")
        if not payload or not payload.IsValid():
            raise RuntimeError(f"Missing {CHARACTERS_ROOT}/{role}")
        if not payload.IsActive():
            payload.SetActive(True)
        skel = _skelroot(stage, payload)
        children = payload.GetChildren()
        folder = children[0].GetName() if children else role
        index = int(role.split("_")[1]) if role.startswith("Shopper_") else 0
        entry = _entry_seconds(index)
        height = _world_height(payload)
        # Never rescale a body that is already adult, and never scale a prim
        # whose bounds have not composed yet.
        height_cm = round(height, 1) if 140.0 <= height <= 220.0 else 0.0
        people.append(
            Person(
                role=role,
                folder=folder,
                payload_path=str(payload.GetPath()),
                skel_path=str(skel.GetPath()),
                agent=None,
                height_cm=height_cm,
                entry_s=entry,
                shopper_index=index,
            )
        )
    return people


async def rehearse_loaded() -> dict:
    """Walk the cast already on stage. Does not spawn, delete, or record."""
    stage = _stage()
    people = bind_cast_from_stage(stage)
    await _rebind_or_online(people)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    marks = _marks()
    director = StoreDirector(stage, people, marks, meters_per_unit, select_gap_prims(stage))
    place_people(stage, people, marks)
    director.prime()
    shopper = next(person for person in people if person.role == "Shopper_01")
    start = _position(shopper.agent)
    for frame in range(FPS * 3):
        director.update(frame / float(FPS))
        await omni.kit.app.get_app().next_update_async()
    end = _position(shopper.agent)
    _pause_timeline()
    moved = math.hypot(end[0] - start[0], end[1] - start[1])
    print(
        f"REHEARSE Shopper_01 moved {moved:.1f} cm from {tuple(round(v, 1) for v in start)} to {tuple(round(v, 1) for v in end)}",
        flush=True,
    )
    if moved < 40.0:
        raise RuntimeError("Shopper_01 did not walk. The behavior agent is not advancing.")
    print("REHEARSE_OK", flush=True)
    prepared = {
        "stage": stage,
        "people": people,
        "director": director,
        "meters_per_unit": meters_per_unit,
        "camera": {"path": BACK_CAMERA},
    }
    globals()["MARKETGEN_PREPARED"] = prepared
    return {"moved_cm": round(moved, 1)}


async def rehearse() -> dict:
    """Spawn and walk three seconds. Does not record and does not hide products.

    Timeline is armed once, before the first play. A second stop/play after the
    behavior body colliders are created has crashed Kit on resume.
    """
    _configure_timeline()
    prepared = await prepare()
    director: StoreDirector = prepared["director"]
    iface = bh.acquire_interface()
    missing = [
        person.role
        for person in prepared["people"]
        if not person.gone and iface.get_agent(person.skel_path) is None
    ]
    if missing:
        await _agents_online(prepared["people"])
    else:
        for person in prepared["people"]:
            if not person.gone:
                person.agent = iface.get_agent(person.skel_path)
    place_people(prepared["stage"], prepared["people"], director.marks)
    director.prime()
    shopper = next(person for person in prepared["people"] if person.role == "Shopper_01")
    start = _position(shopper.agent)
    for frame in range(FPS * 3):
        director.update(frame / float(FPS))
        await omni.kit.app.get_app().next_update_async()
    end = _position(shopper.agent)
    _pause_timeline()
    moved = math.hypot(end[0] - start[0], end[1] - start[1])
    print(f"REHEARSE Shopper_01 moved {moved:.1f} cm from {tuple(round(v,1) for v in start)} to {tuple(round(v,1) for v in end)}")
    if moved < 40.0:
        raise RuntimeError("Shopper_01 did not walk. The behavior agent is not advancing.")
    print("REHEARSE_OK")
    globals()["MARKETGEN_PREPARED"] = prepared
    return {"moved_cm": round(moved, 1), "camera": prepared["camera"]}


def _restore_for_take(prepared: dict) -> None:
    """Reactivate shoppers and the shelf after a previous take, and use this module's director."""
    old = prepared["director"]
    stage = old.stage
    for path in old.gap_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            prim.SetActive(True)
    for person in old.people:
        prim = stage.GetPrimAtPath(person.payload_path)
        if prim and prim.IsValid():
            prim.SetActive(True)
            UsdGeom.Imageable(prim).MakeVisible()
        person.gone = False
        person.hidden = False
        person.clip = ""
        person.goal = None
        person.notes = []
        person.phase = "parked"
        person.phase_until = -1.0
        person.hand_slot = -1
        person.scanned = False
    prepared["director"] = StoreDirector(stage, old.people, old.marks, old.mpu, list(old.gap_paths))


def _set_cast_active(stage: Usd.Stage, people: list[Person], drive: bool) -> None:
    """Hide every body while the navmesh bakes, then bring all eight shoppers and both staff back."""
    for person in people:
        prim = stage.GetPrimAtPath(person.payload_path)
        keep = bool(drive)
        if prim and prim.IsValid():
            prim.SetActive(keep)
            if keep:
                UsdGeom.Imageable(prim).MakeVisible()
        person.gone = not keep
        person.hidden = not keep


async def apply_layout(prepared: dict) -> dict:
    """Move the two cameras and rebake the tight navmesh. Does not open the encoders."""
    director: StoreDirector = prepared["director"]
    stage = director.stage
    checkout_before = _camera_pose(stage, CHECKOUT_CAMERA)
    _set_cast_active(stage, director.people, False)
    await _wait_frames(2)
    lights = ensure_shop_lights(stage)
    back = place_back_camera(stage, director.mpu)
    left = place_left_back_camera(stage, director.mpu)
    entrance_end = place_entrance_end_camera(stage, director.mpu)
    entrance = tilt_entrance_camera(stage)
    checkout_after = _camera_pose(stage, CHECKOUT_CAMERA)
    eye_shift = math.dist(checkout_before["eye"], checkout_after["eye"])
    if eye_shift > 0.5 or abs(checkout_before["depression"] - checkout_after["depression"]) > 0.2:
        raise RuntimeError(f"Checkout camera moved: before {checkout_before} after {checkout_after}")
    volume = extend_walkable_volume(stage, director.mpu)
    radius = set_tight_agent()
    baked = await bake_navmesh()
    walk = verify_tight_navmesh()
    _set_cast_active(stage, director.people, True)
    carb.settings.get_settings().set("/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
    print(
        f"CAMERA_BACK eye={back['eye_cm']} forward={back['forward']} up={back['up']} "
        f"depression={back['depression_deg']} "
        f"focal={back['focal_length_mm']} hap={back['horizontal_aperture_mm']}"
    )
    print(
        f"CAMERA_BACK_LEFT eye={left['eye_cm']} forward={left['forward']} up={left['up']} "
        f"depression={left['depression_deg']} "
        f"focal={left['focal_length_mm']} hap={left['horizontal_aperture_mm']}"
    )
    print(f"LIGHTS sun={lights['sun']} rects={len(lights['rects'])}")
    print(
        f"CAMERA_ENTRANCE eye={entrance['eye_cm']} depression={entrance['depression_deg']} "
        f"focal={entrance['focal_length_mm']}"
    )
    print(
        f"CAMERA_ENTRANCE_END eye={entrance_end['eye_cm']} "
        f"depression={entrance_end['depression_deg']} focal={entrance_end['focal_length_mm']}"
    )
    print(
        f"CAMERA_CHECKOUT eye={[round(v, 1) for v in checkout_after['eye']]} "
        f"depression={checkout_after['depression']:.2f} unchanged"
    )
    print(
        f"NAVMESH_TIGHT radius_cm={radius['min_cm']:.1f}/{radius['max_cm']:.1f} "
        f"slope_deg={radius['slope_deg']:.2f} baked={baked['baked']} "
        f"aisle_cm={walk['entry_to_aisle_cm']} exit_cm={walk['door_to_exit_cm']} "
        f"exit={walk['probes']['exit_node']}"
    )
    print(f"NAVMESH_VOLUME {volume['min_cm']} -> {volume['max_cm']}")
    return {"back": back, "entrance": entrance, "walk": walk, "radius": radius}


def _snapshot_cameras(stage: Usd.Stage) -> dict:
    return {path: _camera_pose(stage, path) for path in CAMERAS}


def _assert_cameras_frozen(stage: Usd.Stage, before: dict) -> None:
    for path, old in before.items():
        new = _camera_pose(stage, path)
        shift = math.dist(old["eye"], new["eye"])
        dep = abs(float(old["depression"]) - float(new["depression"]))
        focal_shift = abs(float(old["focal"] or 0.0) - float(new["focal"] or 0.0))
        if shift > 0.5 or dep > 0.5 or focal_shift > 0.05:
            raise RuntimeError(f"Camera moved {path}: before {old} after {new}")
        print(
            f"CAMERA_HELD {path} eye={[round(v, 2) for v in new['eye']]} "
            f"dep={new['depression']:.2f} focal={new['focal']}"
        )


async def track_k_prepare() -> tuple[StoreDirector, dict]:
    """Rebake the floor mesh and bind the cast. Does not move the five cameras."""
    stage = _stage()
    before = _snapshot_cameras(stage)
    for path in CAMERAS:
        if not stage.GetPrimAtPath(path):
            raise RuntimeError(f"Missing camera {path}")
    people = bind_cast_from_stage(stage)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    _set_cast_active(stage, people, False)
    await _wait_frames(2)
    volume = extend_walkable_volume(stage, meters_per_unit)
    radius = set_tight_agent()
    baked = await bake_navmesh()
    walk = verify_tight_navmesh()
    _assert_cameras_frozen(stage, before)
    _set_cast_active(stage, people, True)
    await _wait_frames(4)
    for person in people:
        payload = stage.GetPrimAtPath(person.payload_path)
        children = list(payload.GetChildren())
        if children:
            person.folder = children[0].GetName()
        height = _world_height(payload)
        if 140.0 <= height <= 220.0:
            person.height_cm = round(height, 1)
        if person.height_cm < 140.0:
            raise RuntimeError(f"{person.role} height is {person.height_cm} cm (measured {height}).")
        print(
            f"ACTOR {person.role} mesh={person.folder} height={person.height_cm}cm "
            f"entry={person.entry_s} active={payload.IsActive()}"
        )
    milk = select_milk_prims(stage)
    chips = select_chip_prims(stage)
    print("MILK " + " ".join(path.rsplit("/", 1)[-1] for path in milk))
    print("CHIPS " + " ".join(path.rsplit("/", 1)[-1] for path in chips))
    print(
        f"NAVMESH radius_cm={radius['min_cm']:.1f}/{radius['max_cm']:.1f} "
        f"slope_deg={radius['slope_deg']:.2f} baked={baked['baked']} "
        f"aisle_cm={walk['entry_to_aisle_cm']} exit_cm={walk['door_to_exit_cm']} "
        f"exit={walk['probes']['exit_node']} "
        f"dairy={walk['probes']['dairy_aisle']} snack={walk['probes']['snack_aisle']}"
    )
    print(f"NAVMESH_VOLUME {volume['min_cm']} -> {volume['max_cm']}")
    print(
        "NAVMESH_NOTE minimum agent radius is 0.20 m. Maximum stays "
        f"{AGENT_MAX_RADIUS_CM:.0f} cm because these bodies report about 23-26 cm "
        "and a 0.20 m maximum accepts move_to without moving them."
    )
    configure_spacing(people)
    marks = _marks()
    browse_text = " ".join(
        f"{index}:{tuple(round(v, 1) for v in point)}"
        for index, point in enumerate(marks["browse"], start=1)
    )
    print(
        f"MARKS staff2={tuple(round(v, 1) for v in marks['staff2'])} "
        f"register2={tuple(round(v, 1) for v in marks['staff2_register'])} "
        f"exit={tuple(round(v, 1) for v in marks['exit'])} browse={browse_text}"
    )
    director = StoreDirector(stage, people, marks, meters_per_unit, milk_paths=milk, chip_paths=chips)
    return director, before


async def main() -> dict:
    os.makedirs(VIDEO_DIR, exist_ok=True)
    print("TRACK_K start")
    print("CLIP_FILES retail_cashier_typing_loop.usd clip_shelf_audit.usd clip_reach_shelf.usd are not in this build")
    print("CLIP_BIND cashier=pose_hand FIST/POINT staff2=idle then pose_hand shoppers=reach_hand")
    director, before = await track_k_prepare()
    if not globals().get("TRACK_K_RECORD", True):
        _assert_cameras_frozen(director.stage, before)
        globals()["MARKETGEN_DIRECTOR"] = director
        print("TRACK_K_SETUP_OK")
        return {"setup": True}
    await _agents_online(director.people)
    place_people(director.stage, director.people, director.marks)
    director.prime()
    _assert_cameras_frozen(director.stage, before)
    frames = int(RECORD_SECONDS * FPS)
    # Do not save the stage. The take is the five mp4 files.
    videos = await record(director, frames)
    _assert_cameras_frozen(director.stage, before)
    print("TRACK_K_OK")
    for video in videos:
        print(f"VIDEO {video['path']} bytes={video['bytes']}")
    return {"videos": videos}


def _bootstrap() -> asyncio.Task | None:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        return loop.create_task(main())
    loop.run_until_complete(main())
    return None


if globals().get("MARKETGEN_FLOW_AUTO", True):
    _bootstrap()
