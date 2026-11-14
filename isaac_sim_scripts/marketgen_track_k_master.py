"""Cogno Valley Track K on the open ConvenienceStore01 stage.

Isaac Sim 6.0.1 / Kit 110. The stage is already open. This module never calls
open_stage(), never resets the viewport, never saves the layer, and never
starts a second Kit. It does not call IRA setup_simulation() or
start_data_generation_async(): those reopen the base USD and drop the session.

Characters are already on /World/Characters. Spawning or deleting that cast
crashes Kit, so the script binds the existing payloads and places them.

The export is centimeters, Z-up, metersPerUnit 0.01. USD cameras look down
local -Z with +Y up. A rotateXYZ(-36, 0, -180) rolls that camera. The
back-aisle mount stays the upright back-right eye.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field

import carb
import carb.settings
import omni.anim.behavior.core as bh
import omni.kit.app
import omni.kit.commands
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, Semantics, Usd, UsdGeom

OUTPUT_DIR = "isaac_sim_scripts"
VIDEO_DIR = "demo_videos"
METADATA_PATH = "metadata.json"
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "marketgen_tracking_model.pth")
PROGRESS_PATH = "/tmp/trackk_master_progress.log"
STAGE_TOKEN = "ConvenienceStore01.usd"
WORLD_ROOT = "/World"
STORE_ROOT = "/Root"
CHARACTERS_ROOT = "/World/Characters"

CHECKOUT_CAMERA = "/World/CCTV_Checkout_Shelves"
ENTRANCE_CAMERA = "/World/CCTV_Entrance_Aisle"
ENTRANCE_END_CAMERA = "/World/CCTV_Entrance_Aisle_End"
BACK_CAMERA = "/World/CCTV_Back_Aisle"
LEFT_BACK_CAMERA = "/World/CCTV_Back_Aisle_Left"
FROZEN_CAMERAS = (LEFT_BACK_CAMERA, ENTRANCE_END_CAMERA)

# Upright back-right mount. [3.25, 3.80, 2.75] m is outside: Y grows out the
# front door, so +3.80 m is in the parking lot, and (3.25, -3.80, 2.75) m is
# the near-counter spot that framed the tills. The euler also rolls the image.
BACK_EYE_CM = (1120.0, -1155.0, 275.0)
BACK_DEPRESSION_DEG = 36.0
ENTRANCE_DEPRESSION_DEG = 40.0
ENTRANCE_AIM_CM = (720.0, -280.0, 120.0)
VERTICAL_APERTURE_MM = 20.25
F_STOP = 11.0

ENTRANCE_FOCAL_MM = 14.0
ENTRANCE_HAP_MM = 36.0
CHECKOUT_FOCAL_MM = 16.0
CHECKOUT_HAP_MM = 36.0
BACK_FOCAL_MM = 16.0
BACK_HAP_MM = 36.0

# Bodies report get_radius() of about 23-26 cm. A navmesh whose maximum is
# 20 cm accepts move_to and then never moves them. Minimum is 0.20 m.
# Recast keeps a span only when its up-component is greater than cos(maxSlope).
# Slope 0 rejects the flat floor. 0.05 degrees is the smallest bake that
# still connects the aisle and the exterior exit.
AGENT_RADIUS_CM = 20.0
AGENT_MAX_RADIUS_CM = 26.0
SEPARATION_WEIGHT = 2.5
AGENT_MAX_SLOPE_DEG = 0.05
EXIT_NODE_CM = (-600.0, 0.0, 0.0)
NAVMESH_INCLUDE = "/World/NavMesh_StoreFloor"
NAVMESH_EXCLUDE = "/World/NavMesh_NotchExclude"
NAVMESH_MIN_RADIUS = "/exts/omni.anim.navigation.core/navMesh/config/agentMinRadius"
NAVMESH_MAX_RADIUS = "/exts/omni.anim.navigation.core/navMesh/config/agentMaxRadius"
NAVMESH_MAX_SLOPE = "/exts/omni.anim.navigation.core/navMesh/config/agentMaxFloorSlope"

WALK_MPS = 1.15
BRISK_MPS = 1.45
RECORD_SECONDS = 90.0
FPS = 30
VIDEO_SIZE = (1280, 720)
DWELL_SECONDS = 20.0
BEVERAGE_DWELL_SECONDS = 15.0

# Shopper-side milk and chip clusters are behind the gondolas. A tight-box
# pass measured them at occlusion 0.8-1.0 from every camera. These two faces
# are the ones the renders actually see: dairy on the entrance camera, chips
# on the left back-aisle camera. Stock 0 is this facing group, not the whole shelf.
FACING_DAIRY = (
    ("SM_MilkPacket01_19", "SM_MilkPacket01", "ShelfBox7"),
    ("SM_MilkPacket01_13", "SM_MilkPacket01", "ShelfBox7"),
    ("SM_MilkPacket01_7", "SM_MilkPacket01", "ShelfBox7"),
    ("SM_MilkPacket01_1", "SM_MilkPacket01", "ShelfBox7"),
    ("SM_MilkPacket02_79", "SM_MilkPacket02", "ShelfBox7"),
    ("SM_MilkPacket02_73", "SM_MilkPacket02", "ShelfBox7"),
    ("SM_MilkPacket02_67", "SM_MilkPacket02", "ShelfBox7"),
    ("SM_MilkPacket02_61", "SM_MilkPacket02", "ShelfBox7"),
)
FACING_SNACKS = (
    ("SM_Packet2_3_Chips_1", "SM_Packet2_3_Chips", "ShelfBox6"),
    ("SM_Packet2_3_Chips_6", "SM_Packet2_3_Chips", "ShelfBox6"),
    ("SM_Packet2_3_Chips_11", "SM_Packet2_3_Chips", "ShelfBox6"),
    ("SM_Packet2_3_Chips_16", "SM_Packet2_3_Chips", "ShelfBox6"),
    ("SM_Packet2_3_Chips_21", "SM_Packet2_3_Chips", "ShelfBox6"),
    ("SM_Packet2_3_Chips_26", "SM_Packet2_3_Chips", "ShelfBox6"),
    ("SM_Packet2_3_Chips_31", "SM_Packet2_3_Chips", "ShelfBox6"),
    ("SM_Packet2_3_Chips_36", "SM_Packet2_3_Chips", "ShelfBox6"),
    ("SM_Packet2_3_Chips_41", "SM_Packet2_3_Chips", "ShelfBox6"),
    ("SM_Packet2_3_Chips_46", "SM_Packet2_3_Chips", "ShelfBox6"),
)
MILK_ALERT = (
    "[TRACK_K_ALERT][SHELF] 8 items removed from Category:Dairy, "
    "Row_1_Front. Current Stock: 0. Ticket Status: Open."
)
CHIPS_ALERT = (
    "[TRACK_K_ALERT][SHELF] 10 items removed from Category:Snacks, "
    "Row_2_Back. Current Stock: 0. Ticket Status: Open."
)
QUEUE_ALERT = (
    "[TRACK_K_ALERT][QUEUE] Queue length 4 breaks threshold. "
    "Est. wait time: 3.5 mins. Dispatching backup."
)

PRIMARY_REGISTER = "/Root/BP_CashierArea_01/SM_CashRegister_Out_StaticMeshComponent0"
BACKUP_REGISTER = "/Root/BP_CashierArea_01/SM_CashRegister_Out2_StaticMeshComponent0"
SEMANTIC_CLASSES = ("shopper", "staff", "shelf_structure", "grocery_item")

SURVEILLANCE_TAKES = (
    (CHECKOUT_CAMERA, "CCTV_Checkout_Shelves.mp4"),
    (ENTRANCE_CAMERA, "CCTV_Entrance_Aisle.mp4"),
    (ENTRANCE_END_CAMERA, "CCTV_Entrance_Aisle_End.mp4"),
    (BACK_CAMERA, "CCTV_Back_Aisle.mp4"),
    (LEFT_BACK_CAMERA, "CCTV_Back_Aisle_Left.mp4"),
)
SINGLE_CASHIER_DIR = "single_cashier"
DYNAMIC_STAFF_DIR = "dynamic_staff"
SHELF_DELETION_DIR = "shelf_deletion"
PROTECTED_VIDEOS = (
    "cam_checkout.mp4",
    "cam_entrance.mp4",
    "cam_entrance_end.mp4",
    "cam_back_aisle.mp4",
    "cam_back_aisle_left.mp4",
    "baseline_single_cashier.mp4",
    "dynamic_staff_dispatch.mp4",
    "shelf_depletion_audit.mp4",
)


def _trace(line: str) -> None:
    print(line, flush=True)
    try:
        with open(PROGRESS_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def _enable(ext_name: str) -> None:
    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled(ext_name):
        manager.set_extension_enabled_immediate(ext_name, True)


def _hook_open_stage() -> tuple[Usd.Stage, str]:
    """Use the stage that is already open. Never calls open_stage()."""
    source = "omni.usd.get_context().get_stage"
    stage = None
    try:
        import omni.isaac.core.utils.stage as isaac_stage

        stage = isaac_stage.get_current_stage()
        source = "omni.isaac.core.utils.stage.get_current_stage"
    except Exception:
        try:
            from isaacsim.core.utils.stage import get_current_stage

            stage = get_current_stage()
            source = "isaacsim.core.utils.stage.get_current_stage"
        except Exception:
            stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open in Isaac Sim.")
    identifier = stage.GetRootLayer().identifier.replace("\\", "/")
    if STAGE_TOKEN not in identifier:
        raise RuntimeError(
            f"The open stage is {identifier}. Refusing to edit it. "
            f"Open {STAGE_TOKEN} and run this script again."
        )
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise RuntimeError("Expected a Z-up store.")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if abs(meters_per_unit - 0.01) > 1e-6:
        raise RuntimeError(
            f"Expected metersPerUnit 0.01 so camera gates are millimeters. Got {meters_per_unit}."
        )
    return stage, source


def _ira_runtime() -> str:
    """Confirm the agent extension is loaded. Do not set up or generate."""
    _enable("isaacsim.replicator.agent.core")
    _enable("omni.anim.behavior.core")
    _enable("omni.anim.navigation.core")
    _enable("omni.replicator.core")
    import isaacsim.replicator.agent.core as ira
    from isaacsim.replicator.agent.core.extension_info import get_ext_version

    banned = ("setup_simulation", "start_data_generation_async")
    for name in banned:
        if not hasattr(ira, name):
            raise RuntimeError(f"isaacsim.replicator.agent.core is missing {name}.")
    return str(get_ext_version())


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


def _camera_pose(stage: Usd.Stage, path: str) -> dict:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing camera {path}")
    camera = UsdGeom.Camera(prim)
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    eye = matrix.ExtractTranslation()
    forward = _camera_forward(prim)
    up = _camera_axis(prim, Gf.Vec3d(0.0, 1.0, 0.0))
    focal = camera.GetFocalLengthAttr().Get()
    horizontal = camera.GetHorizontalApertureAttr().Get()
    vertical = camera.GetVerticalApertureAttr().Get()
    return {
        "eye": (float(eye[0]), float(eye[1]), float(eye[2])),
        "forward": (float(forward[0]), float(forward[1]), float(forward[2])),
        "up": (float(up[0]), float(up[1]), float(up[2])),
        "depression": _depression_of(forward),
        "focal": None if focal is None else float(focal),
        "hap": None if horizontal is None else float(horizontal),
        "vap": None if vertical is None else float(vertical),
    }


def _set_optics(prim: Usd.Prim, focal_mm: float, horizontal_mm: float, vertical_mm: float) -> None:
    """Write millimeter gates. At metersPerUnit 0.01 the attributes are already millimeters."""
    camera = UsdGeom.Camera(prim)
    camera.CreateFocalLengthAttr(float(focal_mm))
    camera.CreateHorizontalApertureAttr(float(horizontal_mm))
    camera.CreateVerticalApertureAttr(float(vertical_mm))
    camera.CreateFStopAttr(F_STOP)
    read_focal = float(camera.GetFocalLengthAttr().Get())
    read_horizontal = float(camera.GetHorizontalApertureAttr().Get())
    read_vertical = float(camera.GetVerticalApertureAttr().Get())
    if abs(read_focal - focal_mm) > 0.05 or abs(read_horizontal - horizontal_mm) > 0.05:
        raise RuntimeError(
            f"{prim.GetPath()} optics read back focal={read_focal} hap={read_horizontal}, "
            f"expected {focal_mm} / {horizontal_mm}."
        )
    if abs(read_vertical - vertical_mm) > 0.05:
        raise RuntimeError(f"{prim.GetPath()} vertical aperture read back {read_vertical}.")


def _assert_pose_held(before: dict, after: dict, path: str, eyes: bool, aim: bool) -> None:
    if eyes and math.dist(before["eye"], after["eye"]) > 0.5:
        raise RuntimeError(f"{path} eye moved from {before['eye']} to {after['eye']}.")
    if aim and abs(float(before["depression"]) - float(after["depression"])) > 0.6:
        raise RuntimeError(
            f"{path} depression moved from {before['depression']:.2f} to {after['depression']:.2f}."
        )
    if aim and abs(before["up"][2]) > 0.0 and after["up"][2] < 0.5:
        raise RuntimeError(f"{path} rolled. Up {after['up']}.")


def apply_surveillance_optics(stage: Usd.Stage) -> dict:
    """Wide-angle gates on the three named cameras. Left and entrance-end stay put."""
    for path in (
        CHECKOUT_CAMERA,
        ENTRANCE_CAMERA,
        BACK_CAMERA,
        LEFT_BACK_CAMERA,
        ENTRANCE_END_CAMERA,
    ):
        if not stage.GetPrimAtPath(path):
            raise RuntimeError(f"Missing camera {path}")
    frozen = {path: _camera_pose(stage, path) for path in FROZEN_CAMERAS}
    checkout_before = _camera_pose(stage, CHECKOUT_CAMERA)
    back_before = _camera_pose(stage, BACK_CAMERA)

    entrance = stage.GetPrimAtPath(ENTRANCE_CAMERA)
    entrance_before = _camera_pose(stage, ENTRANCE_CAMERA)
    eye = Gf.Vec3d(*entrance_before["eye"])
    target = Gf.Vec3d(*ENTRANCE_AIM_CM)
    horizontal = _normalize(Gf.Vec3d(target[0] - eye[0], target[1] - eye[1], 0.0))
    depression = math.radians(ENTRANCE_DEPRESSION_DEG)
    aimed = _normalize(
        Gf.Vec3d(
            horizontal[0] * math.cos(depression),
            horizontal[1] * math.cos(depression),
            -math.sin(depression),
        )
    )
    xform = UsdGeom.Xformable(entrance)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(_look_matrix(eye, eye + aimed * 500.0))
    _set_optics(entrance, ENTRANCE_FOCAL_MM, ENTRANCE_HAP_MM, VERTICAL_APERTURE_MM)
    entrance_after = _camera_pose(stage, ENTRANCE_CAMERA)
    _assert_pose_held(entrance_before, entrance_after, ENTRANCE_CAMERA, True, False)
    if entrance_after["up"][2] < 0.5 or entrance_after["forward"][1] >= 0.0:
        raise RuntimeError(
            f"Entrance camera is rolled or looking out of the store. "
            f"Forward {entrance_after['forward']} up {entrance_after['up']}."
        )
    if abs(entrance_after["depression"] - ENTRANCE_DEPRESSION_DEG) > 0.8:
        raise RuntimeError(f"Entrance depression is {entrance_after['depression']:.2f}.")

    checkout = stage.GetPrimAtPath(CHECKOUT_CAMERA)
    _set_optics(checkout, CHECKOUT_FOCAL_MM, CHECKOUT_HAP_MM, VERTICAL_APERTURE_MM)
    checkout_after = _camera_pose(stage, CHECKOUT_CAMERA)
    _assert_pose_held(checkout_before, checkout_after, CHECKOUT_CAMERA, True, True)

    back = stage.GetPrimAtPath(BACK_CAMERA)
    back_eye_gap = math.dist(back_before["eye"], BACK_EYE_CM)
    back_upright = back_before["up"][2] > 0.5 and back_before["forward"][1] > 0.7
    if back_eye_gap > 5.0 or not back_upright:
        pitch = math.radians(BACK_DEPRESSION_DEG)
        forward = _normalize(Gf.Vec3d(0.0, math.cos(pitch), -math.sin(pitch)))
        eye = Gf.Vec3d(*BACK_EYE_CM)
        xform = UsdGeom.Xformable(back)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(_look_matrix(eye, eye + forward))
    _set_optics(back, BACK_FOCAL_MM, BACK_HAP_MM, VERTICAL_APERTURE_MM)
    back_after = _camera_pose(stage, BACK_CAMERA)
    if math.dist(back_after["eye"], BACK_EYE_CM) > 1.0:
        raise RuntimeError(f"Back camera eye is {back_after['eye']}, expected {BACK_EYE_CM}.")
    if back_after["up"][2] < 0.5 or back_after["forward"][1] < 0.7:
        raise RuntimeError(f"Back camera is not upright down the aisle. {back_after}")
    if abs(back_after["depression"] - BACK_DEPRESSION_DEG) > 1.0:
        raise RuntimeError(f"Back depression is {back_after['depression']:.2f}.")

    for path, before in frozen.items():
        after = _camera_pose(stage, path)
        _assert_pose_held(before, after, path, True, True)
        if abs(float(before["focal"] or 0.0) - float(after["focal"] or 0.0)) > 0.05:
            raise RuntimeError(f"{path} focal changed from {before['focal']} to {after['focal']}.")
        if abs(float(before["hap"] or 0.0) - float(after["hap"] or 0.0)) > 0.05:
            raise RuntimeError(f"{path} aperture changed from {before['hap']} to {after['hap']}.")
    return {
        "entrance": entrance_after,
        "checkout": checkout_after,
        "back": back_after,
        "frozen": {path: _camera_pose(stage, path) for path in FROZEN_CAMERAS},
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


def _define_exclude(stage: Usd.Stage, path: str) -> Usd.Prim:
    import NavSchema

    existing = stage.GetPrimAtPath(path)
    if existing and existing.IsValid():
        prim = existing
    else:
        prim = NavSchema.NavMeshVolume.Define(stage, path).GetPrim()
    volume = NavSchema.NavMeshVolume(prim)
    attr = volume.GetNavVolumeTypeAttr()
    if not attr:
        attr = volume.CreateNavVolumeTypeAttr()
    attr.Set("Exclude")
    return prim


def _exclude_tree(prim: Usd.Prim) -> list[str]:
    """NavMeshExcludeAPI on an xform does not exclude child meshes."""
    import NavSchema

    meshes = []
    for item in Usd.PrimRange(prim):
        if not item.HasAPI(NavSchema.NavMeshExcludeAPI):
            item.ApplyAPI(NavSchema.NavMeshExcludeAPI)
        if item.IsA(UsdGeom.Mesh):
            meshes.append(str(item.GetPath()))
    return meshes


def _open_service_leaf(stage: Usd.Stage) -> None:
    """Park the fridge leaf in the backroom so the walk line at x=150 stays clear.

    The closed pose is RotZ(-90). Another -90 degrees is RotZ(-180), which
    lays the leaf along x=52..81, y=-940..-792. Setting that quaternion
    outright keeps a second run from spinning the door again.
    """
    prim = stage.GetPrimAtPath("/Root/SM_EntDoor_L")
    if not prim or not prim.IsValid():
        raise RuntimeError("Missing /Root/SM_EntDoor_L.")
    orient = None
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpName() == "xformOp:orient":
            orient = op
            break
    if orient is None:
        raise RuntimeError("/Root/SM_EntDoor_L has no orient op.")
    orient.Set(Gf.Quatd(0.0, Gf.Vec3d(0.0, 0.0, -1.0)))


def author_backroom_floor(stage: Usd.Stage) -> dict:
    """Copy the store floor box onto the root layer across the empty notch.

    The back room has no floor tiles. Session-layer meshes never enter the
    baker's triangle soup. A single root-layer quad is counted as terrain and
    then dropped. Closed copies of SM_Floor_11, with world-space points and
    no scale op, are the surfaces the baker keeps.
    """
    source = UsdGeom.Mesh(stage.GetPrimAtPath("/Root/SupermarketFramework/FloorComponent_1_7/SM_Floor_11"))
    if not source.GetPrim().IsValid():
        raise RuntimeError("Missing the store floor tile used as the backroom template.")
    points = list(source.GetPointsAttr().Get())
    counts = list(source.GetFaceVertexCountsAttr().Get())
    indices = list(source.GetFaceVertexIndicesAttr().Get())
    normals = source.GetNormalsAttr().Get()
    interpolation = source.GetNormalsInterpolation()
    root = stage.GetRootLayer()
    centers = [(x, y) for x in (50.0, 150.0, 250.0) for y in (-1050.0, -950.0, -850.0, -780.0)]
    with Usd.EditContext(stage, root):
        UsdGeom.Xform.Define(stage, "/World/NavMesh_BackroomTiles")
        for x_cm, y_cm in centers:
            path = f"/World/NavMesh_BackroomTiles/Tile_x{int(x_cm)}_y{int(abs(y_cm))}"
            existing = stage.GetPrimAtPath(path)
            if existing and existing.IsValid():
                stage.RemovePrim(path)
            mesh = UsdGeom.Mesh.Define(stage, path)
            world = [Gf.Vec3f(point[0] + x_cm, point[1] + y_cm, point[2]) for point in points]
            mesh.CreatePointsAttr(world)
            mesh.CreateFaceVertexCountsAttr(counts)
            mesh.CreateFaceVertexIndicesAttr(indices)
            if normals is not None:
                mesh.CreateNormalsAttr(normals)
                mesh.SetNormalsInterpolation(interpolation)
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateExtentAttr(UsdGeom.PointBased(mesh.GetPrim()).ComputeExtent(world))
    sample = stage.GetPrimAtPath("/World/NavMesh_BackroomTiles/Tile_x150_y1050")
    layer = sample.GetPrimStack()[-1].layer.identifier if sample and sample.IsValid() else ""
    if STAGE_TOKEN not in layer:
        raise RuntimeError(f"Backroom floor tile is not on the root layer: {layer}")
    return {"tiles": len(centers), "layer": layer}


def _exclude_service_door(stage: Usd.Stage) -> list[str]:
    """Keep the service door and its leaf from blocking the backroom throat."""
    framework = stage.GetPrimAtPath("/Root/SupermarketFramework")
    leaf = stage.GetPrimAtPath("/Root/SM_EntDoor_L")
    if not framework or not framework.IsValid():
        raise RuntimeError("Missing /Root/SupermarketFramework.")
    if not leaf or not leaf.IsValid():
        raise RuntimeError("Missing /Root/SM_EntDoor_L.")
    found = []
    with Usd.EditContext(stage, stage.GetRootLayer()):
        door_roots = [prim for prim in Usd.PrimRange(framework) if prim.GetName() == "DoorComponent_1"]
        if not door_roots:
            raise RuntimeError("DoorComponent_1 is not on the open stage.")
        for prim in door_roots:
            found.extend(_exclude_tree(prim))
        found.extend(_exclude_tree(leaf))
        _open_service_leaf(stage)
    if not any("SM_EntDoor_L" in path for path in found):
        raise RuntimeError("The service-door leaf meshes were not excluded.")
    return found


def extend_walkable_volume(stage: Usd.Stage, meters_per_unit: float) -> dict:
    """Include the exit and open a door-sized hole through the backroom exclude."""
    prim = stage.GetPrimAtPath(NAVMESH_INCLUDE)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing {NAVMESH_INCLUDE}.")
    minimum = Gf.Vec3d(-720.0, -1205.0, -20.0)
    maximum = Gf.Vec3d(1305.0, 80.0, 220.0)
    _fit_nav_volume(prim, minimum, maximum, meters_per_unit)
    # The old notch exclude covered the whole back room, so Staff_02 could not
    # path through DoorComponent_1 (x=62..238, y about -816..-796). These boxes
    # keep the exterior cap and the wall beside the door blocked, and leave the
    # standing pad plus the door throat walkable.
    strips = {
        NAVMESH_EXCLUDE: (Gf.Vec3d(-8.0, -1208.0, -20.0), Gf.Vec3d(308.0, -1100.0, 220.0)),
        NAVMESH_EXCLUDE + "_left": (Gf.Vec3d(-8.0, -1100.0, -20.0), Gf.Vec3d(40.0, -792.0, 220.0)),
        NAVMESH_EXCLUDE + "_right": (Gf.Vec3d(260.0, -1100.0, -20.0), Gf.Vec3d(308.0, -792.0, 220.0)),
        NAVMESH_EXCLUDE + "_door_left": (Gf.Vec3d(40.0, -820.0, -20.0), Gf.Vec3d(70.0, -792.0, 220.0)),
        NAVMESH_EXCLUDE + "_door_right": (Gf.Vec3d(230.0, -820.0, -20.0), Gf.Vec3d(260.0, -792.0, 220.0)),
    }
    for path, (low, high) in strips.items():
        _fit_nav_volume(_define_exclude(stage, path), low, high, meters_per_unit)
    return {
        "include": [minimum[0], minimum[1], minimum[2], maximum[0], maximum[1], maximum[2]],
        "door_throat_cm": [70.0, -820.0, 230.0, -760.0],
        "backroom_pad_cm": [40.0, -1100.0, 260.0, -820.0],
    }


def set_tight_agent() -> dict:
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

    settings = carb.settings.get_settings()
    settings.set("/exts/omni.anim.navigation.core/navMesh/bakeWaitTimeoutSeconds", 0.0)
    settings.set("/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
    for key in (
        "/exts/omni.anim.navigation.core/navMesh/cache/enabled",
        "/persistent/exts/omni.anim.navigation.core/navMesh/cache/enabled",
        "/exts/omni.anim.navigation.core/navMesh/cache/enableUjitso",
        "/persistent/exts/omni.anim.navigation.core/navMesh/cache/enableUjitso",
    ):
        settings.set(key, False)
    # A disconnected pad smaller than the default 200 cm island radius is
    # deleted after the bake. 10 cm keeps the new floor if the door weld is thin.
    settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentMinIslandRadius", 10.0)
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/agentMinIslandRadius", 10.0)
    app = omni.kit.app.get_app()
    await app.next_update_async()
    await app.next_update_async()
    interface = nav.acquire_interface()
    try:
        interface.clear_cache_dir()
    except Exception:
        pass
    baked = bool(interface.start_navmesh_baking_and_wait())
    navmesh = interface.get_navmesh()
    if navmesh is None:
        raise RuntimeError("NavMesh bake did not produce a mesh.")
    return {"baked": baked, "navmesh": navmesh}


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


def _path_length(navmesh, start: tuple[float, float, float], end: tuple[float, float, float], agent=None) -> float | None:
    kwargs = {}
    if agent is not None:
        kwargs["agent"] = agent
    a = navmesh.query_closest_point(target=start, **kwargs)
    b = navmesh.query_closest_point(target=end, **kwargs)
    if not a or not b or a[0] is None or b[0] is None:
        return None
    found = navmesh.query_shortest_path(start_pos=a[0], end_pos=b[0], **kwargs)
    if found is None or not hasattr(found, "length"):
        return None
    return round(float(found.length()), 1)


def verify_navmesh(navmesh) -> dict:
    from omni.anim.navigation.core import NavAgentDesc

    probes = {
        "dairy_aisle": _probe_nav(navmesh, (760.0, -430.0, 30.0)),
        "snack_aisle": _probe_nav(navmesh, (760.0, -740.0, 30.0)),
        "entry_lane": _probe_nav(navmesh, (420.0, -180.0, 30.0)),
        "backroom": _probe_nav(navmesh, (150.0, -1050.0, 30.0)),
        "door_threshold": _probe_nav(navmesh, (150.0, -800.0, 30.0)),
        "door_outside": _probe_nav(navmesh, (150.0, -740.0, 30.0)),
        "counter2": _probe_nav(navmesh, (200.0, -590.0, 30.0)),
        "exit_node": _probe_nav(navmesh, (EXIT_NODE_CM[0], EXIT_NODE_CM[1], 30.0)),
        "back_cap": _probe_nav(navmesh, (150.0, -1180.0, 30.0)),
    }
    for name in ("dairy_aisle", "snack_aisle", "entry_lane", "backroom", "door_threshold", "door_outside", "counter2"):
        hit = probes[name]
        if hit is None or hit["distance_cm"] > 80.0:
            raise RuntimeError(f"{name} is not walkable: {hit}")
    exit_hit = probes["exit_node"]
    if exit_hit is None or exit_hit["distance_cm"] > 80.0:
        raise RuntimeError(f"Exit node {EXIT_NODE_CM} is not on the navmesh: {exit_hit}")
    cap = probes["back_cap"]
    if cap is not None and cap["distance_cm"] < 40.0:
        raise RuntimeError(f"Backroom exterior cap was baked as walkable: {cap}")
    body = NavAgentDesc(radius=AGENT_MAX_RADIUS_CM, height=200.0, collision_gap=0.0)
    door_path = _path_length(navmesh, (150.0, -1050.0, 20.0), (200.0, -590.0, 20.0), body)
    exit_path = _path_length(navmesh, (390.0, -40.0, 20.0), (EXIT_NODE_CM[0], EXIT_NODE_CM[1], 20.0), body)
    aisle_path = _path_length(navmesh, (420.0, -180.0, 20.0), (760.0, -430.0, 20.0), body)
    if door_path is None:
        raise RuntimeError("No 26 cm path from the backroom through the service door to counter 2.")
    if exit_path is None or aisle_path is None:
        raise RuntimeError(f"Shopper paths missing. aisle={aisle_path} exit={exit_path}")
    return {
        "probes": probes,
        "backroom_to_counter2_cm": door_path,
        "door_to_exit_cm": exit_path,
        "entry_to_aisle_cm": aisle_path,
    }


class MetadataCatalog:
    """Shelf and item registry from metadata.json. Coordinates are centimeters."""

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or "Items" not in payload or "ShelfBoxes" not in payload:
            raise RuntimeError(f"{path} has no Items/ShelfBoxes registry.")
        self.path = path
        self.items = list(payload["Items"])
        self.boxes = list(payload["ShelfBoxes"])
        self.by_id = {item["ID"]: item for item in self.items}
        self.shelf_class: dict[str, tuple[str, str, str]] = {}
        for box in self.boxes:
            for shelf in box.get("Shelves") or []:
                self.shelf_class[shelf["ID"]] = (
                    box["ID"],
                    str(shelf.get("zone type") or ""),
                    str(shelf.get("GoodsCategory") or ""),
                )

    def cluster(
        self,
        box_id: str,
        name: str,
        z_range: tuple[float, float],
        y_range: tuple[float, float],
        anchor: tuple[float, float, float],
        count: int,
        span: tuple[float, float, float],
    ) -> list[str]:
        found = []
        for item in self.items:
            if item.get("ShelfBoxID") != box_id or item.get("Name") != name:
                continue
            x, y, z = (float(v) for v in item["Location"])
            if not (z_range[0] <= z <= z_range[1] and y_range[0] <= y <= y_range[1]):
                continue
            found.append((x, y, z, f"{STORE_ROOT}/{item['ID']}"))
        ranked = sorted(
            found,
            key=lambda row: (row[0] - anchor[0]) ** 2 + (row[1] - anchor[1]) ** 2 + (row[2] - anchor[2]) ** 2,
        )
        chosen = []
        for row in ranked:
            if not chosen:
                chosen.append(row)
            else:
                origin = chosen[0]
                if (
                    abs(row[0] - origin[0]) <= span[0]
                    and abs(row[1] - origin[1]) <= span[1]
                    and abs(row[2] - origin[2]) <= span[2]
                ):
                    chosen.append(row)
            if len(chosen) == count:
                break
        paths = [row[3] for row in chosen]
        if len(paths) < count:
            raise RuntimeError(
                f"metadata.json resolved {len(paths)} of {count} {name} prims on {box_id} "
                f"(scanned {len(found)})."
            )
        return paths

    def facing(self, spec: tuple[tuple[str, str, str], ...]) -> list[str]:
        """Prims whose metadata name and shelf match the camera-facing removal."""
        paths = []
        for item_id, name, box_id in spec:
            item = self.by_id.get(item_id)
            if item is None or item.get("Name") != name or item.get("ShelfBoxID") != box_id:
                raise RuntimeError(f"metadata.json has no {name} {item_id} on {box_id}.")
            paths.append(f"{STORE_ROOT}/{item_id}")
        return paths


def _require_prims(stage: Usd.Stage, paths: list[str]) -> list[str]:
    resolved = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"metadata prim is not on the open stage: {path}")
        if not prim.IsActive():
            prim.SetActive(True)
        resolved.append(path)
    return resolved


def _label_prim(prim: Usd.Prim, label: str) -> None:
    semantic = Semantics.SemanticsAPI.Get(prim, "class")
    if not semantic:
        semantic = Semantics.SemanticsAPI.Apply(prim, "class")
        semantic.CreateSemanticTypeAttr()
        semantic.CreateSemanticDataAttr()
    semantic.GetSemanticTypeAttr().Set("class")
    semantic.GetSemanticDataAttr().Set(label)


def apply_semantics(stage: Usd.Stage, catalog: MetadataCatalog, people: list[Person]) -> dict:
    """Map metadata classification onto the four Track K semantic classes."""
    counts = {name: 0 for name in SEMANTIC_CLASSES}
    missing_items = 0
    with Sdf.ChangeBlock():
        for person in people:
            prim = stage.GetPrimAtPath(person.payload_path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"Missing cast prim {person.payload_path}")
            label = "staff" if person.role.startswith("Staff_") else "shopper"
            _label_prim(prim, label)
            counts[label] += 1
        for box in catalog.boxes:
            prim = stage.GetPrimAtPath(f"{STORE_ROOT}/{box['ID']}")
            if prim and prim.IsValid():
                _label_prim(prim, "shelf_structure")
                prim.SetCustomDataByKey("marketgen:class", "shelf_structure")
                counts["shelf_structure"] += 1
            for shelf in box.get("Shelves") or []:
                shelf_prim = stage.GetPrimAtPath(f"{STORE_ROOT}/{shelf['ID']}")
                if not shelf_prim or not shelf_prim.IsValid():
                    continue
                _label_prim(shelf_prim, "shelf_structure")
                shelf_prim.SetCustomDataByKey("marketgen:category", str(shelf.get("GoodsCategory") or ""))
                shelf_prim.SetCustomDataByKey("marketgen:zone", str(shelf.get("zone type") or ""))
                counts["shelf_structure"] += 1
        for item in catalog.items:
            prim = stage.GetPrimAtPath(f"{STORE_ROOT}/{item['ID']}")
            if not prim or not prim.IsValid():
                missing_items += 1
                continue
            _label_prim(prim, "grocery_item")
            shelf_id = str(item.get("ShelfID") or "")
            category = catalog.shelf_class.get(shelf_id, ("", "", ""))[2]
            prim.SetCustomDataByKey("marketgen:class", "grocery_item")
            prim.SetCustomDataByKey("marketgen:category", category)
            prim.SetCustomDataByKey("marketgen:name", str(item.get("Name") or ""))
            counts["grocery_item"] += 1
    if counts["shopper"] != 8 or counts["staff"] != 2:
        raise RuntimeError(f"Semantic cast labels are wrong: {counts}")
    if counts["shelf_structure"] < 7 or counts["grocery_item"] < 1000:
        raise RuntimeError(f"Semantic registry is incomplete: {counts} missing_items={missing_items}")
    return {"counts": counts, "missing_items": missing_items}


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
    hand_slot: int = -1
    reach_sent: bool = False


def _entry_seconds(index: int) -> float:
    """Shoppers 1-3 at 0s, 4-6 at 12s, 7-8 at 24s."""
    if index <= 3:
        return 0.0
    if index <= 6:
        return 12.0
    return 24.0


def _skelroot(stage: Usd.Stage, prim: Usd.Prim) -> Usd.Prim:
    """Return the first SkelRoot under prim. A dead branch must not raise."""
    del stage
    for descendant in Usd.PrimRange(prim):
        if descendant.GetTypeName() == "SkelRoot":
            return descendant
    raise RuntimeError(f"No SkelRoot under {prim.GetPath()}")


def _world_height(prim: Usd.Prim) -> float:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], True)
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if bounds.IsEmpty():
        return 0.0
    return float(bounds.GetSize()[2])


def bind_cast(stage: Usd.Stage) -> list[Person]:
    people: list[Person] = []
    roles = ["Staff_01", "Staff_02"] + [f"Shopper_{index:02d}" for index in range(1, 9)]
    for role in roles:
        payload = stage.GetPrimAtPath(f"{CHARACTERS_ROOT}/{role}")
        if not payload or not payload.IsValid():
            raise RuntimeError(f"Missing {CHARACTERS_ROOT}/{role}. Refusing to spawn a second cast.")
        if not payload.IsActive():
            payload.SetActive(True)
        skel = _skelroot(stage, payload)
        children = payload.GetChildren()
        folder = children[0].GetName() if children else role
        index = int(role.split("_")[1]) if role.startswith("Shopper_") else 0
        height = _world_height(payload)
        people.append(
            Person(
                role=role,
                folder=folder,
                payload_path=str(payload.GetPath()),
                skel_path=str(skel.GetPath()),
                agent=None,
                height_cm=round(height, 1) if 140.0 <= height <= 220.0 else 0.0,
                entry_s=_entry_seconds(index),
                shopper_index=index,
            )
        )
    return people


def _set_cast_active(stage: Usd.Stage, people: list[Person], active: bool) -> None:
    for person in people:
        prim = stage.GetPrimAtPath(person.payload_path)
        if prim and prim.IsValid():
            prim.SetActive(active)
            if active:
                UsdGeom.Imageable(prim).MakeVisible()
        person.gone = not active
        person.hidden = not active


def _restore_take(stage: Usd.Stage, people: list[Person], product_paths: list[str]) -> None:
    for person in people:
        prim = stage.GetPrimAtPath(person.payload_path)
        if prim and prim.IsValid():
            prim.SetActive(True)
            UsdGeom.Imageable(prim).MakeVisible()
        person.gone = False
        person.hidden = False
        person.notes = []
        person.goal = None
        person.phase = "parked"
        person.phase_until = -1.0
        person.hand_slot = -1
        person.reach_sent = False
        person.clip = ""
    for path in product_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            prim.SetActive(True)


def _vec3(point: tuple[float, float, float]) -> carb.Float3:
    return carb.Float3(float(point[0]), float(point[1]), float(point[2]))


def _cancel(agent) -> None:
    task_id = agent.get_action_task_id()
    if task_id:
        agent.cancel_task(task_id)


def _position(agent) -> tuple[float, float, float]:
    pos = agent.get_world_translation()
    return float(pos[0]), float(pos[1]), float(pos[2])


def _close(agent, goal: tuple[float, float, float], radius_cm: float = 70.0) -> bool:
    pos = _position(agent)
    return math.hypot(pos[0] - goal[0], pos[1] - goal[1]) <= radius_cm


def _speed(agent, meters_per_unit: float, meters_per_second: float) -> None:
    agent.set_speed(meters_per_second / meters_per_unit)


def _hand_ok(task_id) -> bool:
    return task_id is not None and task_id != bh.BEHAVIOR_TASK_ID_INVALID


def _reach_shelf(person: Person, duration: float) -> str:
    pos = _position(person.agent)
    target = (
        pos[0] + person.face[0] * 45.0,
        pos[1] + person.face[1] * 45.0,
        125.0 if person.shopper_index <= 3 else 40.0,
    )
    try:
        task_id = person.agent.reach_hand(
            hand_usage=bh.BehaviorHandUsage.RIGHT_HAND,
            target=_vec3(target),
            duration=float(duration),
        )
    except Exception as exc:
        _trace(f"REACH_HAND_FAIL {person.role} {exc}")
        return ""
    if not _hand_ok(task_id):
        return ""
    return "reach_hand"


def _tap_keys(person: Person, t: float) -> str:
    """Fist/point loop. retail_cashier_typing_loop.usd is not in this build."""
    slot = int(t / 0.35)
    if person.hand_slot == slot:
        return "pose_hand"
    person.hand_slot = slot
    hand = bh.BehaviorHandUsage.RIGHT_HAND if slot % 2 == 0 else bh.BehaviorHandUsage.LEFT_HAND
    preset = bh.BehaviorHandPosePreset.FIST if slot % 2 == 0 else bh.BehaviorHandPosePreset.POINT
    try:
        task_id = person.agent.pose_hand(hand_usage=hand, preset=preset, duration=0.4)
    except Exception as exc:
        _trace(f"POSE_HAND_FAIL {person.role} {exc}")
        return ""
    if not _hand_ok(task_id):
        return ""
    return "pose_hand"


def _walk(person: Person, goal: tuple[float, float, float], meters_per_unit: float, speed: float, allow_teleport: bool) -> bool:
    _cancel(person.agent)
    _speed(person.agent, meters_per_unit, speed)
    task_id = person.agent.move_to(_vec3(goal), auto_brake=True)
    person.goal = goal
    if task_id == bh.BEHAVIOR_TASK_ID_INVALID:
        _trace(f"WALK_REJECT {person.role} -> {tuple(round(v, 1) for v in goal)}")
        if allow_teleport:
            person.agent.teleport(_vec3(goal), _vec3(person.face))
        return False
    return True


def configure_spacing(people: list[Person]) -> dict:
    """Request separation weight 2.5. This Kit build has no such setter.

    Auto avoidance is enabled. The value 2.5 is not written into avoidance mass,
    because that mass is a different control and is left at the extension default.
    """
    settings = carb.settings.get_settings()
    settings.set("/exts/omni.anim.behavior.core/enableAutoAvoidance", True)
    sample = next((person.agent for person in people if person.agent is not None), None)
    agent_names: list[str] = []
    module_names = [name for name in dir(bh) if "separ" in name.lower() or "weight" in name.lower()]
    nav_names: list[str] = []
    try:
        import omni.anim.navigation.core as nav

        nav_names = [
            name
            for name in dir(nav.acquire_interface())
            if "separ" in name.lower() or "weight" in name.lower()
        ]
    except Exception as exc:
        nav_names = [f"unavailable:{exc}"]
    setter = None
    if sample is not None:
        agent_names = [name for name in dir(sample) if "separ" in name.lower() or "weight" in name.lower()]
        for name in agent_names:
            if name.startswith("set_"):
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
        _trace(
            f"SEPARATION_WEIGHT {SEPARATION_WEIGHT} has no setter on this Kit build "
            f"(agent={agent_names} module={module_names} nav={nav_names}). "
            "Auto avoidance is enabled. NavMesh minimum radius is 0.20 m and "
            f"maximum stays {AGENT_MAX_RADIUS_CM:.0f} cm so the bodies can walk."
        )
    else:
        _trace(f"SEPARATION_WEIGHT set via {setter}({SEPARATION_WEIGHT}) on {applied} agents")
    return {"setter": setter, "applied": applied, "avoidance": True}


async def _wait_frames(count: int) -> None:
    app = omni.kit.app.get_app()
    for _ in range(count):
        await app.next_update_async()


async def _agents_online(people: list[Person], frames: int = 180) -> None:
    import BehaviorSchema
    import omni.timeline

    stage = omni.usd.get_context().get_stage()
    for person in people:
        if person.gone:
            continue
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


def _snap(point: tuple[float, float, float]) -> tuple[float, float, float]:
    import omni.anim.navigation.core as nav
    from omni.anim.navigation.core import NavAgentDesc

    mesh = nav.acquire_interface().get_navmesh()
    if mesh is None:
        raise RuntimeError("NavMesh is missing.")
    body = NavAgentDesc(radius=AGENT_MAX_RADIUS_CM, height=200.0, collision_gap=0.0)
    closest, _island = mesh.query_closest_point(target=_vec3((point[0], point[1], 30.0)), agent=body)
    if closest is None:
        raise RuntimeError(f"No navmesh near {point}")
    return (float(closest[0]), float(closest[1]), float(closest[2]))


def _exit_goal() -> tuple[float, float, float]:
    """A point on the mesh at or beyond the exterior parking node."""
    import omni.anim.navigation.core as nav

    mesh = nav.acquire_interface().get_navmesh()
    best = None
    for x_cm in (-700.0, -660.0, -640.0, -620.0, -600.0):
        hit = _probe_nav(mesh, (x_cm, 0.0, 30.0))
        if hit is None or hit["distance_cm"] > 50.0:
            continue
        if hit["hit"][0] <= -590.0:
            best = tuple(hit["hit"])
            break
    if best is None:
        best = _snap((EXIT_NODE_CM[0], EXIT_NODE_CM[1], 30.0))
    gap = math.hypot(best[0] - EXIT_NODE_CM[0], best[1] - EXIT_NODE_CM[1])
    if best[0] > -580.0 or gap > 160.0:
        raise RuntimeError(f"Exit goal {best} does not reach parking {EXIT_NODE_CM}.")
    return (float(best[0]), float(best[1]), float(best[2]))


def build_marks() -> dict:
    queue = [
        (400.0, -345.0, 1.8),
        (400.0, -435.0, 1.8),
        (400.0, -525.0, 1.8),
        (400.0, -615.0, 1.8),
        (400.0, -705.0, 1.8),
    ]
    backup = [(400.0, -590.0, 1.8), (400.0, -680.0, 1.8)]
    door = [(300.0, -90.0, 1.8), (400.0, -90.0, 1.8), (500.0, -90.0, 1.8)]
    park = [
        (180.0, 60.0, 1.8),
        (280.0, 60.0, 1.8),
        (560.0, 60.0, 1.8),
        (700.0, 60.0, 1.8),
        (860.0, 60.0, 1.8),
    ]
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
    backroom = _snap((150.0, -1050.0, 1.8))
    if backroom[1] > -900.0 or backroom[0] < 40.0 or backroom[0] > 260.0:
        raise RuntimeError(f"Staff_02 backroom snap {backroom} is not inside the service room.")
    door_out = _snap((150.0, -740.0, 1.8))
    return {
        "staff1": _snap((200.0, -345.0, 1.8)),
        "staff2": backroom,
        "staff2_door": door_out,
        "staff2_register": _snap((200.0, -590.0, 1.8)),
        "counter1": _snap((400.0, -345.0, 1.8)),
        "queue": [_snap(point) for point in queue],
        "backup": [_snap(point) for point in backup],
        "door": [_snap(point) for point in door],
        "park": [_snap(point) for point in park],
        "browse": [_snap(point) for point in browse],
        "exit": _exit_goal(),
    }


def _face_for(index: int) -> tuple[float, float, float]:
    if index <= 3:
        return (0.0, 1.0, 0.0)
    if index <= 6:
        return (0.0, -1.0, 0.0)
    return (1.0, 0.0, 0.0)


def _focus_for(index: int, stand: tuple[float, float, float]) -> tuple[float, float, float]:
    if index <= 3:
        return (stand[0], stand[1] + 70.0, 125.0)
    if index <= 6:
        return (stand[0], stand[1] - 70.0, 40.0)
    return (stand[0] + 70.0, stand[1], 140.0)


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
    staff2.face = (0.0, 1.0, 0.0)
    staff2.focus = (150.0, -800.0, 140.0)
    staff2.agent.teleport(_vec3(marks["staff2"]), _vec3(staff2.face))
    staff2.phase = "idle"

    for person in people:
        if person.shopper_index <= 0:
            continue
        person.face = _face_for(person.shopper_index)
        if person.entry_s <= 0.0:
            stand = marks["door"][person.shopper_index - 1]
            person.phase = "entering"
            UsdGeom.Imageable(stage.GetPrimAtPath(person.payload_path)).MakeVisible()
            person.hidden = False
        else:
            stand = marks["park"][person.shopper_index - 4]
            person.phase = "parked"
            UsdGeom.Imageable(stage.GetPrimAtPath(person.payload_path)).MakeInvisible()
            person.hidden = True
        person.agent.teleport(_vec3(stand), _vec3((0.0, -1.0, 0.0)))


class StoreDirector:
    """One 90-second clock. dispatch=False keeps the single-cashier queue."""

    def __init__(
        self,
        stage: Usd.Stage,
        people: list[Person],
        marks: dict,
        meters_per_unit: float,
        milk_paths: list[str],
        chip_paths: list[str],
        scenario: str,
    ):
        self.stage = stage
        self.people = people
        self.by_role = {person.role: person for person in people}
        self.marks = marks
        self.mpu = meters_per_unit
        self.milk_paths = list(milk_paths)
        self.chip_paths = list(chip_paths)
        self.scenario = scenario
        self.dispatch = scenario == "dispatch"
        self.milk_done = False
        self.chips_done = False
        self.queue_done = False
        self.split_done = False

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
        self.milk_done = False
        self.chips_done = False
        self.queue_done = False
        self.split_done = False
        staff1 = self.by_role["Staff_01"]
        staff2 = self.by_role["Staff_02"]
        staff1.hand_slot = -1
        staff1.phase_until = -1.0
        staff2.phase = "idle"
        staff2.phase_until = -1.0
        self._hold_billing(staff1, 0.0)
        self._hold_idle(staff2, 0.0)
        configure_spacing(self.people)
        _trace(f"CAST_ACTION scenario={self.scenario} billing=pose_hand staff2=idle shoppers=reach_hand")

    def update(self, t: float) -> None:
        self.rebind()
        if self.dispatch and t >= 25.0 and not self.milk_done:
            self._hide_products(self.milk_paths, MILK_ALERT, len(self.milk_paths))
            self.milk_done = True
        if self.dispatch and t >= 40.0 and not self.chips_done:
            self._hide_products(self.chip_paths, CHIPS_ALERT, len(self.chip_paths))
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
        if self.dispatch and t >= 50.0 and not self.queue_done:
            measured = self._queue_count()
            _trace(QUEUE_ALERT)
            _trace(f"QUEUE_MEASURED length={measured} phases={self._queue_phases()}")
            self.queue_done = True
            self._split_lane()

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
        if not self.dispatch or t < 50.0 or not self.queue_done:
            if person.phase not in ("to_door", "to_register", "backup_till"):
                person.phase = "idle"
                self._hold_idle(person, t)
            elif person.phase == "backup_till":
                self._hold_billing(person, t)
            return
        if person.phase == "backup_till":
            self._hold_billing(person, t)
            return
        if person.phase == "to_door" and _close(person.agent, self.marks["staff2_door"], 80.0):
            person.phase = "to_register"
            person.face = (1.0, 0.0, 0.0)
            _walk(person, self.marks["staff2_register"], self.mpu, BRISK_MPS, allow_teleport=False)
            _trace("STAFF_02 through the door, walking to counter 2")
            return
        if person.phase == "to_register" and _close(person.agent, self.marks["staff2_register"], 70.0):
            person.phase = "backup_till"
            person.phase_until = t
            person.hand_slot = -1
            person.face = (1.0, 0.0, 0.0)
            person.focus = (360.0, self.marks["staff2_register"][1], 115.0)
            person.agent.idle(facing=_vec3(person.face))
            _tap_keys(person, t)
            _trace(f"STAFF_02 typing at counter 2 t={t:.1f}")

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
        if index == 1 and t >= 32.0 and person.phase in ("to_shelf", "dwell", "hold"):
            self._send_queue(person, self.marks["counter1"], "to_counter")
            return
        if index in (2, 3, 4, 5) and t >= 40.0 and person.phase in ("to_shelf", "dwell", "hold"):
            self._send_queue(person, self.marks["queue"][index - 1], "to_counter")
            return
        if person.phase == "to_counter" and person.goal and _close(person.agent, person.goal, 75.0):
            person.phase = "billing" if index == 1 else "queue"
            person.face = (-1.0, 0.0, 0.0)
            person.agent.idle(facing=_vec3(person.face))
            _trace(f"{person.role} at counter 1 t={t:.1f}")
            return
        if person.phase == "to_backup" and person.goal and _close(person.agent, person.goal, 75.0):
            person.phase = "backup"
            person.face = (-1.0, 0.0, 0.0)
            person.agent.idle(facing=_vec3(person.face))
            _trace(f"{person.role} at counter 2 t={t:.1f}")
            return
        exit_at = self._exit_at(index)
        checkout = ("queue", "billing", "backup", "to_counter", "to_backup")
        if exit_at is not None and t >= exit_at and person.phase in checkout:
            self._send_exit(person, t)
            return
        if person.phase == "exit":
            pos = _position(person.agent)
            # The body has to be at or past the parking node. A 40 cm early
            # threshold used to delete people at about x=-560.
            if pos[0] <= EXIT_NODE_CM[0] + 5.0 and abs(pos[1] - EXIT_NODE_CM[1]) <= 160.0:
                prim = self.stage.GetPrimAtPath(person.payload_path)
                if prim and prim.IsValid():
                    prim.SetActive(False)
                person.gone = True
                person.phase = "cleared"
                _trace(f"{person.role} cleared past parking ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})")

    def _exit_at(self, index: int) -> float | None:
        if not self.dispatch:
            return 45.0 if index == 1 else None
        return {1: 45.0, 2: 60.0, 3: 70.0, 4: 65.0, 5: 75.0}.get(index)

    def _split_lane(self) -> None:
        if self.split_done:
            return
        self.split_done = True
        staff = self.by_role["Staff_02"]
        if staff.agent is not None and staff.phase == "idle":
            staff.phase = "to_door"
            staff.face = (0.0, 1.0, 0.0)
            _walk(staff, self.marks["staff2_door"], self.mpu, BRISK_MPS, allow_teleport=False)
            _trace(
                "STAFF_02 leave backroom through door -> "
                f"{tuple(round(v, 1) for v in self.marks['staff2_door'])}"
            )
        for index in (4, 5):
            person = self.by_role[f"Shopper_{index:02d}"]
            if person.gone or person.agent is None:
                continue
            if "split" in person.notes:
                continue
            person.notes.append("split")
            person.phase = "to_backup"
            person.face = (-1.0, 0.0, 0.0)
            goal = self.marks["backup"][index - 4]
            _walk(person, goal, self.mpu, WALK_MPS, allow_teleport=False)
            _trace(f"{person.role} step to counter 2")

    def _enter(self, person: Person, t: float) -> None:
        UsdGeom.Imageable(self.stage.GetPrimAtPath(person.payload_path)).MakeVisible()
        person.hidden = False
        if person.shopper_index <= 3:
            start = self.marks["door"][person.shopper_index - 1]
        else:
            start = self.marks["door"][(person.shopper_index - 1) % 3]
        person.agent.teleport(_vec3(start), _vec3((0.0, -1.0, 0.0)))
        self._send_browse(person)
        _trace(f"{person.role} enter t={t:.1f}s mesh={person.folder}")

    def _begin_dwell(self, person: Person, t: float) -> None:
        dwell = _dwell_for(person.shopper_index)
        person.phase = "dwell"
        person.phase_until = t + dwell
        _cancel(person.agent)
        person.agent.idle(facing=_vec3(person.face))
        _trace(f"{person.role} dwell {dwell:.0f}s from t={t:.1f}")

    def _maybe_reach(self, person: Person, t: float) -> None:
        if person.reach_sent or person.shopper_index > 6:
            return
        due = 15.0 if person.shopper_index <= 3 else 25.0
        at_shelf = person.phase in ("dwell", "hold") or (
            person.phase == "to_shelf" and person.goal is not None and _close(person.agent, person.goal, 80.0)
        )
        if t < due or not at_shelf:
            return
        used = _reach_shelf(person, 3.0)
        person.reach_sent = True
        _trace(f"{person.role} reach t={t:.1f} action={used or 'idle'}")

    def _send_browse(self, person: Person) -> None:
        goal = self.marks["browse"][person.shopper_index - 1]
        person.face = _face_for(person.shopper_index)
        person.focus = _focus_for(person.shopper_index, goal)
        person.phase = "to_shelf"
        _walk(person, goal, self.mpu, WALK_MPS, allow_teleport=True)

    def _send_queue(self, person: Person, goal: tuple[float, float, float], phase: str) -> None:
        UsdGeom.Imageable(self.stage.GetPrimAtPath(person.payload_path)).MakeVisible()
        person.hidden = False
        person.phase = phase
        person.face = (-1.0, 0.0, 0.0)
        person.focus = (300.0, goal[1], 115.0)
        _walk(person, goal, self.mpu, BRISK_MPS, allow_teleport=True)
        _trace(f"{person.role} -> {phase}")

    def _send_exit(self, person: Person, t: float) -> None:
        goal = self.marks["exit"]
        person.phase = "exit"
        person.face = (-1.0, 0.0, 0.0)
        _walk(person, goal, self.mpu, BRISK_MPS, allow_teleport=False)
        _trace(f"{person.role} exit t={t:.1f} via parking {tuple(round(v, 1) for v in goal)}")

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
            if person.phase in ("queue", "to_counter", "billing"):
                count += 1
        return count

    def _queue_phases(self) -> str:
        parts = []
        for index in (2, 3, 4, 5):
            person = self.by_role[f"Shopper_{index:02d}"]
            parts.append(f"{index}:{person.phase}")
        return ",".join(parts)


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


def _take_paths(folder: str) -> tuple[list[str], list[str]]:
    cameras = [camera for camera, _filename in SURVEILLANCE_TAKES]
    filenames = [os.path.join(folder, filename) for _camera, filename in SURVEILLANCE_TAKES]
    return cameras, filenames


def _protected_snapshot() -> dict[str, tuple[int, int]]:
    snapshot = {}
    for name in PROTECTED_VIDEOS:
        path = os.path.join(VIDEO_DIR, name)
        if not os.path.isfile(path):
            raise RuntimeError(f"Protected video is missing: {path}")
        stat = os.stat(path)
        snapshot[path] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _assert_protected(snapshot: dict[str, tuple[int, int]]) -> None:
    for path, (size, mtime_ns) in snapshot.items():
        stat = os.stat(path)
        if stat.st_size != size or stat.st_mtime_ns != mtime_ns:
            raise RuntimeError(f"Protected video changed: {path}")


def _reset_replicator() -> None:
    from omni.syntheticdata import SyntheticData

    SyntheticData.Get().reset(usd=True, remove_activated_render_vars=False)
    _trace("SDG_RESET")


def _open_encoders(filenames: list[str]) -> list[tuple[str, subprocess.Popen]]:
    os.makedirs(VIDEO_DIR, exist_ok=True)
    width, height = VIDEO_SIZE
    encoders = []
    for filename in filenames:
        path = os.path.join(VIDEO_DIR, filename)
        if os.path.basename(path) in PROTECTED_VIDEOS and os.path.dirname(path) == VIDEO_DIR:
            raise RuntimeError(f"Refusing to overwrite {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        command = [
            "/usr/bin/ffmpeg",
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
        raise RuntimeError(f"Frame is {len(raw)} bytes, expected {expected}.")
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


def _bbox_counts(payload) -> tuple[dict, dict]:
    """Count boxes whose semantic class is one of the four Track K labels."""
    counts = {name: 0 for name in SEMANTIC_CLASSES}
    if not isinstance(payload, dict):
        return counts, {}
    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    raw = info.get("idToLabels") or info.get("id_to_labels") or payload.get("idToLabels") or {}
    labels: dict[int, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                text = str(value.get("class") or next(iter(value.values()), ""))
            else:
                text = str(value)
            try:
                labels[int(key)] = text
            except (TypeError, ValueError):
                continue
    rows = payload.get("data", payload)
    try:
        semantic_ids = rows["semanticId"]
    except Exception:
        return counts, info if isinstance(info, dict) else {}
    for semantic_id in semantic_ids:
        label = labels.get(int(semantic_id), "")
        if label in counts:
            counts[label] += 1
    return counts, info if isinstance(info, dict) else {}


class AnnotationTape:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, scenario: str, camera: str, t: float, counts: dict) -> None:
        self.rows.append({"scenario": scenario, "camera": camera, "t": round(t, 3), "counts": dict(counts)})

    def save(self, path: str, catalog_counts: dict, milk_paths: list[str], chip_paths: list[str]) -> dict:
        import torch

        class_index = {name: index for index, name in enumerate(SEMANTIC_CLASSES)}
        matrix = []
        for row in self.rows:
            matrix.append([float(row["counts"].get(name, 0)) for name in SEMANTIC_CLASSES])
        if not matrix:
            matrix = [[0.0, 0.0, 0.0, 0.0]]
        counts = torch.tensor(matrix, dtype=torch.float32)
        occupancy = counts.sum(dim=0)
        registry = torch.tensor(
            [float(catalog_counts.get(name, 0)) for name in SEMANTIC_CLASSES],
            dtype=torch.float32,
        )
        state = {
            "annotation_counts": counts,
            "class_occupancy": occupancy,
            "registry_counts": registry,
            "class_index": torch.tensor([class_index[name] for name in SEMANTIC_CLASSES], dtype=torch.int64),
        }
        payload = {
            "format": "marketgen_track_k_annotation_weights_v1",
            "state_dict": state,
            "classes": list(SEMANTIC_CLASSES),
            "class_to_index": class_index,
            "milk_prims": list(milk_paths),
            "chip_prims": list(chip_paths),
            "samples": self.rows,
            "note": (
                "Replicator bounding-box occupancy compiled from the Track K takes. "
                "Not a trained detector and not an accuracy claim."
            ),
        }
        torch.save(payload, path)
        size = os.path.getsize(path)
        if size < 200:
            raise RuntimeError(f"Tracking weights file is {size} bytes.")
        return {"path": path, "bytes": size, "samples": len(self.rows)}


async def record_pass(
    director: StoreDirector,
    cameras: list[str],
    filenames: list[str],
    tape: AnnotationTape,
) -> list[dict]:
    if len(cameras) != len(filenames):
        raise RuntimeError("Camera and video lists differ.")
    if len(cameras) != 5:
        raise RuntimeError(f"Each take must record all 5 cameras, got {len(cameras)}.")
    _reset_replicator()
    _restore_take(director.stage, director.people, director.milk_paths + director.chip_paths)
    _configure_timeline()
    width, height = VIDEO_SIZE
    products = [rep.create.render_product(camera, (width, height)) for camera in cameras]
    rgb_ann = []
    box_ann = []
    for product in products:
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
        rgb_ann.append(annotator)
        try:
            boxes = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
            boxes.attach([product])
        except Exception as exc:
            _trace(f"BBOX_ATTACH_FAIL {exc}")
            boxes = None
        box_ann.append(boxes)
    await _wait_frames(3)
    await _agents_online(director.people)
    place_people(director.stage, director.people, director.marks)
    director.prime()
    for _ in range(8):
        await rep.orchestrator.step_async(pause_timeline=False)
    await _agents_online(director.people)
    place_people(director.stage, director.people, director.marks)
    director.prime()
    frames = int(RECORD_SECONDS * FPS)
    encoders = _open_encoders(filenames)
    try:
        written = 0
        empty_run = 0
        _trace(f"RECORD_ARMED scenario={director.scenario} frames={frames} cameras={cameras}")
        while written < frames:
            t = written / float(FPS)
            director.update(t)
            await rep.orchestrator.step_async(pause_timeline=False)
            grabbed = []
            frames_ok = True
            for annotator in rgb_ann:
                data = annotator.get_data()
                if data is None or len(data) == 0:
                    frames_ok = False
                    break
                grabbed.append(data)
            if not frames_ok:
                empty_run += 1
                if empty_run > 20:
                    raise RuntimeError(f"Camera produced an empty frame at t={t:.2f} scenario={director.scenario}")
                continue
            empty_run = 0
            for data, (_path, process) in zip(grabbed, encoders):
                _write_rgb(process, data)
            if written % FPS == 0:
                for camera, boxes in zip(cameras, box_ann):
                    if boxes is None:
                        continue
                    try:
                        box_data = boxes.get_data()
                    except Exception as exc:
                        _trace(f"BBOX_READ_FAIL {camera} {exc}")
                        continue
                    counts, info = _bbox_counts(box_data if isinstance(box_data, dict) else {"data": box_data})
                    if written == 0:
                        keys = sorted(info.keys()) if isinstance(info, dict) else []
                        _trace(f"BBOX_INFO {camera} keys={keys} counts={counts}")
                    tape.add(director.scenario, camera, t, counts)
            if written % (FPS * 5) == 0:
                spots = []
                for person in director.people:
                    if person.gone or person.agent is None:
                        continue
                    pos = _position(person.agent)
                    spots.append(f"{person.role}:{person.phase}({pos[0]:.0f},{pos[1]:.0f})")
                _trace(f"RECORD {director.scenario} t={t:.1f}s frame={written} {' '.join(spots)}")
            written += 1
    finally:
        videos = _close_encoders(encoders)
        for annotator in rgb_ann + [item for item in box_ann if item is not None]:
            try:
                annotator.detach()
            except Exception:
                pass
        for product in products:
            try:
                product.destroy()
            except Exception:
                pass
    return videos


async def prepare() -> dict:
    ira_version = _ira_runtime()
    stage, stage_source = _hook_open_stage()
    import omni.timeline

    omni.timeline.get_timeline_interface().stop()
    for path in (PRIMARY_REGISTER, BACKUP_REGISTER):
        if not stage.GetPrimAtPath(path):
            raise RuntimeError(f"Missing register {path}")
    import torch

    _trace(f"STAGE {stage_source} IRA {ira_version} torch {torch.__version__}")
    cameras = apply_surveillance_optics(stage)
    people = bind_cast(stage)
    _set_cast_active(stage, people, False)
    await _wait_frames(2)
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    door_prims = _exclude_service_door(stage)
    floor = author_backroom_floor(stage)
    volume = extend_walkable_volume(stage, meters_per_unit)
    volume["service_door"] = door_prims
    volume["backroom_floor"] = floor
    _trace(f"BACKROOM_FLOOR tiles={floor['tiles']} layer={floor['layer']} door_meshes={len(door_prims)}")
    radius = set_tight_agent()
    baked = await bake_navmesh()
    walk = verify_navmesh(baked["navmesh"])
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
            raise RuntimeError(f"{person.role} height is {person.height_cm} cm.")
        _trace(
            f"ACTOR {person.role} mesh={person.folder} height={person.height_cm}cm entry={person.entry_s}"
        )
    catalog = MetadataCatalog(METADATA_PATH)
    milk = _require_prims(stage, catalog.facing(FACING_DAIRY))
    chips = _require_prims(stage, catalog.facing(FACING_SNACKS))
    if len(milk) != 8 or len(chips) != 10:
        raise RuntimeError(f"Facing removal is {len(milk)} dairy and {len(chips)} snacks.")
    _trace("DAIRY_VIEW CCTV_Entrance_Aisle " + " ".join(path.rsplit("/", 1)[-1] for path in milk))
    _trace("SNACK_VIEW CCTV_Back_Aisle_Left " + " ".join(path.rsplit("/", 1)[-1] for path in chips))
    semantics = apply_semantics(stage, catalog, people)
    marks = build_marks()
    _trace(
        f"NAVMESH radius_cm={radius['min_cm']:.1f}/{radius['max_cm']:.1f} "
        f"slope_deg={radius['slope_deg']:.2f} baked={baked['baked']} "
        f"door_path_cm={walk['backroom_to_counter2_cm']} exit_cm={walk['door_to_exit_cm']} "
        f"backroom={walk['probes']['backroom']} door={walk['probes']['door_threshold']} "
        f"exit={walk['probes']['exit_node']}"
    )
    _trace(
        f"MARKS staff2={tuple(round(v, 1) for v in marks['staff2'])} "
        f"door={tuple(round(v, 1) for v in marks['staff2_door'])} "
        f"register2={tuple(round(v, 1) for v in marks['staff2_register'])} "
        f"exit={tuple(round(v, 1) for v in marks['exit'])}"
    )
    _trace(
        f"CAMERA_ENTRANCE focal={cameras['entrance']['focal']} hap={cameras['entrance']['hap']} "
        f"dep={cameras['entrance']['depression']:.2f} eye={[round(v, 1) for v in cameras['entrance']['eye']]}"
    )
    _trace(
        f"CAMERA_CHECKOUT focal={cameras['checkout']['focal']} hap={cameras['checkout']['hap']} "
        f"dep={cameras['checkout']['depression']:.2f} eye={[round(v, 1) for v in cameras['checkout']['eye']]}"
    )
    _trace(
        f"CAMERA_BACK focal={cameras['back']['focal']} hap={cameras['back']['hap']} "
        f"dep={cameras['back']['depression']:.2f} eye={[round(v, 1) for v in cameras['back']['eye']]} "
        f"up={[round(v, 3) for v in cameras['back']['up']]}"
    )
    _trace(f"SEMANTICS {semantics}")
    _trace(
        "NAVMESH_NOTE minimum agent radius is 0.20 m. Maximum stays "
        f"{AGENT_MAX_RADIUS_CM:.0f} cm because these bodies are about 23-26 cm."
    )
    return {
        "stage": stage,
        "people": people,
        "marks": marks,
        "meters_per_unit": meters_per_unit,
        "milk": milk,
        "chips": chips,
        "semantics": semantics,
        "walk": walk,
        "radius": radius,
        "volume": volume,
        "cameras": cameras,
    }


def _copy_take(videos: list[dict], folder: str) -> list[dict]:
    copied = []
    for video in videos:
        dest_dir = os.path.join(VIDEO_DIR, folder)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(video["path"]))
        shutil.copy2(video["path"], dest)
        size = os.path.getsize(dest)
        if size != video["bytes"]:
            raise RuntimeError(f"Copied {dest} is {size} bytes, source is {video['bytes']}.")
        copied.append({"path": dest, "bytes": size})
    return copied


async def main() -> dict:
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    protected = _protected_snapshot()
    try:
        os.remove(PROGRESS_PATH)
    except OSError:
        pass
    _trace("TRACK_K_MASTER start")
    _trace(
        "CLIP_FILES retail_cashier_typing_loop.usd clip_shelf_audit.usd clip_reach_shelf.usd "
        "are not in this build. Cashiers use pose_hand. Shoppers use reach_hand."
    )
    prepared = await prepare()
    if not globals().get("TRACK_K_MASTER_RECORD", True):
        _trace("TRACK_K_MASTER_SETUP_OK")
        return {"setup": True, "milk": prepared["milk"], "chips": prepared["chips"], "walk": prepared["walk"]}
    tape = AnnotationTape()
    stage = prepared["stage"]
    people = prepared["people"]
    baseline = StoreDirector(
        stage,
        people,
        prepared["marks"],
        prepared["meters_per_unit"],
        prepared["milk"],
        prepared["chips"],
        "baseline",
    )
    baseline_cameras, baseline_files = _take_paths(SINGLE_CASHIER_DIR)
    baseline_videos = await record_pass(baseline, baseline_cameras, baseline_files, tape)
    dispatch = StoreDirector(
        stage,
        people,
        prepared["marks"],
        prepared["meters_per_unit"],
        prepared["milk"],
        prepared["chips"],
        "dispatch",
    )
    dispatch_cameras, dispatch_files = _take_paths(DYNAMIC_STAFF_DIR)
    dispatch_videos = await record_pass(dispatch, dispatch_cameras, dispatch_files, tape)
    shelf_videos = _copy_take(dispatch_videos, SHELF_DELETION_DIR)
    _trace(
        "SHELF_COPY dynamic_staff and shelf_deletion are the same five angles. "
        "Dairy leaves the entrance view at t=25. Chips leave the left back-aisle view at t=40."
    )
    weights = tape.save(
        WEIGHTS_PATH,
        prepared["semantics"]["counts"],
        prepared["milk"],
        prepared["chips"],
    )
    frozen = {path: _camera_pose(stage, path) for path in FROZEN_CAMERAS}
    for path, before in prepared["cameras"]["frozen"].items():
        after = frozen[path]
        _assert_pose_held(before, after, path, True, True)
    videos = baseline_videos + dispatch_videos + shelf_videos
    if len(videos) != 15:
        raise RuntimeError(f"Expected 15 videos, wrote {len(videos)}.")
    _assert_protected(protected)
    _trace(f"WEIGHTS {weights}")
    for video in videos:
        _trace(f"VIDEO {video['path']} bytes={video['bytes']}")
    _trace("TRACK_K_MASTER_OK")
    return {"videos": videos, "weights": weights}


def _bootstrap() -> asyncio.Task | None:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        return loop.create_task(main())
    loop.run_until_complete(main())
    return None


if globals().get("TRACK_K_MASTER_AUTO", True):
    _bootstrap()
