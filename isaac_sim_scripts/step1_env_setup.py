"""Step 1 — light, CCTV, and navmesh setup for the open convenience store.

Runs inside a live Isaac Sim 6.0.1 session (Script Editor, or the
``isaacsim.code_editor.python_server`` on 127.0.0.1:8226). The open stage is
edited in place. This script never opens, reloads, or resets the stage.

The MarketGen export is a centimeter, Z-up shop
(``ConvenienceStore01.usd``). Camera focal length is a true 18 mm lens on the
USD academy gate, and the navmesh agent radius of 0.3 m is written as 30 cm
because navigation settings are stored in centimeters.
"""

from __future__ import annotations

import asyncio
import math
import os
from typing import Iterable

import carb.settings
import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

OUTPUT_DIR = "isaac_sim_scripts"
STAGE_TOKEN = "ConvenienceStore01.usd"
WORLD_ROOT = "/World"
STORE_ROOT = "/Root"
FRAMEWORK = "/Root/SupermarketFramework"
CASHIER = "/Root/BP_CashierArea_01"

CHECKOUT_CAMERA = "/World/CCTV_Checkout_Shelves"
ENTRANCE_CAMERA = "/World/CCTV_Entrance_Aisle"
NAVMESH_INCLUDE = "/World/NavMesh_StoreFloor"
NAVMESH_EXCLUDE = "/World/NavMesh_NotchExclude"

FOCAL_LENGTH_MM = 18.0
# Academy gate, in millimeters. 18 mm on this gate is a 60.4° x 46.0° wide lens.
HORIZONTAL_APERTURE_MM = 20.955
VERTICAL_APERTURE_MM = 15.2908
F_STOP = 11.0

CHECKOUT_DEPRESSION_DEG = 38.0
# Header mount. 38° from this door looks into the mat and drops the aisle;
# 36° keeps shoppers, the entry lane, and the primary aisle inside the 18 mm gate.
ENTRANCE_DEPRESSION_DEG = 36.0

# Navigation bake settings are stored in centimeters (see omni.anim.navigation.core).
AGENT_RADIUS_METERS = 0.3
AGENT_RADIUS_CM = AGENT_RADIUS_METERS * 100.0
# Export20 leaves the authored sun at full strength and lights the aisles with
# the ceiling panels. The 35% cut left the back of this shop dark.
LIGHT_KEEP = 1.0
ORIGINAL_INTENSITY = "marketgen:originalIntensity"

NAVMESH_MIN_RADIUS = "/exts/omni.anim.navigation.core/navMesh/config/agentMinRadius"
NAVMESH_MAX_RADIUS = "/exts/omni.anim.navigation.core/navMesh/config/agentMaxRadius"
NAVMESH_MIN_RADIUS_PERSISTENT = "/persistent" + NAVMESH_MIN_RADIUS
NAVMESH_MAX_RADIUS_PERSISTENT = "/persistent" + NAVMESH_MAX_RADIUS

_PURPOSES = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]


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


def _range(stage: Usd.Stage, prim: Usd.Prim, purposes: list | None = None) -> Gf.Range3d:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes or _PURPOSES, True)
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if bounds.IsEmpty():
        raise RuntimeError(f"No world bounds for {prim.GetPath()}")
    return bounds


def _vec(value: Gf.Vec3d) -> Gf.Vec3d:
    return Gf.Vec3d(value[0], value[1], value[2])


def _mm_to_tenths(millimeters: float, meters_per_unit: float) -> float:
    """USD camera lengths are tenths of a scene unit, which is 1 mm when the stage is in cm."""
    tenth_mm = meters_per_unit * 100.0
    if tenth_mm <= 0.0:
        raise RuntimeError(f"Invalid metersPerUnit {meters_per_unit}")
    return millimeters / tenth_mm


def _meters_to_stage(meters: float, meters_per_unit: float) -> float:
    return meters / meters_per_unit


def ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def _light_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    """Global lights: dome lights, and the store's directional sun when no dome is present."""
    found: list[Usd.Prim] = []
    for prim in stage.Traverse():
        if not prim.IsActive():
            continue
        type_name = prim.GetTypeName()
        if type_name in ("DomeLight", "DistantLight") or prim.IsA(UsdLux.DomeLight) or prim.IsA(UsdLux.DistantLight):
            found.append(prim)
    return found


def _intensity(prim: Usd.Prim) -> tuple[Usd.Attribute, float]:
    for name in ("inputs:intensity", "intensity"):
        attr = prim.GetAttribute(name)
        if attr and attr.IsValid() and attr.HasValue():
            value = attr.Get()
            if value is not None:
                return attr, float(value)
    raise RuntimeError(f"{prim.GetPath()} has no intensity")


def soften_global_lights(stage: Usd.Stage) -> list[dict]:
    """Keep the authored sun intensity. Re-running does not stack a second cut.

    The export has no DomeLight. ``/Root/Lighting/DirectionalLight`` stays at
    its authored white 6500 K value. The aisle brightness comes from the
    ceiling rect lights, matched to export20.
    """
    lights = _light_prims(stage)
    if not lights:
        raise RuntimeError("No DomeLight or DistantLight prims were found on the open stage.")
    report = []
    for prim in lights:
        attr, current = _intensity(prim)
        if prim.HasCustomDataKey(ORIGINAL_INTENSITY):
            original = float(prim.GetCustomDataByKey(ORIGINAL_INTENSITY))
        else:
            original = current
            prim.SetCustomDataByKey(ORIGINAL_INTENSITY, original)
        softened = original * LIGHT_KEEP
        attr.Set(softened)
        mirror = prim.GetAttribute("intensity" if attr.GetName() == "inputs:intensity" else "inputs:intensity")
        if mirror and mirror.IsValid() and mirror.HasValue():
            mirror.Set(softened)
        report.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "original": original,
                "intensity": softened,
            }
        )
    return report


def _children_named(parent: Usd.Prim, prefix: str) -> list[Usd.Prim]:
    return [child for child in parent.GetChildren() if child.GetName().startswith(prefix)]


def _union(ranges: Iterable[Gf.Range3d]) -> Gf.Range3d:
    merged = Gf.Range3d()
    any_range = False
    for bounds in ranges:
        if bounds.IsEmpty():
            continue
        if not any_range:
            merged = bounds
            any_range = True
        else:
            merged.UnionWith(bounds)
    if not any_range:
        raise RuntimeError("No bounds to union.")
    return merged


def floor_tiles(stage: Usd.Stage) -> list[Gf.Range3d]:
    framework = stage.GetPrimAtPath(FRAMEWORK)
    if not framework:
        raise RuntimeError(f"Missing store shell at {FRAMEWORK}")
    tiles = []
    for child in _children_named(framework, "FloorComponent"):
        # Default purpose is the 1 m tile. Proxy bounds on these components
        # span the whole shell and would punch a hole through the sales floor.
        bounds = _range(stage, child, purposes=[UsdGeom.Tokens.default_])
        tiles.append(bounds)
    if not tiles:
        raise RuntimeError("The store shell has no FloorComponent tiles.")
    return tiles


def ceiling_z(stage: Usd.Stage) -> float:
    framework = stage.GetPrimAtPath(FRAMEWORK)
    walls = [
        child
        for child in framework.GetChildren()
        if "Wall" in child.GetName()
    ]
    if not walls:
        raise RuntimeError("No wall components to measure the ceiling.")
    return max(_range(stage, wall).GetMax()[2] for wall in walls)


def front_door(stage: Usd.Stage) -> Gf.Range3d:
    framework = stage.GetPrimAtPath(FRAMEWORK)
    doors = _children_named(framework, "DoorComponent")
    if not doors:
        raise RuntimeError("No DoorComponent prims on the store shell.")
    # The customer entry is the door on the front wall (largest Y). The side
    # leaf in the service notch sits much further into the building.
    return max((_range(stage, door) for door in doors), key=lambda bounds: bounds.GetMax()[1])


def _horizontal_runs(stage: Usd.Stage) -> list[Gf.Range3d]:
    root = stage.GetPrimAtPath(STORE_ROOT)
    runs = []
    for child in root.GetChildren():
        if not child.GetName().startswith("ShelfBox"):
            continue
        bounds = _range(stage, child)
        size = bounds.GetSize()
        if size[0] > size[1]:
            runs.append(bounds)
    # Front of the shop is +Y. Walk the runs from the door toward the back wall.
    runs.sort(key=lambda bounds: (bounds.GetMin()[1] + bounds.GetMax()[1]) * 0.5, reverse=True)
    return runs


def primary_aisle(stage: Usd.Stage) -> Gf.Vec3d:
    """First real gondola-to-gondola gap inside the front door.

    Shelf runs are parallel to the front wall, so the shopping aisles run along X.
    The gap that opens onto the front plaza is circulation, not the shopping aisle.
    """
    runs = _horizontal_runs(stage)
    if len(runs) < 2:
        raise RuntimeError("Need at least two shelf runs to find the shopping aisle.")
    best = None
    examined = []
    for before, after in zip(runs, runs[1:]):
        # ``before`` is the run closer to the front (larger Y). A positive gap is
        # the clear width between the two faces.
        width = before.GetMin()[1] - after.GetMax()[1]
        examined.append(round(width, 1))
        if width < 120.0:
            continue
        center_y = (before.GetMin()[1] + after.GetMax()[1]) * 0.5
        center_x = (max(before.GetMin()[0], after.GetMin()[0]) + min(before.GetMax()[0], after.GetMax()[0])) * 0.5
        # Skip a gap that starts at the front wall. That is the entry plaza.
        if before.GetMax()[1] > -80.0:
            continue
        candidate = (center_x, center_y, width)
        if best is None:
            best = candidate
    if best is None:
        raise RuntimeError(
            f"No shopping aisle wider than 1.2 m. Gaps examined from {len(runs)} shelf runs: {examined}."
        )
    return Gf.Vec3d(best[0], best[1], best[2])


def _normalize(vector: Gf.Vec3d) -> Gf.Vec3d:
    length = vector.GetLength()
    if length < 1e-6:
        raise RuntimeError("Cannot normalize a zero vector.")
    return vector / length


def _depression_of(forward: Gf.Vec3d) -> float:
    horizontal = math.hypot(forward[0], forward[1])
    return math.degrees(math.atan2(-forward[2], horizontal))


def _camera_forward(prim: Usd.Prim) -> Gf.Vec3d:
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return _normalize(matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)))


def _look_matrix(eye: Gf.Vec3d, target: Gf.Vec3d) -> Gf.Matrix4d:
    """Camera-to-world matrix. USD cameras look down local -Z, with +Y up."""
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


def _point_on_depression(eye: Gf.Vec3d, aim_xy: Gf.Vec2d, target_z: float, depression_deg: float) -> Gf.Vec3d:
    direction = _normalize(Gf.Vec3d(aim_xy[0] - eye[0], aim_xy[1] - eye[1], 0.0))
    drop = eye[2] - target_z
    if drop <= 0.0:
        raise RuntimeError("Camera is not above its subject.")
    horizontal = drop / math.tan(math.radians(depression_deg))
    return Gf.Vec3d(eye[0] + direction[0] * horizontal, eye[1] + direction[1] * horizontal, target_z)


def place_camera(
    stage: Usd.Stage,
    path: str,
    eye: Gf.Vec3d,
    target: Gf.Vec3d,
    meters_per_unit: float,
    role: str,
) -> dict:
    if not stage.GetPrimAtPath(WORLD_ROOT):
        UsdGeom.Xform.Define(stage, WORLD_ROOT)
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    camera.CreateFocalLengthAttr(_mm_to_tenths(FOCAL_LENGTH_MM, meters_per_unit))
    camera.CreateHorizontalApertureAttr(_mm_to_tenths(HORIZONTAL_APERTURE_MM, meters_per_unit))
    camera.CreateVerticalApertureAttr(_mm_to_tenths(VERTICAL_APERTURE_MM, meters_per_unit))
    camera.CreateFStopAttr(F_STOP)
    distance = (target - eye).GetLength()
    camera.CreateFocusDistanceAttr(distance)
    # Near clip sits in front of the housing. Far clip covers the whole shop.
    near = _meters_to_stage(0.05, meters_per_unit)
    far = _meters_to_stage(200.0, meters_per_unit)
    camera.CreateClippingRangeAttr(Gf.Vec2f(near, far))

    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(_look_matrix(eye, target))
    forward = _camera_forward(camera.GetPrim())
    aimed = _normalize(target - eye)
    if Gf.Dot(forward, aimed) < 0.98:
        raise RuntimeError(f"{path} is not looking at its target. Forward {forward}, aim {aimed}.")
    camera.GetPrim().SetCustomDataByKey("marketgen:role", role)
    return {
        "path": path,
        "eye_cm": [round(eye[0], 1), round(eye[1], 1), round(eye[2], 1)],
        "target_cm": [round(target[0], 1), round(target[1], 1), round(target[2], 1)],
        "depression_deg": round(_depression_of(forward), 2),
        "focal_length_mm": FOCAL_LENGTH_MM,
        "f_stop": F_STOP,
        "focus_distance_cm": round(distance, 1),
    }


def build_cameras(stage: Usd.Stage, meters_per_unit: float) -> list[dict]:
    tiles = floor_tiles(stage)
    floor = _union(tiles)
    door = front_door(stage)
    cashier = _range(stage, stage.GetPrimAtPath(CASHIER))
    aisle = primary_aisle(stage)
    gondolas = [
        bounds
        for bounds in _horizontal_runs(stage)
        if bounds.GetMax()[1] < -80.0 and bounds.GetMin()[0] > 200.0
    ]
    if not gondolas:
        raise RuntimeError("No interior gondola runs were found.")
    gondola_min_x = min(bounds.GetMin()[0] for bounds in gondolas)
    shelf_z = max(bounds.GetMax()[2] for bounds in gondolas)

    # Checkout: ceiling mount just inboard of the left wall, above the
    # approach to the conveyor. 38° down crosses the register and lands in
    # the customer lane, with the gondolas in the upper half of the 18 mm frame.
    ceiling = ceiling_z(stage)
    checkout_eye = Gf.Vec3d(floor.GetMin()[0] + 140.0, cashier.GetMax()[1] - 65.0, ceiling - 12.0)
    checkout_aim = Gf.Vec2d(gondola_min_x + 50.0, aisle[1])
    checkout_target = _point_on_depression(checkout_eye, checkout_aim, 105.0, CHECKOUT_DEPRESSION_DEG)

    # Entrance: on the header of the front door, just inside the threshold,
    # looking down the walkway that meets the primary aisle.
    door_center_x = (door.GetMin()[0] + door.GetMax()[0]) * 0.5
    spine_x = (cashier.GetMax()[0] + gondola_min_x) * 0.5
    entrance_eye = Gf.Vec3d(door_center_x - 30.0, door.GetMin()[1] - 3.4, door.GetMax()[2] + 12.0)
    entrance_target = _point_on_depression(
        entrance_eye,
        Gf.Vec2d(spine_x, aisle[1]),
        120.0,
        ENTRANCE_DEPRESSION_DEG,
    )

    checkout = place_camera(
        stage,
        CHECKOUT_CAMERA,
        checkout_eye,
        checkout_target,
        meters_per_unit,
        "Wide checkout camera: conveyor in the lower frame, gondola labels behind it.",
    )
    entrance = place_camera(
        stage,
        ENTRANCE_CAMERA,
        entrance_eye,
        entrance_target,
        meters_per_unit,
        "Header camera above the front door, looking down the walkway into the primary aisle.",
    )
    if abs(checkout["depression_deg"] - CHECKOUT_DEPRESSION_DEG) > 0.6:
        raise RuntimeError(f"Checkout camera depression is {checkout['depression_deg']}, expected 38.")
    if abs(entrance["depression_deg"] - ENTRANCE_DEPRESSION_DEG) > 0.6:
        raise RuntimeError(f"Entrance camera depression is {entrance['depression_deg']}, expected 36.")
    checkout["ceiling_cm"] = round(ceiling, 1)
    checkout["shelf_top_cm"] = round(shelf_z, 1)
    entrance["primary_aisle_cm"] = [round(aisle[0], 1), round(aisle[1], 1), round(aisle[2], 1)]
    return [checkout, entrance]


def _missing_floor_rects(tiles: list[Gf.Range3d]) -> list[Gf.Range3d]:
    """Holes in the 1 m floor grid. Each hole is kept separate.

    The service notch is the only real hole. Taking one bounding box around
    every missing cell would also cover the cashier lane if a stray gap
    appeared anywhere else on the grid.
    """
    present = set()
    for bounds in tiles:
        ix = int(round(bounds.GetMin()[0] / 100.0))
        iy = int(round(bounds.GetMin()[1] / 100.0))
        present.add((ix, iy))
    if not present:
        return []
    min_ix = min(index[0] for index in present)
    max_ix = max(index[0] for index in present)
    min_iy = min(index[1] for index in present)
    max_iy = max(index[1] for index in present)
    missing = {
        (ix, iy)
        for ix in range(min_ix, max_ix + 1)
        for iy in range(min_iy, max_iy + 1)
        if (ix, iy) not in present
    }
    seen: set[tuple[int, int]] = set()
    rects = []
    for seed in sorted(missing):
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        cells = []
        while stack:
            cell = stack.pop()
            cells.append(cell)
            cx, cy = cell
            for neighbor in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if neighbor in missing and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        # A single missed tile is a measurement glitch, not an exterior notch.
        if len(cells) < 4:
            continue
        min_x = min(index[0] for index in cells) * 100.0
        max_x = (max(index[0] for index in cells) + 1) * 100.0
        min_y = min(index[1] for index in cells) * 100.0
        max_y = (max(index[1] for index in cells) + 1) * 100.0
        rects.append(Gf.Range3d(Gf.Vec3d(min_x, min_y, 0.0), Gf.Vec3d(max_x, max_y, 0.0)))
    return rects


def _fit_volume(prim: Usd.Prim, minimum: Gf.Vec3d, maximum: Gf.Vec3d, meters_per_unit: float) -> None:
    """Fit a NavMeshVolume whose local extent is a 1 m cube expressed in stage units."""
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


def _define_volume(stage: Usd.Stage, path: str, volume_type: str) -> Usd.Prim:
    import NavSchema

    if not stage.GetPrimAtPath(WORLD_ROOT):
        UsdGeom.Xform.Define(stage, WORLD_ROOT)
    existing = stage.GetPrimAtPath(path)
    if existing and existing.IsValid():
        prim = existing
    else:
        prim = NavSchema.NavMeshVolume.Define(stage, path).GetPrim()
    volume = NavSchema.NavMeshVolume(prim)
    volume_attr = volume.GetNavVolumeTypeAttr()
    if not volume_attr:
        volume_attr = volume.CreateNavVolumeTypeAttr()
    volume_attr.Set(volume_type)
    if not prim.HasAPI(NavSchema.NavMeshAreaAPI):
        prim.ApplyAPI(NavSchema.NavMeshAreaAPI)
    return prim


def _enable_navigation() -> None:
    """NavSchema is registered by the navigation extension, so load it before any volume."""
    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled("omni.anim.navigation.core"):
        manager.set_extension_enabled_immediate("omni.anim.navigation.core", True)


def define_walkable_volume(stage: Usd.Stage, meters_per_unit: float) -> dict:
    _enable_navigation()
    tiles = floor_tiles(stage)
    floor = _union(tiles)
    z0 = floor.GetMin()[2]
    # Cover the slab and a standing person. Stop short of using the ceiling as a floor.
    z_min = z0 - 20.0
    z_max = z0 + 220.0
    pad = 5.0
    include_min = Gf.Vec3d(floor.GetMin()[0] - pad, floor.GetMin()[1] - pad, z_min)
    include_max = Gf.Vec3d(floor.GetMax()[0] + pad, floor.GetMax()[1] + pad, z_max)
    include = _define_volume(stage, NAVMESH_INCLUDE, "Include")
    _fit_volume(include, include_min, include_max, meters_per_unit)

    exclude_report = []
    for index, notch in enumerate(_missing_floor_rects(tiles)):
        # The decorative floor plane continues through this untiled notch. Exclude it
        # so characters stay inside the shop.
        path = NAVMESH_EXCLUDE if index == 0 else f"{NAVMESH_EXCLUDE}_{index}"
        exclude = _define_volume(stage, path, "Exclude")
        exclude_min = Gf.Vec3d(notch.GetMin()[0] - 8.0, notch.GetMin()[1] - 8.0, z_min)
        exclude_max = Gf.Vec3d(notch.GetMax()[0] + 8.0, notch.GetMax()[1] + 8.0, z_max)
        _fit_volume(exclude, exclude_min, exclude_max, meters_per_unit)
        exclude_report.append([exclude_min[0], exclude_min[1], exclude_max[0], exclude_max[1]])

    sky = stage.GetPrimAtPath("/Root/Lighting/SM_SkySphere")
    if sky and sky.IsValid():
        import NavSchema

        if not sky.HasAPI(NavSchema.NavMeshExcludeAPI):
            sky.ApplyAPI(NavSchema.NavMeshExcludeAPI)

    return {
        "include": NAVMESH_INCLUDE,
        "floor_cm": [floor.GetMin()[0], floor.GetMin()[1], floor.GetMax()[0], floor.GetMax()[1]],
        "height_cm": [z_min, z_max],
        "notch_exclude_cm": exclude_report or None,
    }


def set_agent_radius() -> dict:
    """Lock the baked agent radius to 0.3 m (30 cm in the navigation settings)."""
    settings = carb.settings.get_settings()
    # Max is lowered first so it never sits below the minimum mid-update.
    for path in (NAVMESH_MAX_RADIUS, NAVMESH_MAX_RADIUS_PERSISTENT):
        settings.set(path, AGENT_RADIUS_CM)
    for path in (NAVMESH_MIN_RADIUS, NAVMESH_MIN_RADIUS_PERSISTENT):
        settings.set(path, AGENT_RADIUS_CM)
    return {
        "agent_radius_m": AGENT_RADIUS_METERS,
        "stored_cm": AGENT_RADIUS_CM,
        "min_cm": float(settings.get(NAVMESH_MIN_RADIUS)),
        "max_cm": float(settings.get(NAVMESH_MAX_RADIUS)),
    }


async def bake_navmesh() -> dict:
    import omni.anim.navigation.core as nav

    _enable_navigation()
    # 0 disables the wait timeout. A 15k-mesh shop can outlast the default wait
    # and still publish a mesh, which would otherwise look like a failed bake.
    carb.settings.get_settings().set("/exts/omni.anim.navigation.core/navMesh/bakeWaitTimeoutSeconds", 0.0)
    app = omni.kit.app.get_app()
    await app.next_update_async()
    await app.next_update_async()
    interface = nav.acquire_interface()
    baked = bool(interface.start_navmesh_baking_and_wait())
    navmesh = interface.get_navmesh()
    if navmesh is None:
        raise RuntimeError("NavMesh bake did not produce a mesh. Check the include volume and the floor prims.")
    return {
        "baked": baked,
        "agent_min_radius_cm": float(navmesh.get_agent_min_radius()),
        "agent_max_radius_cm": float(navmesh.get_agent_max_radius()),
    }


def _probe(navmesh, point: tuple[float, float, float]) -> dict | None:
    result = navmesh.query_closest_point(target=point)
    if not result:
        return None
    hit = result[0]
    if hit is None:
        return None
    delta = Gf.Vec3d(hit[0] - point[0], hit[1] - point[1], hit[2] - point[2])
    return {
        "hit": [round(hit[0], 1), round(hit[1], 1), round(hit[2], 1)],
        "distance_cm": round(delta.GetLength(), 1),
    }


def verify_navmesh() -> dict:
    import omni.anim.navigation.core as nav

    navmesh = nav.acquire_interface().get_navmesh()
    probes = {
        "primary_aisle": _probe(navmesh, (700.0, -457.0, 30.0)),
        "entry_lane": _probe(navmesh, (420.0, -180.0, 30.0)),
        "front_left_floor": _probe(navmesh, (80.0, -200.0, 30.0)),
        "service_notch": _probe(navmesh, (150.0, -1000.0, 30.0)),
        "inside_gondola": _probe(navmesh, (720.0, -607.0, 40.0)),
    }
    aisle = probes["primary_aisle"]
    entry = probes["entry_lane"]
    front_left = probes["front_left_floor"]
    notch = probes["service_notch"]
    if aisle is None or aisle["distance_cm"] > 80.0:
        raise RuntimeError(f"Primary aisle is not walkable: {aisle}")
    if entry is None or entry["distance_cm"] > 80.0:
        raise RuntimeError(f"Entry lane is not walkable: {entry}")
    if front_left is None or front_left["distance_cm"] > 80.0:
        raise RuntimeError(f"Front-left sales floor is not walkable: {front_left}")
    if notch is not None and notch["distance_cm"] < 50.0:
        raise RuntimeError(f"Service notch was baked as walkable: {notch}")
    start = navmesh.query_closest_point(target=(420.0, -180.0, 20.0))
    end = navmesh.query_closest_point(target=(780.0, -457.0, 20.0))
    path = None
    if start and end and start[0] is not None and end[0] is not None:
        path = navmesh.query_shortest_path(start_pos=start[0], end_pos=end[0])
    path_cm = None
    if path is not None:
        length = path.length() if hasattr(path, "length") else None
        path_cm = None if length is None else round(float(length), 1)
    if path_cm is None:
        raise RuntimeError("No navmesh path from the entry lane to the primary aisle.")
    return {"probes": probes, "entry_to_aisle_cm": path_cm}


async def main() -> dict:
    ensure_output_dir()
    stage = _stage()
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = UsdGeom.GetStageUpAxis(stage)
    if up_axis != UsdGeom.Tokens.z:
        raise RuntimeError(f"Expected a Z-up store, found up-axis {up_axis}.")

    lights = soften_global_lights(stage)
    cameras = build_cameras(stage, meters_per_unit)
    volume = define_walkable_volume(stage, meters_per_unit)
    radius = set_agent_radius()
    baked = await bake_navmesh()
    walk = verify_navmesh()
    report = {
        "stage": stage.GetRootLayer().identifier,
        "app": f"{omni.kit.app.get_app().get_app_name()} {omni.kit.app.get_app().get_app_version()}",
        "up_axis": str(up_axis),
        "meters_per_unit": meters_per_unit,
        "output_dir": OUTPUT_DIR,
        "lights": lights,
        "cameras": cameras,
        "navmesh_volume": volume,
        "agent_radius": radius,
        "bake": baked,
        "walk": walk,
    }
    print("STEP1_OK")
    for light in lights:
        print(
            f"LIGHT {light['path']} {light['type']} "
            f"{light['original']:.2f} -> {light['intensity']:.2f}"
        )
    for camera in cameras:
        print(
            f"CAMERA {camera['path']} eye={camera['eye_cm']} "
            f"depression={camera['depression_deg']} focal={camera['focal_length_mm']}mm"
        )
    print(
        f"NAVMESH radius={baked['agent_min_radius_cm']}..{baked['agent_max_radius_cm']} cm "
        f"path={walk['entry_to_aisle_cm']} cm"
    )
    return report


def _bootstrap() -> asyncio.Task | None:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        return loop.create_task(main())
    loop.run_until_complete(main())
    return None


if globals().get("MARKETGEN_STEP1_AUTO", True):
    _bootstrap()
