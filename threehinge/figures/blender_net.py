"""Blender/Cycles render of the hinge network - studio look for panel f.

Run:  blender -b -P blender_net.py -- <tag>     (tag = lat | dor)
Scene from export_for_blender.py; writes results/_net_<tag>3.png (RGBA)."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import bpy, json, sys, math, os
from mathutils import Matrix, Vector

TR = (_env.ROOT + "/surf/three_hinge/tracto/blender_scene")
OUT = (_env.ROOT + "/surf/three_hinge/results")
argv = sys.argv[sys.argv.index("--") + 1:]
TAG = argv[0]
DARK = "dark" in argv[1:]
FAST = "fast" in argv[1:]
CLAY = "clay" in argv[1:] or TAG == "clay"

C_TUBE = (0.085, 0.165, 0.390, 1)        # mid slate blue: dark enough to read on the clay,
                                         # light enough that a crossing region is not a black mass
C_HINGE = (0.800, 0.150, 0.050, 1)       # orange #DE7A5A
C_CROWN = (0.42, 0.51, 0.63, 1)          # pale blue-gray
C_SHELL = (0.72, 0.73, 0.78, 1)
C_SHELL_DARK = (0.30, 0.36, 0.50, 1)
C_CLAY = (0.520, 0.512, 0.500, 1)        # matte cortex; a hair warm, to sit off the blue fibres

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.device = "GPU"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "OPTIX"
prefs.get_devices()
for d in prefs.devices:
    d.use = True
sc.cycles.samples = 96 if FAST else 384
sc.cycles.use_denoising = True
sc.cycles.transparent_max_bounces = 64
sc.render.film_transparent = not DARK
if CLAY:
    # Standard, not Filmic: the clay wall is a narrow band of pale greys sitting on a white page,
    # and a film curve either lifts it into the paper or crushes the folding shading out of it.
    sc.view_settings.view_transform = "Standard"
else:
    sc.view_settings.view_transform = "Filmic"
    sc.view_settings.look = "Medium High Contrast"


def load(name):
    bpy.ops.wm.ply_import(filepath=f"{TR}/{name}.ply")
    ob = bpy.context.selected_objects[0]
    ob.name = name
    for p in ob.data.polygons:
        p.use_smooth = True
    return ob


def shell_mat(name, alpha, rough):
    m = bpy.data.materials.new(name); m.use_nodes = True
    col = C_SHELL_DARK if DARK else C_SHELL
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    tra = nt.nodes.new("ShaderNodeBsdfTransparent")
    pr = nt.nodes.new("ShaderNodeBsdfPrincipled")
    pr.inputs["Base Color"].default_value = col
    pr.inputs["Roughness"].default_value = rough
    mix.inputs["Fac"].default_value = alpha          # 0 = fully transparent
    nt.links.new(tra.outputs[0], mix.inputs[1])
    nt.links.new(pr.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs[0])
    return m


def clay_mat():
    """Fully opaque matte cortex - the shell pass.

    Every transparent version of a folded pial fails the same way: a lateral ray crosses the
    surface a dozen times and the alphas compound into cotton wool. Keying the mix on Backfacing
    does not help either, because on a folded surface the near-side sulcal walls are themselves
    back-facing and close the silhouette again. So the shell is rendered opaque, as its own pass,
    and lightened toward the page afterwards (compose_clay.py): the folding then reads at full
    relief with no accumulation anywhere, and the network is composited over it."""
    m = bpy.data.materials.new("clay"); m.use_nodes = True
    pr = m.node_tree.nodes["Principled BSDF"]
    pr.inputs["Base Color"].default_value = C_CLAY
    pr.inputs["Roughness"].default_value = 0.72
    try:
        pr.inputs["Specular IOR Level"].default_value = 0.18
    except KeyError:
        pass
    return m


def dec_mat(ob, rough=0.38, glow=float(os.environ.get("DEC_GLOW", 0.14))):
    """Base Colour driven by the mesh's own vertex colours - the DEC tube material.

    A slice of self-emission is deliberate. Under pure reflected light a tube deep in the
    hemisphere is lit far less than one on the near surface, so hue and brightness both vary and
    the reader cannot tell a dim green from a dark blue. Emission flattens that out: the colour
    then encodes direction only, and depth is carried by the mist fade instead."""
    m = bpy.data.materials.new("tubes_dec"); m.use_nodes = True
    pr = m.node_tree.nodes["Principled BSDF"]
    pr.inputs["Roughness"].default_value = rough
    try:
        pr.inputs["Specular IOR Level"].default_value = 0.20   # highlights would wash the hue out
    except KeyError:
        pass
    ca = ob.data.color_attributes
    if not len(ca):
        raise RuntimeError("tubes.ply carries no colour attribute - re-run export_for_blender.py")
    n = m.node_tree.nodes.new("ShaderNodeVertexColor")
    n.layer_name = ca[0].name
    m.node_tree.links.new(n.outputs["Color"], pr.inputs["Base Color"])
    m.node_tree.links.new(n.outputs["Color"], pr.inputs["Emission Color"])
    pr.inputs["Emission Strength"].default_value = glow
    return m


def solid_mat(name, color, rough, spec=0.5, glow=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    pr = m.node_tree.nodes["Principled BSDF"]
    pr.inputs["Base Color"].default_value = color
    pr.inputs["Roughness"].default_value = rough
    try:
        pr.inputs["Specular IOR Level"].default_value = spec
    except KeyError:
        pass
    if glow:                       # a whisper of emission lifts the colour through the shells
        pr.inputs["Emission Color"].default_value = color
        pr.inputs["Emission Strength"].default_value = glow
    return m


if CLAY:
    # backdrop only: the two passes are composited flat, so this surface never occludes the
    # network - it supplies the silhouette and the folding relief, nothing else. 'white' is the
    # surface the 3HG patches and the streamlines actually live on; 'pial' gives the outer
    # brain shape. CLAY_SHELL picks which.
    pial = load("white_s" if os.environ.get("CLAY_SHELL", "white") == "white" else "pial")
    pial.name = "pial"
    pial.data.materials.append(clay_mat())
else:
    pial = load("pial")
    pial.data.materials.append(shell_mat("pial", 0.055 if DARK else 0.060, 0.20))
    white = load("white")
    white.data.materials.append(shell_mat("white", 0.070 if DARK else 0.080, 0.35))
tubes = load("tubes")
if CLAY:
    tubes.data.materials.append(dec_mat(tubes))
else:
    tubes.data.materials.append(solid_mat("tubes", (0.10, 0.24, 0.55, 1) if DARK else C_TUBE, 0.32, glow=0.55 if DARK else 0.06))
hinge = load("hinge"); hinge.data.materials.append(solid_mat("hinge", C_HINGE, 0.22, 0.7, glow=2.2 if DARK else 0.35))
if not CLAY:
    # no crown-crown streamlines are drawn, so in the clay panel the crown nodes are dots that
    # explain nothing - they only speckle the cortex
    crown = load("crown")
    crown.data.materials.append(solid_mat("crown", C_CROWN, 0.45, 0.3, glow=0.25 if DARK else 0.0))

CAM_KEY = "clay_w" if (CLAY and os.environ.get("CLAY_SHELL", "white") == "white") else TAG
cams = json.load(open(f"{TR}/cams.json"))[CAM_KEY]
view = Vector(cams["view"]); up = Vector(cams["up"]); right = Vector(cams["right"])
ctr = Vector(cams["center"])
cam = bpy.data.cameras.new("cam")
if cams.get("persp"):
    cam.type = "PERSP"; cam.lens = 80; cam.sensor_width = 36
    dist = cams["dist"]
else:
    cam.type = "ORTHO"; cam.ortho_scale = cams["ortho_scale"]
    dist = 500
cam.sensor_fit = "HORIZONTAL"
cam.clip_start = 1.0; cam.clip_end = 3000
co = bpy.data.objects.new("cam", cam)
R = Matrix(((right.x, up.x, view.x), (right.y, up.y, view.y), (right.z, up.z, view.z)))
co.matrix_world = Matrix.Translation(ctr + view * dist) @ R.to_4x4()
sc.collection.objects.link(co); sc.camera = co

# studio: big soft key above-left-front, cool fill right, thin rim behind
def area(name, direction, energy, size, color=(1, 1, 1)):
    lt = bpy.data.lights.new(name, "AREA")
    lt.energy = energy; lt.size = size; lt.color = color
    ob = bpy.data.objects.new(name, lt)
    d = Vector(direction).normalized()
    ob.location = ctr + d * 320
    q = d.to_track_quat("Z", "Y")        # light points along -Z toward the scene
    ob.rotation_euler = q.to_euler()
    sc.collection.objects.link(ob)

key_dir = (view * 0.55 + up * 0.75 + right * -0.45)
fil_dir = (view * 0.55 + up * -0.25 + right * 0.75)
rim_dir = (-view * 0.75 + up * 0.55)
if CLAY:
    # the interior wall is lit only through the open near side, so the key sits close to the
    # camera axis; the rim is dropped, it would just glare off the wall's rear pole
    area("key", view * 0.80 + up * 0.50 + right * -0.35, 0.95e6, 320)
    area("fill", view * 0.70 + up * -0.35 + right * 0.70, 0.30e6, 380, color=(0.95, 0.97, 1.0))
else:
    area("key", key_dir, (2.2e6 if DARK else 5.5e6), 260)
    area("fill", fil_dir, (0.9e6 if DARK else 2.2e6), 300, color=(0.92, 0.96, 1.0))
    area("rim", rim_dir, (2.4e6 if DARK else 2.8e6), 220)
w = bpy.data.worlds.new("w"); sc.world = w
w.use_nodes = True
if CLAY:
    # a dim ambient only: too much fill and the sulci stop reading as depth
    w.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.22
elif DARK:
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.010, 0.016, 0.032, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 1.0
else:
    w.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.35
if DARK:
    # soft glow around the emissive nodes/tubes: fog-glow glare in the compositor
    sc.use_nodes = True
    nt = sc.node_tree; nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    gl = nt.nodes.new("CompositorNodeGlare")
    gl.glare_type = "FOG_GLOW"; gl.threshold = 0.55; gl.size = 8; gl.mix = -0.35
    cmp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Image"], gl.inputs["Image"])
    nt.links.new(gl.outputs["Image"], cmp.inputs["Image"])

rx, ry = cams["res"]
if FAST:
    rx, ry = rx // 2, ry // 2
sc.render.resolution_x, sc.render.resolution_y = rx, ry
sc.render.image_settings.color_mode = "RGBA"

if CLAY:
    # depth cue for the network pass: fade toward the page with distance from the camera, so a
    # streamline on the far bank of a sulcus recedes instead of competing with a near one.
    hd = cams.get("halfdepth", 60.0)
    ms = w.mist_settings
    ms.use_mist = True; ms.falloff = "LINEAR"
    ms.start = cams["dist"] - hd * 1.02; ms.depth = hd * 2.04
    bpy.context.view_layer.use_pass_mist = True
    sc.use_nodes = True
    nt = sc.node_tree; nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    fac = nt.nodes.new("CompositorNodeMath"); fac.operation = "MULTIPLY"
    MIST = float(os.environ.get("DEC_MIST", 0.18))     # how far the deepest fibre washes out
    fac.inputs[1].default_value = MIST
    mixc = nt.nodes.new("CompositorNodeMixRGB")
    mixc.inputs[2].default_value = (1, 1, 1, 1)
    sa = nt.nodes.new("CompositorNodeSetAlpha")        # the mix would otherwise flatten alpha
    cmp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Mist"], fac.inputs[0])
    nt.links.new(fac.outputs[0], mixc.inputs[0])
    nt.links.new(rl.outputs["Image"], mixc.inputs[1])
    nt.links.new(mixc.outputs[0], sa.inputs["Image"])
    nt.links.new(rl.outputs["Alpha"], sa.inputs["Alpha"])
    nt.links.new(sa.outputs[0], cmp.inputs["Image"])
    for pas, shown in (("shell", {"pial"}), ("net", {"tubes", "hinge"} if CLAY else {"tubes", "hinge", "crown"})):
        for ob in bpy.data.objects:
            if ob.type == "MESH":
                ob.hide_render = ob.name not in shown
        fac.inputs[1].default_value = 0.0 if pas == "shell" else MIST
        sc.render.filepath = f"{OUT}/_net_clay_{pas}.png"
        bpy.ops.render.render(write_still=True)
    print("done clay (shell + net)")
else:
    sc.render.filepath = f"{OUT}/_net_{TAG}_dark.png" if DARK else f"{OUT}/_net_{TAG}3.png"
    bpy.ops.render.render(write_still=True)
    print("done", TAG)
