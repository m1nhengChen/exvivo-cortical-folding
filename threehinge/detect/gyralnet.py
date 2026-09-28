#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025
All Rights Reserved for  Medical Imaging and Neuroscientific Discovery Lab, Department of Computer Science & Engineering, University of Texas at Arlington
Author: Lu Zhang, Chao Cao, Minheng Chen, Tong Chen
Email for contact: lz50@iu.edu; dajiang.zhu@uta.edu
Date: 2026-02-12
License: MIT
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env


import vtk
import nibabel.freesurfer.io as io
from collections import defaultdict,deque
import numpy as np
import os
import shutil
import time
import networkx as nx
import argparse
import signal
import heapq
import sys
import atexit
from multiprocessing import active_children
from concurrent.futures import ProcessPoolExecutor, as_completed
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import math
from collections import Counter
import logging
import sys
import os
from datetime import datetime
import itertools

def corridor_path_skeleton(Gskel, a, b):
    # On the skeleton, use hop-based shortest path
    # (consistent with the hop definition in find_close_3hg_pairs)
    return nx.shortest_path(Gskel, a, b)

def pick_mid_on_path(path):
    return path[len(path)//2]

def cut_edge_if_exists(G, u, v):
    if G.has_edge(u, v):
        G.remove_edge(u, v)

def cut_corridor_edges(Gskel, corridor_nodes):
    for u, v in zip(corridor_nodes[:-1], corridor_nodes[1:]):
        cut_edge_if_exists(Gskel, u, v)

def cut_non_corridor_edges(Gskel, hub, corridor_neighbor, num_cut_edges=2):
    """
    At the hub node (a or b), remove edges that are NOT the corridor edge.
    Remove up to `num_cut_edges` such edges.

    Returns the neighbors whose edges were cut (i.e., the branch start nodes).
    """
    others = [n for n in Gskel.neighbors(hub) if n != corridor_neighbor]
    # Optional: sort here to ensure deterministic behavior (e.g., by node ID)
    others = sorted(others)
    cut = others[:num_cut_edges]
    for n in cut:
        cut_edge_if_exists(Gskel, hub, n)
    return cut

def walk_k_hops_to_endpoint(Gskel, start, prev, k=15):
    """
    Traverse along the skeleton for up to k hops (stop if a leaf node or junction is encountered).
    Used to push the branch start further toward a more distal endpoint.
    """
    cur, pre = start, prev
    steps = 0
    while steps < k:
        nxts = [x for x in Gskel.neighbors(cur) if x != pre]
        if len(nxts) == 0:
            break 
        if len(nxts) > 1:
            break  
        pre, cur = cur, nxts[0]
        steps += 1
    return cur

def shortest_path_weighted(Gsurf, s, t):
    # Your Gsurf edges have weight='weight'
    return nx.shortest_path(Gsurf, source=s, target=t, weight='weight')

def vertex_disjoint_paths_to_mid_weighted(Gsurf, endpoints, mid, forbid_shared_endpoints=False):
    """
    Greedy strategy: for each endpoint in sequence, compute the weighted shortest path to the mid node.
    After a path is found, remove its internal vertices to prevent reuse in subsequent paths.
    If forbid_shared_endpoints=True, the constraint is stricter: even endpoints cannot be shared
    (typically not necessary).
    """
    Gtmp = Gsurf.copy()
    used = set([mid])
    paths = []

    for ep in endpoints:
        if ep == mid:
            paths.append([mid])
            continue

        p = shortest_path_weighted(Gtmp, ep, mid)

        if forbid_shared_endpoints:
            internal = set(p[:-1]) - {mid} 
        else:
            internal = set(p[1:-1])          

        # Enforce that internal vertices cannot be shared (except for the mid node)
        Gtmp.remove_nodes_from(internal)
        used |= internal
        paths.append(p)

    return paths

def add_path_edges_to_skeleton(Gskel, path):
    # Add the surface route back as new edges in the skeleton graph
    for u, v in zip(path[:-1], path[1:]):
        if u != v:
            Gskel.add_edge(u, v)

def _k_hop_nodes(G, src, k, banned=None):
    """Return nodes exactly k hops away from src (excluding banned)."""
    if banned is None:
        banned = set()
    dmap = nx.single_source_shortest_path_length(G, src, cutoff=k)
    return [n for n, d in dmap.items() if d == k and n not in banned]

def _pick_mid_avoid_hinge(G, corridor):
    """Prefer degree==2 mid inside corridor, fallback to hop-mid."""
    inner = corridor[1:-1]
    cand = [x for x in inner if G.degree(x) == 2]
    if cand:
        return cand[len(cand)//2]
    return corridor[len(corridor)//2]

def _edge_list_from_path(path):
    return list(zip(path[:-1], path[1:]))
def _pick_mid_simple(a, b, corridor):

    # d==1: mid = a (per your requirement)
    if len(corridor) == 2:
        return a
    # otherwise pick middle vertex on corridor (pure index middle)
    return corridor[len(corridor)//2]

def _choose_branch_seeds(Gskel, hinge, *, k=2, banned=None):
    """
    Seeds are 1-hop neighbors of hinge in CURRENT skeleton (after corridor cut).
    Deterministic: sorted by node id.
    """
    banned = banned or set()
    seeds = [x for x in Gskel.neighbors(hinge) if x not in banned]
    seeds = sorted(seeds)
    return seeds[:k]

def cut_branch_by_hops(Gskel, hinge, seed, cut_hops, *, print_debug=False, tag=""):
    """
    True hop-cut: starting from edge (hinge, seed), walk away from hinge and cut up to cut_hops edges.
    Endpoint (stub) is the detached-side node of the LAST successfully cut edge.
    NOTE: This mutates Gskel in-place.
    Returns: (cut_edges, stub_endpoint)
    """
    cut_edges = []
    prev = hinge
    cur = seed
    stub = None

    for step in range(cut_hops):
        # must exist to cut
        if not Gskel.has_edge(prev, cur):
            if print_debug:
                print(f"[CUT-HOPS]{tag} step={step} edge-missing ({prev},{cur}) stop")
            break

        # cut this edge
        Gskel.remove_edge(prev, cur)
        cut_edges.append((prev, cur))
        stub = cur  # detached side at this step

        if print_debug:
            print(f"[CUT-HOPS]{tag} step={step+1}/{cut_hops} cut ({prev},{cur})")

        # move forward away from hinge: choose any neighbor of cur except prev
        nbrs = [x for x in Gskel.neighbors(cur) if x != prev]
        if not nbrs:
            # branch ended naturally
            if print_debug:
                print(f"[CUT-HOPS]{tag} step={step+1} reached dead-end at {cur}")
            break

        # deterministic: smallest id
        nxt = sorted(nbrs)[0]
        prev, cur = cur, nxt

    return cut_edges, stub
def cut_branch_front_n_edges(G, hinge, n_edges, *, print_debug=False, tag=""):
    """
    For each neighbor branch of hinge, walk away from hinge and cut up to n_edges
    along that branch (simple path preference).
    """
    removed = []
    nbrs = list(G.neighbors(hinge))

    for nb in nbrs:
        prev = hinge
        cur = nb
        steps = 0

        while steps < n_edges and G.has_edge(prev, cur):
            # cut this edge
            G.remove_edge(prev, cur)
            removed.append((prev, cur))
            steps += 1

            # continue forward along the branch
            nxts = [x for x in G.neighbors(cur) if x != prev]
            if not nxts:
                break
            nxts.sort()  # deterministic
            prev, cur = cur, nxts[0]

    if print_debug:
        print(f"[CUT-BRANCH-N]{tag} hinge={hinge} n_edges={n_edges} removed={len(removed)}")
        if len(removed) <= 30:
            print(f"  edges={removed}")
    return removed


def cut_edge_record(G, u, v, removed_edges, *, print_debug=False, tag=""):
    """Cut (u,v) if exists; record it; return True if cut happened."""
    if G.has_edge(u, v):
        G.remove_edge(u, v)
        removed_edges.append((u, v))
        if print_debug:
            print(f"[CUT-EDGE]{tag} ({u},{v})")
        return True
    return False

def stubs_from_removed_edges(removed_edges, a, b):
    """
    Strict stub definition:
      - A-side stubs are nodes x where edge (a,x) got removed
      - B-side stubs are nodes x where edge (b,x) got removed
    Return (stubsA, stubsB) as sorted unique lists.
    """
    stubsA, stubsB = set(), set()
    for u, v in removed_edges:
        if u == a and v != b:
            stubsA.add(v)
        elif v == a and u != b:
            stubsA.add(u)
        if u == b and v != a:
            stubsB.add(v)
        elif v == b and u != a:
            stubsB.add(u)
    return sorted(stubsA), sorted(stubsB)

def refine_close_3hg_pair_disjoint(
    Gskel, Gsurf, a, b,
    *,
    branches_per_side=2,     # each side pick 2 branches => 4 endpoints total
    cut_hops=4,              # TRUE "cut 4 hops" along each selected branch
    cut_full_corridor=True,  # remove all edges on corridor a..b
    enforce_disjoint_routes=True,
    print_debug=False,
):
    """
    Rewire close 3HG pair (a,b):
      1) corridor on skeleton (hop shortest): a..b
      2) choose mid:
          - if d==1 => mid=a
          - else => middle vertex of corridor
      3) cut skeleton:
          - cut corridor edges (optional)
          - pick branches_per_side seeds from a-side and b-side (hinge neighbors AFTER corridor cut)
          - for each seed, cut cut_hops edges along that branch (true hop cut)
          - endpoints = resulting stubs (one per branch)
      4) route each endpoint on surface graph to mid
      5) enforce vertex-disjoint routes except mid
      6) add routed edges back into skeleton
    """

    # ---- sanity ----
    if a not in Gskel or b not in Gskel:
        return False
    if not nx.has_path(Gskel, a, b):
        return False
    if a not in Gsurf or b not in Gsurf:
        # surface graph must cover these nodes
        return False

    # ---- 1) corridor + mid ----
    corridor = nx.shortest_path(Gskel, a, b)
    d = len(corridor) - 1
    mid = _pick_mid_simple(a, b, corridor)

    if print_debug:
        print(f"\n[PAIR] a={a} b={b} skel_hops={d} corridor_len={len(corridor)} mid={mid}")
        if len(corridor) <= 30:
            print(f"[CORRIDOR] {corridor}")

    # ---- 2) cut corridor on skeleton ----
    removed_edges = []

    if cut_full_corridor:
        for u, v in _edge_list_from_path(corridor):
            if Gskel.has_edge(u, v):
                Gskel.remove_edge(u, v)
                removed_edges.append((u, v))
                if print_debug:
                    print(f"[CUT-CORRIDOR] cut ({u},{v})")

    # ---- 3) choose seeds AFTER corridor cut ----
    # banned seeds: mid (optional) to avoid instant route degeneracy
    banned_seed = {mid}

    seedsA = _choose_branch_seeds(Gskel, a, k=branches_per_side, banned=banned_seed)
    seedsB = _choose_branch_seeds(Gskel, b, k=branches_per_side, banned=banned_seed)

    if print_debug:
        print(f"[SEEDS] branches_per_side={branches_per_side} cut_hops={cut_hops}")
        print(f"  seedsA(from a={a}) = {seedsA}")
        print(f"  seedsB(from b={b}) = {seedsB}")

    if len(seedsA) < branches_per_side or len(seedsB) < branches_per_side:
        if print_debug:
            print("[FAIL] Not enough branch seeds after corridor cut.")
        return False

    # ---- 4) hop-cut along each seed branch; endpoints = stubs ----
    endpoints = []
    cut_detail = []

    for i, s in enumerate(seedsA):
        ce, stub = cut_branch_by_hops(Gskel, a, s, cut_hops, print_debug=print_debug, tag=f" A{i}")
        removed_edges += ce
        cut_detail.append(("A", a, s, ce, stub))
        if stub is None:
            if print_debug:
                print(f"[FAIL] A-side seed {s} produced no stub.")
            return False
        endpoints.append(stub)

    for i, s in enumerate(seedsB):
        ce, stub = cut_branch_by_hops(Gskel, b, s, cut_hops, print_debug=print_debug, tag=f" B{i}")
        removed_edges += ce
        cut_detail.append(("B", b, s, ce, stub))
        if stub is None:
            if print_debug:
                print(f"[FAIL] B-side seed {s} produced no stub.")
            return False
        endpoints.append(stub)

    # endpoints must be 4
    if len(endpoints) != 2 * branches_per_side:
        if print_debug:
            print("[FAIL] endpoints count mismatch.")
        return False

    # uniqueness check (endpoints shouldn't collapse)
    if len(set(endpoints)) != len(endpoints):
        if print_debug:
            print(f"[FAIL] endpoints not unique: {endpoints}")
        return False

    if print_debug:
        print(f"[CUT] corridor_cut={cut_full_corridor} total_removed_edges={len(removed_edges)}")
        for side, hinge, seed, ce, stub in cut_detail:
            print(f"[CUT-SUM]{side} hinge={hinge} seed={seed} cut_edges={len(ce)} stub={stub}")
        print(f"[ENDPOINTS] {endpoints}")
        print(f"[MID] mid={mid}")

    # ---- 5) route each endpoint on surface graph to mid ----
    routes = []
    for idx, ep in enumerate(endpoints):
        if ep not in Gsurf or mid not in Gsurf:
            if print_debug:
                print(f"[FAIL] ep or mid missing in Gsurf: ep={ep} mid={mid}")
            return False
        try:
            path = nx.shortest_path(Gsurf, ep, mid, weight="weight")
        except nx.NetworkXNoPath:
            if print_debug:
                print(f"[FAIL] No surface path from ep={ep} to mid={mid}")
            return False
        routes.append(path)

        if print_debug:
            print(f"[ROUTE-{idx}] ep={ep} -> mid={mid} lenV={len(path)}")
            if len(path) <= 40:
                print(f"  path={path}")
            else:
                print(f"  head={path[:12]} ... tail={path[-12:]}")

    # ---- 6) disjoint constraint ----
    if enforce_disjoint_routes:
        used = {mid}
        for i, p in enumerate(routes):
            # forbid any overlap excluding mid
            internal = set(p[:-1])  # includes endpoint; excludes mid (last)
            inter = internal & used
            if inter:
                if print_debug:
                    sample = sorted(list(inter))[:30]
                    print(f"[FAIL][DISJOINT] route-{i} overlaps (except mid). examples={sample}")
                return False
            used |= internal

        if print_debug:
            print("[DISJOINT] PASS: no shared vertices except mid.")

    # ---- 7) add routed edges back to skeleton ----
    for p in routes:
        for u, v in _edge_list_from_path(p):
            Gskel.add_edge(u, v)

    if print_debug:
        print(f"[ADD] added routed edges. now |V|={Gskel.number_of_nodes()} |E|={Gskel.number_of_edges()}")
        print(f"[SUMMARY] mid={mid}, endpoints={endpoints}")

    return True

def route_4_disjoint_paths_to_mid(H, endpoints, mid, *, weight='weight', forbid_nodes=None):
    """
    Return dict {s: path} where each path is node-disjoint except at mid.
    - H: surface routing graph (already removed skeleton nodes/edges as needed)
    - endpoints: list of 4 endpoints
    - mid: sink
    - forbid_nodes: optional set of nodes forbidden for ALL routes (corridor nodes etc.)
    """
    endpoints = [int(x) for x in endpoints]
    mid = int(mid)

    base_forbid = set(forbid_nodes) if forbid_nodes else set()
    base_forbid.discard(mid)
    for s in endpoints:
        base_forbid.discard(s)

    # precompute distances for ordering heuristics
    dist = {}
    for s in endpoints:
        try:
            dist[s] = nx.shortest_path_length(H, s, mid, weight=weight)
        except nx.NetworkXNoPath:
            dist[s] = float('inf')

    # try permutations: harder first (longer dist first) tends to help
    orders = sorted(itertools.permutations(endpoints), key=lambda ord_: -sum(dist[s] for s in ord_))

    for order in orders:
        used = set()  # used internal vertices
        paths = {}
        ok = True

        for s in order:
            H2 = H.copy()
            # remove forbidden + used (but keep endpoints and mid)
            ban = base_forbid | (used - {mid})
            H2.remove_nodes_from(ban)

            try:
                p = nx.shortest_path(H2, s, mid, weight=weight)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                ok = False
                break

            # enforce disjoint: internal vertices cannot be reused
            internal = set(p[1:-1])
            if internal & used:
                ok = False
                break

            paths[s] = p
            used |= internal

        if ok and len(paths) == len(endpoints):
            # final sanity: pairwise disjoint except mid
            all_used = []
            for s,p in paths.items():
                all_used.append(set(p) - {mid})
            inter = set.intersection(*all_used) if all_used else set()
            if inter:
                # endpoints themselves might overlap if duplicate endpoints; treat as fail
                continue
            return paths

    return None

class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

def setup_logging(log_dir, name="run"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # --- 关键：避免重复 handler（强烈建议 clear）---
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    formatter = logging.Formatter(
        ""
    )

    # file handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # console handler：必须写到原始 stdout
    import io
    ch = logging.StreamHandler(sys.__stdout__)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # redirect stdout / stderr：StreamToLogger 内部应当调用 logger.log(...)
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    return log_file

def _edges_to_adj_list_of_sets(connected_lines_list, point_num=None):
    """
    Build list-of-sets adjacency from an edge list [[u, v], ...].
    If point_num is None, infer from max index.
    """
    if point_num is None:
        point_num = (0 if not connected_lines_list else
                     max(max(u, v) for u, v in connected_lines_list) + 1)
    adj = [set() for _ in range(point_num)]
    for u, v in connected_lines_list:
        u = int(u); v = int(v)
        if u == v: 
            continue
        adj[u].add(v); adj[v].add(u)
    return adj

def _adj_list_of_sets_to_edges(adj):
    """Create de-duplicated undirected edge list from list-of-sets adjacency."""
    edges = []
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            if u < v:
                edges.append([int(u), int(v)])
    return edges

def _to_int_array(mapping_or_array, length=None, fill=-1, dtype=np.int32):
    if isinstance(mapping_or_array, (list, tuple, np.ndarray)):
        arr = np.asarray(mapping_or_array, dtype=dtype)
        return arr
    # dict path
    max_key = max(mapping_or_array.keys(), default=-1)
    size = max((length or 0), max_key + 1)
    arr = np.full(size, fill, dtype=dtype)
    for k, v in mapping_or_array.items():
        arr[int(k)] = int(v)
    return arr

def _to_float_array(mapping_or_array, length=None, fill=0.0, dtype=np.float32):
    if isinstance(mapping_or_array, (list, tuple, np.ndarray)):
        arr = np.asarray(mapping_or_array, dtype=dtype)
        return arr
    # dict path
    max_key = max(mapping_or_array.keys(), default=-1)
    size = max((length or 0), max_key + 1)
    arr = np.full(size, fill, dtype=dtype)
    for k, v in mapping_or_array.items():
        arr[int(k)] = float(v)
    return arr

def _max_index_in_neighbors(mapping) -> int:
    """Return max index found among keys and all neighbor ids; -1 if empty."""
    max_idx = -1
    for k, v in mapping.items():
        ki = int(k)
        if ki > max_idx:
            max_idx = ki
        if v:
            mv = int(max(v))
            if mv > max_idx:
                max_idx = mv
    return max_idx

def _to_neighbors_list(point_connect_points_dict, length=None):
    """
    Normalize neighbors to list[np.ndarray]. Auto-sizes to cover all keys/values.
    """
    import numpy as np

    # If already a list-like, return array-ified copy (optionally pad)
    if isinstance(point_connect_points_dict, list):
        arrs = [np.asarray(x, dtype=np.intp) for x in point_connect_points_dict]
        if length is not None and length > len(arrs):
            # pad with empty arrays
            arrs.extend([np.empty(0, dtype=np.intp) for _ in range(length - len(arrs))])
        return arrs

    # Dict path
    max_needed = _max_index_in_neighbors(point_connect_points_dict)
    size = max((length or 0), max_needed + 1)
    out = [np.empty(0, dtype=np.intp) for _ in range(size)]
    for k, v in point_connect_points_dict.items():
        out[int(k)] = np.asarray(v, dtype=np.intp)
    return out



def read_vtk_file(vtk_file):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    Header = reader.GetHeader()

    polydata = reader.GetOutput()

    nCells = polydata.GetNumberOfCells()
    nPolys = polydata.GetNumberOfPolys()
    nLines = polydata.GetNumberOfLines()
    nStrips = polydata.GetNumberOfStrips()
    nPieces = polydata.GetNumberOfPieces()
    nVerts = polydata.GetNumberOfVerts()
    nPoints = polydata.GetNumberOfPoints()
    Points = polydata.GetPoints()
    Point = polydata.GetPoints().GetPoint(0)

    return polydata


def get_point_data_array(polydata, array_name: str) -> np.ndarray:
    arr = polydata.GetPointData().GetArray(array_name)
    if arr is None:
        raise ValueError(f"Point data array '{array_name}' not found on surface")

    values = vtk_to_numpy(arr)

    # ensure 1D
    if values.ndim > 1 and values.shape[1] == 1:
        values = values.ravel()

    # avoid copy if already float64
    if values.dtype == np.float64:
        return values

    # otherwise cast, no extra copy if NumPy can reuse memory
    return values.astype(np.float64, copy=False)

EXECUTOR = None


def _terminate_children_and_exit():
    try:
        if EXECUTOR is not None:
            try:
                EXECUTOR.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
    except Exception:
        pass
    try:
        for p in active_children():
            try:
                p.terminate()
            except Exception:
                pass
    except Exception:
        pass
    try:
        _kill_all_child_processes()
    except Exception:
        pass
    sys.exit(1)


def _signal_handler(signum, frame):
    print('Received signal ' + str(signum) + ', terminating...', flush=True)
    _terminate_children_and_exit()


def _list_child_pids_linux(parent_pid):
    child_pids = []
    try:
        for pid in os.listdir('/proc'):
            if pid.isdigit():
                status_path = '/proc/' + pid + '/status'
                try:
                    with open(status_path, 'r') as f:
                        for line in f:
                            if line.startswith('PPid:'):
                                ppid = int(line.split()[1])
                                if ppid == parent_pid:
                                    pid_int = int(pid)
                                    child_pids.append(pid_int)
                                    child_pids.extend(_list_child_pids_linux(pid_int))
                                break
                except Exception:
                    continue
    except Exception:
        pass
    # de-duplicate
    return list(set(child_pids))


def _kill_all_child_processes():
    if os.name != 'posix':
        return
    # Best-effort: SIGTERM then SIGKILL
    child_pids = _list_child_pids_linux(os.getpid())
    for pid in child_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    try:
        time.sleep(0.5)
    except Exception:
        pass
    for pid in child_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass

atexit.register(lambda: _kill_all_child_processes())

def get_connect_points(surf_polydata):
    """
    Build undirected vertex adjacency from a triangle mesh.
    Returns defaultdict(list) mapping point_id -> list of unique neighbors.
    """
    n_pts = surf_polydata.GetNumberOfPoints()
    cell_num = surf_polydata.GetNumberOfCells()
    ca = surf_polydata.GetPolys()

    # Try legacy layout: [3, a, b, c] * cell_num
    tri = None
    data = ca.GetData()
    if data is not None:
        arr = vtk_to_numpy(data)
        if arr.size == cell_num * 4:                      # legacy triangle cell array
            tri = arr.reshape(cell_num, 4)[:, 1:4].astype(np.intp, copy=False)

    # Fallback for VTK9 separated storage (offsets + connectivity)
    if tri is None:
        conn = vtk_to_numpy(ca.GetConnectivityArray())
        offs = vtk_to_numpy(ca.GetOffsetsArray())
        counts = np.diff(np.r_[offs, conn.size])

        # If not purely triangles, handle polygons generically
        if not np.all(counts == 3):
            adj_sets = [set() for _ in range(n_pts)]
            start = 0
            for cnt in counts:
                face = conn[start:start+cnt]
                start += cnt
                m = len(face)
                for i in range(m):
                    u = int(face[i]); v = int(face[(i+1) % m])
                    if u != v:
                        adj_sets[u].add(v); adj_sets[v].add(u)
            out = defaultdict(list)
            for i, s in enumerate(adj_sets):
                if s:
                    out[i] = list(s)
            return out

        tri = conn.reshape(cell_num, 3).astype(np.intp, copy=False)

    # tri: (N,3) with columns a,b,c
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]

    # Build each undirected edge once: (a,b), (a,c), (b,c)
    u = np.concatenate([a, a, b]).astype(np.intp, copy=False)
    v = np.concatenate([b, c, c]).astype(np.intp, copy=False)

    # Fill adjacency sets; sets dedupe shared edges across triangles
    adj_sets = [set() for _ in range(n_pts)]
    for ui, vi in zip(u, v):
        ui = int(ui); vi = int(vi)
        if ui != vi:
            adj_sets[ui].add(vi)
            adj_sets[vi].add(ui)

    # Convert to requested type
    out = defaultdict(list)
    for i, s in enumerate(adj_sets):
        if s:   # only store non-empty entries (matches your behavior)
            out[i] = list(s)
    return out


def get_connect_points_gyri_part(surf_polydata, sulc_data):
    """
    Build adjacency only between vertices whose sulc < 0.
    Returns: defaultdict(list) mapping point -> list of neighbors (unique, order unspecified).
    """
    n_pts = surf_polydata.GetNumberOfPoints()
    cell_num = surf_polydata.GetNumberOfCells()

    # Legacy vtkCellArray layout for triangles: [3, a, b, c] * cell_num
    polys = surf_polydata.GetPolys().GetData()
    arr = vtk_to_numpy(polys)
    # Fast path: reshape to (cell_num, 4) → take the last 3 columns as triangle vertex ids
    tri = arr.reshape(cell_num, 4)[:, 1:4]   # shape (N,3), columns = a,b,c

    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]

    # Mask of "sulci" vertices (your condition uses < 0)
    is_sulc = (sulc_data < 0)

    # Masks for edges that should be connected (both ends sulc < 0)
    m_ab = is_sulc[a] & is_sulc[b]
    m_ac = is_sulc[a] & is_sulc[c]
    m_bc = is_sulc[b] & is_sulc[c]

    # Collect all undirected edges to add
    u = np.concatenate([a[m_ab], a[m_ac], b[m_bc]]).astype(np.intp, copy=False)
    v = np.concatenate([b[m_ab], c[m_ac], c[m_bc]]).astype(np.intp, copy=False)

    # Array-of-sets is faster than dict-of-sets when ids are 0..n-1
    adj_sets = [set() for _ in range(n_pts)]
    # Insert both directions
    for ui, vi in zip(u, v):
        adj_sets[ui].add(int(vi))
        adj_sets[vi].add(int(ui))

    # Convert to requested type: defaultdict(list)
    out = defaultdict(list)
    for i, s in enumerate(adj_sets):
        if s:  # only store non-empty keys, like your original
            out[i] = list(s)
    return out



def delete_isolated_point(point_num, point_neighbor_points_dict, sulc_data):
    # Unpack marginals once
    marginal_points, marginal_points_gyri, marginal_points_sulc = \
        find_marginal_point(list(point_neighbor_points_dict.keys()),
                            point_neighbor_points_dict, sulc_data)

    # Boolean masks for O(1) membership checks
    marginal_gyri_mask = np.zeros(point_num, dtype=bool)
    marginal_gyri_mask[np.asarray(marginal_points_gyri, dtype=np.int64)] = True
    marginal_sulc_mask = np.zeros(point_num, dtype=bool)
    marginal_sulc_mask[np.asarray(marginal_points_sulc, dtype=np.int64)] = True

    # Normalize neighbors to NumPy arrays for faster indexing
    nbr = {}
    for p, lst in point_neighbor_points_dict.items():
        nbr[p] = np.asarray(lst, dtype=np.int64)

    # Main sweep (in-place like original)
    for point in range(point_num):
        sulc = sulc_data[point]
        if sulc == 0:
            continue  # matches original behavior (no branch triggers)

        n1 = nbr[point]
        n1_sulc = sulc_data[n1]

        # isolated point check (all neighbors opposite sign)
        if sulc > 0:
            if (n1_sulc < 0).all():
                sulc_data[point] = -1
                continue
            # one-round patch: exclude marginal gyri
            first = n1[~marginal_gyri_mask[n1]]
        else:  # sulc < 0
            if (n1_sulc > 0).all():
                sulc_data[point] = 1
                continue
            # one-round patch: exclude marginal sulc
            first = n1[~marginal_sulc_mask[n1]]

        # Build 2nd ring efficiently:
        # union of neighbors-of-first, excluding self and the first ring itself
        if first.size == 0:
            second = np.empty(0, dtype=np.int64)
        else:
            second_set = set()
            # Avoid creating a new list each iteration by reading from prebuilt arrays
            for f in first:
                second_set.update(nbr[f])
            # exclude self and first ring
            if second_set:
                second_set.discard(point)
                if first.size:
                    second_set.difference_update(first.tolist())
            second = np.fromiter(second_set, dtype=np.int64, count=len(second_set))

        # Note: .all() on empty arrays is True, matching your original sum==len behavior
        if sulc > 0:
            if (sulc_data[second] < 0).all():
                sulc_data[np.r_[point, first]] = -1
        else:  # sulc < 0
            if (sulc_data[second] > 0).all():
                sulc_data[np.r_[point, first]] = 1

    return sulc_data


def delete_isolated_point1(point_num, point_neighbor_points_dict, sulc_data):
    for point in range(point_num):
        sulc = sulc_data[point]
        '''isolated point'''
        nerighbor_sulcs = sulc_data[point_neighbor_points_dict[point]]
        if sulc > 0 and np.sum(nerighbor_sulcs < 0) == nerighbor_sulcs.shape[0]:
            sulc_data[point] = -1
        elif sulc < 0 and np.sum(nerighbor_sulcs > 0) == nerighbor_sulcs.shape[0]:
            sulc_data[point] = 1
        '''isolated one-round patch'''
        first_neighbor_list = point_neighbor_points_dict[point]
        second_neighbor_list = list()
        for first_neighbor in first_neighbor_list:
            neighbors_of_first_round = point_neighbor_points_dict[first_neighbor]
            for neighbor_of_first_round in neighbors_of_first_round:
                if neighbor_of_first_round not in [point] + first_neighbor_list:
                    second_neighbor_list.append(neighbor_of_first_round)
        second_neighbor_list = list(set(second_neighbor_list))
        if sulc > 0 and np.sum(sulc_data[second_neighbor_list] < 0) == len(second_neighbor_list):
            sulc_data[[point] + first_neighbor_list] = -1
        elif sulc < 0 and np.sum(sulc_data[second_neighbor_list] > 0) == len(second_neighbor_list):
            sulc_data[[point] + first_neighbor_list] = 1

    return sulc_data


def find_marginal_point(points_list, point_neighbor_points_dict, sulc_data):
    """
    A point is 'marginal' if its neighbors are not all <0 and not all >0.
    Return: (marginal_points, marginal_points_gyri, marginal_points_sulc) as lists.
    """
    n = len(sulc_data)
    # normalize neighbors to list[np.ndarray] for speed
    nbrs = _to_neighbors_list(point_neighbor_points_dict, n)

    marginal_points = []
    marginal_points_gyri = []
    marginal_points_sulc = []

    sulc = np.asarray(sulc_data)
    for p in points_list:
        idx = nbrs[p]
        if idx.size == 0:
            continue
        ns = sulc[idx]
        all_neg = (ns < 0).all()
        all_pos = (ns > 0).all()
        if not all_neg and not all_pos:
            marginal_points.append(p)
            if sulc[p] < 0:
                marginal_points_gyri.append(p)
            else:
                marginal_points_sulc.append(p)

    return marginal_points, marginal_points_gyri, marginal_points_sulc


def update_marginal_points_incremental(marginal_points_set,
                                       marginal_points_gyri_set,
                                       marginal_points_sulc_set,
                                       candidates_set,
                                       point_neighbor_points_dict,
                                       sulc_data):
    """
    Incrementally update marginal point sets based on a local candidate region.
    Only re-evaluates candidates and their 1-hop neighbors, avoiding full-surface scans.
    """
    # Points whose marginal status may change: candidates and their neighbors
    points_to_check = set(candidates_set)
    for p in list(candidates_set):
        points_to_check.update(point_neighbor_points_dict[p])

    for point in points_to_check:
        neighbor_points = point_neighbor_points_dict[point]
        if not neighbor_points:
            # isolated vertex shouldn't happen; ensure removed
            marginal_points_set.discard(point)
            marginal_points_gyri_set.discard(point)
            marginal_points_sulc_set.discard(point)
            continue
        neighbor_sulcs = sulc_data[neighbor_points]
        is_marginal = (np.sum(neighbor_sulcs < 0) != len(neighbor_points)) and (np.sum(neighbor_sulcs > 0) != len(neighbor_points))
        if is_marginal:
            marginal_points_set.add(point)
            if sulc_data[point] < 0:
                marginal_points_gyri_set.add(point)
                marginal_points_sulc_set.discard(point)
            else:
                marginal_points_sulc_set.add(point)
                marginal_points_gyri_set.discard(point)
        else:
            marginal_points_set.discard(point)
            marginal_points_gyri_set.discard(point)
            marginal_points_sulc_set.discard(point)

    return marginal_points_set, marginal_points_gyri_set, marginal_points_sulc_set


def write_featured_sphere_from_file(orig_sphere_polydata, feature_file_dict, output):
    point_num = orig_sphere_polydata.GetNumberOfPoints()
    cell_num = orig_sphere_polydata.GetNumberOfCells()

    f = open(output, 'w')
    f.write("# vtk DataFile Version 3.0\n")
    f.write("mesh surface\n")
    f.write("ASCII\n")
    f.write("DATASET POLYDATA\n")
    f.write("POINTS " + str(point_num) + " float\n")

    for point in range(point_num):
        coordinate = orig_sphere_polydata.GetPoints().GetPoint(point)
        f.write(str(coordinate[0]) + " " + str(coordinate[1]) + " " + str(coordinate[2]) + '\n')

    f.write("POLYGONS " + str(cell_num) + " " + str(4 * cell_num) + '\n')
    CellArray = orig_sphere_polydata.GetPolys()
    Polygons = CellArray.GetData()
    for i in range(0, cell_num):
        triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
        f.write(str(3) + " " + str(triangle[0]) + " " + str(triangle[1]) + " " + str(triangle[2]) + '\n')

    f.write("POINT_DATA " + str(point_num) + '\n')
    scale_dict = {'sulc': 0.0, 'curv': 0.0, 'thickness': 0.00000000001}
    for feature_name in scale_dict.keys():
        feature_file = feature_file_dict[feature_name]
        features = io.read_morph_data(feature_file)

        f.write("SCALARS " + feature_name + '_binary' + " float" + '\n')
        f.write("LOOKUP_TABLE " + feature_name + '_binary' + '\n')

        for point in range(point_num):
            if features[point] >= scale_dict[feature_name]:
                f.write(str(1) + '\n')
            elif features[point] < -1*scale_dict[feature_name]:
                f.write(str(-1) + '\n')
            else:
                f.write(str(0) + '\n')

        f.write("SCALARS " + feature_name + " float" + '\n')
        f.write("LOOKUP_TABLE " + feature_name + '\n')

        for point in range(point_num):
            f.write(str(features[point]) + '\n')

    f.close()


def write_featured_surface_from_surface(
    target_polydata,
    feature_source_polydata,
    output,
    *,
    thresholds=None,
    dtype=np.float32,
    binary_file=True,
    copy_polydata=True,
    replace_existing=True,
):
    """
    Write a surface VTK with binary and raw feature arrays copied from feature_source_polydata.

    For each feature in thresholds, this writes two point-data arrays:
      <name>_binary
      <name>
    """
    n_pts = target_polydata.GetNumberOfPoints()

    if thresholds is None:
        thresholds = {'sulc': 0.0, 'curv': 0.0, 'thickness': 1e-11}

    pd = vtk.vtkPolyData()
    if copy_polydata:
        pd.ShallowCopy(target_polydata)
    else:
        pd = target_polydata

    pdat = pd.GetPointData()
    src_pd = feature_source_polydata.GetPointData()

    if replace_existing:
        existing_names = {pdat.GetArrayName(i) for i in range(pdat.GetNumberOfArrays())}
        for base in thresholds.keys():
            for nm in (f"{base}_binary", base):
                if nm in existing_names:
                    pdat.RemoveArray(nm)

    vtk_scalar_type = vtk.VTK_FLOAT if np.dtype(dtype) == np.float32 else vtk.VTK_DOUBLE

    first_vtk_arr = None

    for feat_name, thr in thresholds.items():
        src_arr = src_pd.GetArray(feat_name)
        if src_arr is None:
            raise ValueError(f"Point data array '{feat_name}' not found on feature_source_polydata")

        feats = vtk_to_numpy(src_arr).astype(dtype, copy=False)

        if feats.ndim != 1 or feats.shape[0] != n_pts:
            raise ValueError(
                f"Feature '{feat_name}' must be 1D with length {n_pts}, got {feats.shape}"
            )

        bin_arr = np.zeros_like(feats, dtype=dtype)

        if thr == 0:
            bin_arr[feats > 0] = 1
            bin_arr[feats < 0] = -1
        else:
            bin_arr[feats >= thr] = 1
            bin_arr[feats <= -thr] = -1

        vtk_bin = numpy_to_vtk(
            num_array=np.ascontiguousarray(bin_arr),
            deep=False,
            array_type=vtk_scalar_type
        )
        vtk_bin.SetName(f"{feat_name}_binary")
        pdat.AddArray(vtk_bin)

        if first_vtk_arr is None:
            first_vtk_arr = vtk_bin

        vtk_raw = numpy_to_vtk(
            num_array=np.ascontiguousarray(feats),
            deep=False,
            array_type=vtk_scalar_type
        )
        vtk_raw.SetName(feat_name)
        pdat.AddArray(vtk_raw)

    if first_vtk_arr is not None:
        pdat.SetScalars(first_vtk_arr)

    if not output.lower().endswith(".vtk"):
        output = output + ".vtk"

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output)
    writer.SetInputData(pd)

    if binary_file:
        writer.SetFileTypeToBinary()
    else:
        writer.SetFileTypeToASCII()

    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTK file: {output}")


def write_featured_sphere_from_variable(orig_sphere_polydata,
                                        feature_name_variable_dict,
                                        output,
                                        *,
                                        dtype=np.float32,
                                        binary=True,
                                        set_active_scalars=True,
                                        copy_polydata=True,
                                        replace_existing=True):
    """
    Write a legacy .vtk PolyData file using VTK's writer (no manual line-by-line I/O).

    - orig_sphere_polydata : vtkPolyData (points & polys already set)
    - feature_name_variable_dict : {name: 1D array-like len == n_points}
    - output : path ('.vtk' will be appended if missing)
    - dtype : stored scalar dtype (default float32)
    - binary : write binary legacy .vtk (smaller/faster). Set False for ASCII.
    - set_active_scalars : make the first array the active Scalars
    - copy_polydata : ShallowCopy before adding arrays (don’t mutate the original)
    - replace_existing : if an array with the same name exists, replace it
    """
    n_pts = orig_sphere_polydata.GetNumberOfPoints()

    # Work on a shallow copy to avoid mutating the original mesh (safe & cheap)
    pd = vtk.vtkPolyData()
    if copy_polydata:
        pd.ShallowCopy(orig_sphere_polydata)
    else:
        pd = orig_sphere_polydata

    pdat = pd.GetPointData()

    # Optionally remove old arrays with the same names
    if replace_existing:
        existing_names = {pdat.GetArrayName(i) for i in range(pdat.GetNumberOfArrays())}
        for name in feature_name_variable_dict.keys():
            if name in existing_names:
                arr = pdat.GetArray(name)
                if arr is not None:
                    pdat.RemoveArray(name)

    first_vtk_arr = None
    for name, values in feature_name_variable_dict.items():
        arr = np.asarray(values, dtype=dtype)
        if arr.ndim != 1 or arr.shape[0] != n_pts:
            raise ValueError(f"Feature '{name}' must be 1D of length {n_pts}, got {arr.shape}")
        # Create VTK array without deep copy (fast); writer will serialize it
        vtk_arr = numpy_to_vtk(num_array=arr, deep=False,
                               array_type=vtk.VTK_FLOAT if np.dtype(dtype) == np.float32 else vtk.VTK_DOUBLE)
        vtk_arr.SetName(str(name))
        pdat.AddArray(vtk_arr)
        if first_vtk_arr is None:
            first_vtk_arr = vtk_arr

    if set_active_scalars and first_vtk_arr is not None:
        pdat.SetScalars(first_vtk_arr)

    # Ensure .vtk extension
    if not output.lower().endswith(".vtk"):
        output = output + ".vtk"

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output)
    writer.SetInputData(pd)
    if binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetFileTypeToASCII()

    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTK file: {output}")


def write_featured_sphere_from_variable_single(orig_sphere_polydata,
                                               feature_name,
                                               feature_variable,
                                               output,
                                               *,
                                               dtype=np.float32,
                                               binary=True,
                                               copy_polydata=True):
    """
    Write a legacy .vtk PolyData with one per-point scalar array.
    Faster than manual ASCII writes; uses vtkPolyDataWriter internally.

    - feature_variable must be 1D with length == number of points.
    - dtype controls storage (default float32 to match your original 'float').
    - binary=True for smaller/faster files (set False for ASCII).
    """
    n_pts = orig_sphere_polydata.GetNumberOfPoints()

    # Work on a shallow copy so we don't mutate the original mesh
    pd = vtk.vtkPolyData()
    if copy_polydata:
        pd.ShallowCopy(orig_sphere_polydata)
    else:
        pd = orig_sphere_polydata

    arr = np.asarray(feature_variable, dtype=dtype)
    if arr.ndim != 1 or arr.shape[0] != n_pts:
        raise ValueError(f"Feature array must be 1D of length {n_pts}, got {arr.shape}")

    vtk_type = vtk.VTK_FLOAT if np.dtype(dtype) == np.float32 else vtk.VTK_DOUBLE
    vtk_arr = numpy_to_vtk(num_array=arr, deep=False, array_type=vtk_type)
    vtk_arr.SetName(str(feature_name))

    pd.GetPointData().AddArray(vtk_arr)
    pd.GetPointData().SetScalars(vtk_arr)  # make it the active Scalars

    if not output.lower().endswith(".vtk"):
        output = output + ".vtk"

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output)
    writer.SetInputData(pd)
    writer.SetFileTypeToBinary() if binary else writer.SetFileTypeToASCII()

    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTK file: {output}")


def find_round_n_neighbor(point, n, point_neighbor_points_dict):
    """
    BFS expansion up to n rounds using sets to avoid duplicates; returns lists for compatibility.
    """
    visited = set()
    current_outer = {point}
    while n > 0:
        visited |= current_outer
        next_outer = set()
        for p in current_outer:
            for neighbor in point_neighbor_points_dict[p]:
                if neighbor not in visited:
                    next_outer.add(neighbor)
        current_outer = next_outer
        n -= 1
    neighbor_list = list(visited | current_outer)
    return neighbor_list, list(current_outer)


def find_missing_gyri_by_sulc_and_curv(sulc_data, curv_data, outer_gyri_curv_thres):
    idx = np.flatnonzero((sulc_data >= 0) & (curv_data < outer_gyri_curv_thres))
    return idx.tolist()




def find_inner_sulci_by_sulc_and_curv(
    sulc_data,
    curv_data,
    point_neighbor_points_dict,
    inner_sulci_curv_thres,
    inner_sulci_round_thres,
    inner_sulci_neighbor_curv_thres,
):
    """
    Find 'inner sulci' points based on sulc, curvature, and local neighborhood checks.

    Parameters
    ----------
    sulc_data : np.ndarray
        Per-vertex sulc values (negative = sulci, positive = gyri).
    curv_data : np.ndarray
        Per-vertex curvature values.
    point_neighbor_points_dict : Dict[int, Sequence[int]]
        Mapping from vertex index -> list/array of 1-ring neighbors.
        This is passed to `find_round_n_neighbor` to obtain n-ring neighborhoods.
    inner_sulci_curv_thres : float
        Candidate center must have curv_data[point] > this threshold.
    inner_sulci_round_thres : int
        The 'n' for n-ring neighbors (argument to `find_round_n_neighbor`).
    inner_sulci_neighbor_curv_thres : float
        The n-ring must contain at least one vertex with curvature > this threshold.

    Returns
    -------
    List[int]
        List of neighbor indices that satisfy the conditions.
        Order and duplicates are preserved exactly like the original implementation.
    """

    # Local aliases (micro-optimization: fewer attribute lookups)
    curv = curv_data
    sulc = sulc_data

    # Precompute boolean masks to avoid repeating the same comparisons in the loop
    # Candidate center vertices: sulc < 0 and curv > inner_sulci_curv_thres
    cand_mask = (sulc < 0) & (curv > inner_sulci_curv_thres)

    # For the n-ring check: whether a vertex has curvature above neighbor threshold
    curv_gt_neighbor_thres = (curv > inner_sulci_neighbor_curv_thres)

    # For the final append step: we only append neighbors with positive curvature
    curv_pos = (curv > 0)

    result = []

    # Iterate only over candidates (saves time vs. scanning every vertex)
    for p in np.flatnonzero(cand_mask):
        # Get:
        #  - neighbor_collection: neighbors we may append to result (keep order/dups)
        #  - round_n_neighbors: vertices used to decide if the n-ring passes the test
        neighbor_collection, round_n_neighbors = find_round_n_neighbor(
            p, inner_sulci_round_thres, point_neighbor_points_dict
        )

        # Convert to ndarray for fast boolean indexing
        rnn = np.asarray(round_n_neighbors, dtype=np.int64)

        # Check if the n-ring contains ANY vertex with curvature > threshold.
        # .any() is faster and clearer than np.sum(...) > 0.
        # Note: .any() on an empty array is False, matching np.sum(...) > 0.
        if curv_gt_neighbor_thres[rnn].any():
            # Only if the ring passes, filter neighbor_collection by curv > 0
            nc = np.asarray(neighbor_collection, dtype=np.int64)

            # Boolean mask preserves original order and duplicates
            keep = curv_pos[nc]
            if keep.any():
                # Convert to Python ints and extend the result list
                result.extend(nc[keep].tolist())

    return result


def update_sulc_data(original_sulc_data, missing_gyri_point_list, inner_sulci_point_list):
    updated_sulc_data = np.array(original_sulc_data, copy=True)
    if len(missing_gyri_point_list):
        updated_sulc_data[np.array(missing_gyri_point_list, dtype=int)] = -1
    if len(inner_sulci_point_list):
        updated_sulc_data[np.array(inner_sulci_point_list, dtype=int)] = 1
    return updated_sulc_data


def thickness_denoise(point_num, point_neighbor_points_dict, thickness_data_raw):
    # Work in-place (same as your original aliasing behavior)
    thickness_data = thickness_data_raw

    # Boolean mask of "positive" (thickness > 0); keep it in sync with thickness_data updates
    pos = (thickness_data > 0)

    # Iterate only points that are zero
    zero_points = np.flatnonzero(~pos)

    # Cache for n-ring queries: (point, step) -> (neighbor_idx_array, outer_idx_array)
    cache = {}

    def get_round_n(point, step):
        key = (point, step)
        if key in cache:
            return cache[key]
        # Expect lists; convert once to numpy arrays for fast boolean indexing
        nbrs, outer = find_round_n_neighbor(point, step, point_neighbor_points_dict)
        nbrs = np.asarray(nbrs, dtype=np.int64)
        outer = np.asarray(outer, dtype=np.int64)
        cache[key] = (nbrs, outer)
        return nbrs, outer

    for p in zero_points:
        # Skip if p was turned positive by an earlier update
        if pos[p]:
            continue
        for step in range(1, 6):  # 1..5
            nbrs, outer = get_round_n(p, step)
            # Check all outer-ring points are positive (no sums; no temporary int array)
            if pos[outer].all():
                # Promote the whole neighborhood
                thickness_data[nbrs] = 1
                pos[nbrs] = True  # keep mask consistent, avoids recomputing (thickness_data > 0)
                break
    return thickness_data

def update_and_delete_isolated(
    point_num,
    point_neighbor_points_dict,
    sulc_data,
    missing_gyri_point_list,
    inner_sulci_point_list,
):
    """
    Equivalent to:
      out = update_sulc_data(sulc_data, missing_gyri_point_list, inner_sulci_point_list)
      out = delete_isolated_point(point_num, point_neighbor_points_dict, out)
    but in one pass with fewer allocations. Returns a new np.ndarray.
    """

    # 1) Start from a copy (same dtype as input; original remains unchanged)
    out = np.array(sulc_data, copy=True)

    # 2) Apply index updates (inner_sulci wins on overlaps, matching your original order)
    mg = np.asarray(missing_gyri_point_list, dtype=np.intp)
    if mg.size:
        out[mg] = -1
    ins = np.asarray(inner_sulci_point_list, dtype=np.intp)
    if ins.size:
        out[ins] = 1

    # 3) Prepare data structures for the isolation cleanup
    #    (marginals computed from the already-updated array, same as your pipeline)
    marginal_points, marginal_points_gyri, marginal_points_sulc = \
        find_marginal_point(list(point_neighbor_points_dict.keys()),
                            point_neighbor_points_dict, out)

    marginal_gyri_mask = np.zeros(point_num, dtype=bool)
    if len(marginal_points_gyri):
        marginal_gyri_mask[np.asarray(marginal_points_gyri, dtype=np.intp)] = True

    marginal_sulc_mask = np.zeros(point_num, dtype=bool)
    if len(marginal_points_sulc):
        marginal_sulc_mask[np.asarray(marginal_points_sulc, dtype=np.intp)] = True

    # Normalize adjacency to NumPy arrays once
    neighbors = [np.asarray(point_neighbor_points_dict[p], dtype=np.intp)
                 for p in range(point_num)]

    # Sign masks kept in sync with writes to `out`
    pos = (out > 0)
    neg = (out < 0)

    # Reusable stamping buffer to build the 2nd ring without Python sets
    stamp_buf = np.zeros(point_num, dtype=np.int32)
    stamp = np.int32(1)

    # 4) Isolation cleanup (same logic as your optimized version)
    for point in range(point_num):
        s = out[point]
        if s == 0:
            continue

        n1 = neighbors[point]
        if s > 0:
            if neg[n1].all():
                out[point] = -1
                pos[point] = False
                neg[point] = True
                continue
            first = n1[~marginal_gyri_mask[n1]]
        else:  # s < 0
            if pos[n1].all():
                out[point] = 1
                pos[point] = True
                neg[point] = False
                continue
            first = n1[~marginal_sulc_mask[n1]]

        if first.size == 0:
            second = np.empty(0, dtype=np.intp)
        else:
            stamp += 1
            for f in first:
                stamp_buf[neighbors[f]] = stamp
            stamp_buf[point] = stamp + 1
            stamp_buf[first] = stamp + 1
            second = np.flatnonzero(stamp_buf == stamp)
            stamp += 1

        # Note: .all() on empty arrays is True (same as sum==len behavior)
        if s > 0:
            if neg[second].all():
                idx = np.r_[point, first]
                out[idx] = -1
                pos[idx] = False
                neg[idx] = True
        else:
            if pos[second].all():
                idx = np.r_[point, first]
                out[idx] = 1
                pos[idx] = True
                neg[idx] = False

    return out
def initialize_sulc_data(orig_sphere_polydata, orig_surf_polydata, point_neighbor_points_dict, inner_sulci_curv_thres, inner_sulci_round_thres, outer_gyri_curv_thres, inner_sulci_neighbor_curv_thres, output_prefix):
    point_num = orig_sphere_polydata.GetNumberOfPoints()
    # cell_num = orig_sphere_polydata.GetNumberOfCells()
    # CellArray = orig_sphere_polydata.GetPolys()
    # Polygons = CellArray.GetData()

    print('update the sulc value by curv:\t' + time.asctime(time.localtime(time.time())))

    thickness_data_raw = get_point_data_array(orig_surf_polydata, 'thickness')
    print("NaN count:", np.isnan(thickness_data_raw).sum())
    print("Min:", np.nanmin(thickness_data_raw))
    print("Max:", np.nanmax(thickness_data_raw))
    nan_mask = np.isnan(thickness_data_raw)
    if nan_mask.any():
        print(f"[INFO] Found {nan_mask.sum()} NaN values in thickness")
        thickness_data_raw[nan_mask] = 0.0

    thickness_data = thickness_denoise(point_num, point_neighbor_points_dict, thickness_data_raw)
    sulc_data_raw = get_point_data_array(orig_surf_polydata, 'sulc')

    sulc_data_delete_thicknessZero = sulc_data_raw.copy()
    sulc_data_delete_thicknessZero[thickness_data <= 0] = 0
    sulc_data_binary = np.where(sulc_data_delete_thicknessZero >= 0, 1, -1)
    sulc_data = delete_isolated_point(point_num, point_neighbor_points_dict, sulc_data_binary)

    curv_data_raw = get_point_data_array(orig_surf_polydata, 'curv')
    print('before delete thickness zero, curv data stats:')
    print("min:", np.nanmin(curv_data_raw))
    print("max:", np.nanmax(curv_data_raw))
    print("NaN count:", np.isnan(curv_data_raw).sum())
    curv_data_delete_thicknessZero = np.where(thickness_data > 0, curv_data_raw, 0)
    print('after delete thickness zero, curv data stats:')
    print("min:", np.nanmin(curv_data_raw))
    print("max:", np.nanmax(curv_data_raw))
    print("NaN count:", np.isnan(curv_data_raw).sum())
    missing_gyri_point_list = find_missing_gyri_by_sulc_and_curv(sulc_data, curv_data_delete_thicknessZero, outer_gyri_curv_thres)
    inner_sulci_point_list = find_inner_sulci_by_sulc_and_curv(sulc_data, curv_data_delete_thicknessZero, point_neighbor_points_dict, inner_sulci_curv_thres, inner_sulci_round_thres, inner_sulci_neighbor_curv_thres)
    # updated_sulc_data = update_sulc_data(sulc_data, missing_gyri_point_list, inner_sulci_point_list)
    # updated_sulc_data = delete_isolated_point(point_num, point_neighbor_points_dict, updated_sulc_data)
    updated_sulc_data = update_and_delete_isolated(point_num, point_neighbor_points_dict, sulc_data, missing_gyri_point_list, inner_sulci_point_list)

    print('draw updated sulc colorful sphere:\t' + time.asctime(time.localtime(time.time())))
    feature_name_variable_dict = {'thickness_data_raw': thickness_data_raw, 'thickness_de_noise': thickness_data,
                                  'sulc_data_raw': sulc_data_raw,
                                  'sulc_data_delete_thicknessZero': sulc_data_delete_thicknessZero,
                                  'sulc_data_binary': sulc_data_binary, 'sulc_data_de_isolation': sulc_data,
                                  'curv_data_raw': curv_data_raw,
                                  'curv_data_delete_thicknessZero': curv_data_delete_thicknessZero,
                                  'updated_sulc_data': updated_sulc_data}
    # write_featured_sphere_from_variable(orig_sphere_polydata, feature_name_variable_dict,
    #                                     output_prefix + '_sphere_feature_updated.vtk')
    write_featured_sphere_from_variable(orig_surf_polydata, feature_name_variable_dict,
                                        output_prefix + '_surf_feature_updated.vtk')
    return updated_sulc_data, sulc_data, curv_data_delete_thicknessZero


def write_thin_gyri_on_sphere_point_marginal_sulc_curv(orig_sphere_polydata, orig_surf_polydata, point_neighbor_points_dict, updated_sulc_data, curv_data_delete_thicknessZero, expend_curv_step_size, output_prefix):
    print('================= begin erosion:\t' + time.asctime(time.localtime(time.time())) + '=========================')
    point_num = orig_sphere_polydata.GetNumberOfPoints()
    cell_num = orig_sphere_polydata.GetNumberOfCells()
    CellArray = orig_sphere_polydata.GetPolys()
    Polygons = CellArray.GetData()

    print('create triangles connection:\t' + time.asctime(time.localtime(time.time())))
    triangle_set = set()
    for cell in range(cell_num):
        v1 = Polygons.GetValue(cell * 4 + 1)
        v2 = Polygons.GetValue(cell * 4 + 2)
        v3 = Polygons.GetValue(cell * 4 + 3)
        triangle_set.add(frozenset((v1, v2, v3)))

    print('find initial marginal points:\t' + time.asctime(time.localtime(time.time())))
    marginal_points, marginal_points_gyri, marginal_points_sulc = find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, updated_sulc_data)
    # Initialize incremental sets
    marginal_points_set = set(marginal_points)
    marginal_points_gyri_set = set(marginal_points_gyri)
    marginal_points_sulc_set = set(marginal_points_sulc)
    # pdb.set_trace()

    # min_curv = curv_data_delete_thicknessZero.min()
    # print('min curv:', min_curv)
    min_curv = np.nanmin(curv_data_delete_thicknessZero)
    print('min curv:', min_curv)
    print('begin expending:\t' + time.asctime(time.localtime(time.time())))
    round = 0
    current_curv = 0
    curv_step = expend_curv_step_size

    while current_curv >= min_curv:
        current_curv = current_curv - curv_step
        finish = 1
        print('\n================ current curv:', current_curv, '====================')
        curv_mask = (curv_data_delete_thicknessZero >= current_curv)
        while finish > 0:
            round += 1
            if round % 50 == 1:
                print('round', str(round), ':\t', time.asctime(time.localtime(time.time())))
            finish = 0
            redlize_points = list()
            marginal_points_candidate_set = set()
            updated_sulc_negative = (updated_sulc_data < 0)
            for point in marginal_points_sulc:
                sulc = updated_sulc_data[point]
                if sulc < 0:
                    print('updated_sulc_data sulc value error!')
                    exit()
                else:
                    # neighbor_points = point_neighbor_points_dict[point]
                    neighbor_points = point_neighbor_points_dict[point]

                    for neighbor_point in neighbor_points:
                        if updated_sulc_negative[neighbor_point] and curv_mask[neighbor_point]:
                            second_neighbors = point_neighbor_points_dict[neighbor_point]
                            marginal_gyri_num = 0
                            marginal_gyri_points = list()
                            for second_neighbor in second_neighbors:
                                if updated_sulc_negative[second_neighbor] and second_neighbor in marginal_points_gyri_set:
                                    marginal_gyri_num += 1
                                    marginal_gyri_points.append(second_neighbor)
                            is_triangle = False
                            if marginal_gyri_num == 2:
                                tri = frozenset((marginal_gyri_points[0], marginal_gyri_points[1], neighbor_point))
                                is_triangle = tri in triangle_set
                            if (marginal_gyri_num < 2) or (marginal_gyri_num == 2 and (np.sum(updated_sulc_negative[second_neighbors]) > 2 or is_triangle)):
                                redlize_points.append(neighbor_point)
                                for sn in second_neighbors:
                                    marginal_points_candidate_set.add(sn)
                                finish = 1
            updated_sulc_data[redlize_points] = 1
            # Incremental update on candidate neighborhood
            marginal_points_set, marginal_points_gyri_set, marginal_points_sulc_set = update_marginal_points_incremental(
                marginal_points_set,
                marginal_points_gyri_set,
                marginal_points_sulc_set,
                marginal_points_candidate_set,
                point_neighbor_points_dict,
                updated_sulc_data
            )
            marginal_points = list(marginal_points_set)
            marginal_points_gyri = list(marginal_points_gyri_set)
            marginal_points_sulc = list(marginal_points_sulc_set)
            # pdb.set_trace()
            if round % 500 == 0:
                write_featured_sphere_from_variable(orig_sphere_polydata, {'sulc_updated': updated_sulc_data}, output_prefix + '_surf_round(' + str(round) + ')_marginal_point.vtk')

    print('draw final thin path on surface:\t' + time.asctime(time.localtime(time.time())))
    # write_featured_sphere_from_variable(orig_sphere_polydata, {'sulc_updated': updated_sulc_data}, output_prefix + '_sphere_thin.vtk')
    write_featured_sphere_from_variable(orig_surf_polydata, {'sulc_updated': updated_sulc_data}, output_prefix + '_surf_thin.vtk')
    return updated_sulc_data


def find_the_patchSize_of_gyri_point(sulc_data, point_neighbor_points_dict):
    """
    Returns {point: patch_size}, where:
      - sulc_data[point] > 0  -> -1
      - sulc_data[point] == 0 -> 0
      - sulc_data[point] < 0  -> graph distance in rings to the nearest vertex with sulc >= 0
    """
    n = len(sulc_data)
    # neighbors: assume dict[int] -> Sequence[int]
    neighbors = point_neighbor_points_dict

    # Multi-source BFS from all non-sulci (sulc >= 0)
    non_sulci = (sulc_data >= 0)
    dist = np.full(n, -1, dtype=np.int32)

    q = deque()
    for i in range(n):
        if non_sulci[i]:
            dist[i] = 0
            q.append(i)

    # Propagate distances into sulci-only region
    while q:
        u = q.popleft()
        du = dist[u]
        for v in neighbors[u]:
            # Only enter sulci nodes that don't have a distance yet
            if sulc_data[v] < 0 and dist[v] == -1:
                dist[v] = du + 1
                q.append(v)

    # Build the dict result (same type as your original function)
    result = {}
    for i in range(n):
        if sulc_data[i] > 0:          # gyri
            result[i] = -1
        else:                          # sulci or zero
            result[i] = int(dist[i])   # 0 for zeros, >=1 for sulci; -1 only if no boundary reachable
    return result


def find_the_max_patchSize_point(unconnected_points_list, point_patchSize_dict_updated):
    """
    Retained for compatibility, but uses a heap-friendly single call.
    Note: if you adopt the heap version in create_tree below, this helper
    is no longer used.
    """
    if not unconnected_points_list:
        raise ValueError("unconnected_points_list is empty")
    pps = point_patchSize_dict_updated
    return max(unconnected_points_list, key=pps.__getitem__)

def _to_patch_array(point_patchSize_dict_updated, point_num):
    """
    Normalize patch sizes to a NumPy array for O(1) indexed access.
    Accepts dict[int->float] or array-like; returns np.ndarray shape (point_num,).
    Missing entries (dict) default to 0.0.
    """
    if isinstance(point_patchSize_dict_updated, (list, tuple, np.ndarray)):
        arr = np.asarray(point_patchSize_dict_updated, dtype=float)
        if arr.shape[0] != point_num:
            raise ValueError("patch size array length != point_num")
        return arr
    # dict path
    arr = np.zeros(point_num, dtype=float)
    for k, v in point_patchSize_dict_updated.items():
        arr[int(k)] = float(v)
    return arr

def find_the_max_patchSize_candidate_connection(point_patchSize_dict_updated, connected_candidate_dict):
    """
    Choose among neighbors grouped by 'point' (child) with:
      1) largest patch size at the child (point),
      2) tie-break by largest patch size at the parent (candidate[1]).
    Returns one [child, parent] pair.
    """
    pps = point_patchSize_dict_updated

    # Track best child patch size and the list of candidate connections for those best children
    best_child_size = -1
    best_conn_list = []  # list of [child, parent]

    for child, conns in connected_candidate_dict.items():
        s = pps[child]
        if s > best_child_size:
            best_child_size = s
            # copy to avoid aliasing the dict's list
            best_conn_list = list(conns)
        elif s == best_child_size:
            # extend our own list, keep order/dups
            if conns:
                best_conn_list.extend(conns)

    # If only one connection is in the best group, return it
    if len(best_conn_list) == 1:
        return best_conn_list[0]

    # Otherwise pick the one whose PARENT has the largest patch size.
    # max(...) returns the first on ties, matching original "> only" behavior.
    return max(best_conn_list, key=lambda cand: pps[cand[1]])


def write_skelenton_by_connectionPair(orig_sphere_polydata, connected_lines_list, output):
    points_new = vtk.vtkPoints()
    lines_cell_new = vtk.vtkCellArray()
    line_id = 0
    for line in connected_lines_list:
        coordinate1 = orig_sphere_polydata.GetPoints().GetPoint(line[0])
        points_new.InsertNextPoint(coordinate1)
        coordinate2 = orig_sphere_polydata.GetPoints().GetPoint(line[1])
        points_new.InsertNextPoint(coordinate2)

        line = vtk.vtkLine()
        line.GetPointIds().SetNumberOfIds(2)
        line.GetPointIds().SetId(0, line_id*2)
        line.GetPointIds().SetId(1, line_id*2 + 1)
        lines_cell_new.InsertNextCell(line)
        line_id += 1

    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points_new)
    polygonPolyData.SetLines(lines_cell_new)
    polygonPolyData.Modified()

    if vtk.VTK_MAJOR_VERSION <= 5:
        polygonPolyData = polygonPolyData.GetProducerPort()
    else:
        polygonPolyData = polygonPolyData

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polygonPolyData)
    writer.SetFileName(output)
    writer.Write()


def write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, connected_lines_list, output):
    point_num = orig_sphere_polydata.GetNumberOfPoints()
    cell_num = orig_sphere_polydata.GetNumberOfCells()
    points_new = vtk.vtkPoints()
    lines_cell_new = vtk.vtkCellArray()
    for point in range(point_num):
        coordinate = orig_sphere_polydata.GetPoints().GetPoint(point)
        points_new.InsertNextPoint(coordinate)

    for connection in connected_lines_list:
        line = vtk.vtkLine()
        line.GetPointIds().SetNumberOfIds(2)
        line.GetPointIds().SetId(0, connection[0])
        line.GetPointIds().SetId(1, connection[1])
        lines_cell_new.InsertNextCell(line)

    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points_new)
    polygonPolyData.SetLines(lines_cell_new)
    polygonPolyData.Modified()

    if vtk.VTK_MAJOR_VERSION <= 5:
        polygonPolyData = polygonPolyData.GetProducerPort()
    else:
        polygonPolyData = polygonPolyData

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polygonPolyData)
    writer.SetFileName(output)
    writer.Write()


def create_gyri_connection_count(point_connect_points_dict_thin_gyri_parts):
    # Build {point: number_of_neighbors} in one pass without extra dict indexing
    return {point: len(neighbors)
            for point, neighbors in point_connect_points_dict_thin_gyri_parts.items()}


def create_tree(orig_sphere_polydata, orig_surf_polydata,
                point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts,
                thin_sulc_data, output_prefix):
    print('================= create tree:\t' + time.asctime(time.localtime()) + ' =========================')
    point_num = orig_sphere_polydata.GetNumberOfPoints()

    # Normalize adjacency to NumPy arrays for fast indexing
    neighbors = [np.asarray(point_connect_points_dict_thin_gyri_parts[p], dtype=np.intp)
                 for p in range(point_num)]

    # Patch sizes as an array
    pps = _to_patch_array(point_patchSize_dict_updated, point_num)

    # Degree per point (modifiable)
    gyri_connection_count = np.array(
        [len(point_connect_points_dict_thin_gyri_parts[p]) for p in range(point_num)],
        dtype=np.int32
    )

    # Unconnected = sulci (<0). Track with a boolean mask for O(1) checks.
    unconnected_mask = (np.asarray(thin_sulc_data) < 0)

    connected_points_list = []
    connected_points_set  = set()
    connected_lines_list  = []        # list of [child, parent]
    father_dict = {}

    # Max-heap (lazy) for picking the next seed of a new component
    # Store (-patch_size, point) so largest pops first.
    seed_heap = [(-pps[i], int(i)) for i in range(point_num) if unconnected_mask[i]]
    heapq.heapify(seed_heap)

    # Candidate children heap keyed by child patch size (lazy)
    cand_heap = []  # elements: (-pps[child], child)
    # For each candidate child, keep the list of possible parents (connected nodes)
    candidate_parents = defaultdict(list)

    def add_seed(point):
        """Connect a new component seed."""
        connected_points_list.append(point)
        connected_points_set.add(point)
        unconnected_mask[point] = False
        father_dict[point] = point
        # Seed contributes candidates: its unconnected neighbors can attach to it
        for nbr in neighbors[point]:
            if unconnected_mask[nbr]:
                if not candidate_parents[nbr]:
                    heapq.heappush(cand_heap, (-pps[nbr], int(nbr)))
                candidate_parents[nbr].append(point)

    def attach_child(child, parent):
        """Attach child to parent and update structures."""
        connected_points_list.append(child)
        connected_points_set.add(child)
        unconnected_mask[child] = False
        connected_lines_list.append([int(child), int(parent)])
        father_dict[child] = int(parent)

        # Update degrees for neighbors of the child
        for c in neighbors[child]:
            gyri_connection_count[c] -= 1

        # Child itself can now serve as parent for its unconnected neighbors
        # (this grows the frontier incrementally)
        for nbr in neighbors[child]:
            if unconnected_mask[nbr]:
                if not candidate_parents[nbr]:
                    heapq.heappush(cand_heap, (-pps[nbr], int(nbr)))
                candidate_parents[nbr].append(child)

        # Optional: free memory for child's candidate list if it existed
        if child in candidate_parents:
            del candidate_parents[child]

    # --- Main loop ---
    while unconnected_mask.any():
        # Try to attach the best candidate child (if any)
        chosen = None
        while cand_heap:
            # Lazy pop: skip children that are no longer unconnected
            _, child = heapq.heappop(cand_heap)
            if not unconnected_mask[child]:
                continue
            # Pick the best parent for this child (max pps among valid-degree parents)
            valid_parents = [p for p in candidate_parents[child] if gyri_connection_count[p] > 0]
            if not valid_parents:
                # No valid parent anymore; drop this candidate
                del candidate_parents[child]
                continue
            parent = max(valid_parents, key=lambda pr: pps[pr])
            chosen = (child, parent)
            break

        if chosen is not None:
            child, parent = chosen
            attach_child(child, parent)
            continue

        # No candidate available: start a new component from the best remaining seed
        while seed_heap and not unconnected_mask[seed_heap[0][1]]:
            heapq.heappop(seed_heap)  # discard stale entries
        if not seed_heap:
            break  # should not happen; safety
        _, seed = heapq.heappop(seed_heap)
        add_seed(seed)

    # Sanity check for duplicated reverse connections
    seen = set()
    for a, b in connected_lines_list:
        if (b, a) in seen:
            print('connection already in connected_lines_list')
            raise SystemExit(1)
        seen.add((a, b))

    write_skelenton_by_connectionPair(
        orig_sphere_polydata, connected_lines_list, output_prefix + '_sphere_initial_tree.vtk'
    )
    write_skelenton_by_connectionPair(
        orig_surf_polydata, connected_lines_list, output_prefix + '_surf_initial_tree.vtk'
    )
    return connected_lines_list, father_dict


def find_skelenton(orig_sphere_polydata, orig_surf_polydata,
                   point_patchSize_dict_updated,
                   curv_data_delete_thicknessZero,
                   original_sulc_data, thin_sulc_data,
                   point_neighbor_points_dict,
                   point_connect_points_dict_thin_gyri_parts,
                   connected_lines_list, father_dict,
                   length_thres_of_long_gyri, neighbor_missing_path_smallest_step,
                   flat_threshold_for_convex_gyri, nearest_skeleton_num,
                   island_gyri_length_thres, output_prefix):
    print('================= build skeleton:\t' + time.asctime(time.localtime(time.time())) + ' =======================')

    n = orig_sphere_polydata.GetNumberOfPoints()
    # normalize helpers
    father = _to_int_array(father_dict, n)
    # gyri_nbrs = _to_neighbors_list(point_connect_points_dict_thin_gyri_parts, n)

    # marginal points (lists as before)
    marginal_points, marginal_points_gyri, marginal_points_sulc = \
        find_marginal_point(list(range(n)), point_neighbor_points_dict, thin_sulc_data)

    # ---- trimming stage ----
    print('begin trimming  \t' + time.asctime(time.localtime(time.time())))
    tree_adj = create_tree_connection_dict(connected_lines_list)  # LoS
    inner_gyri_mask = np.ones(n, dtype=bool)
    inner_gyri_mask[marginal_points_gyri] = False
    tree_adj, deleted_round1_points = trim_allRound1_long(
        tree_adj, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts,
        father, inner_gyri_mask
    )
    initial_edges = create_connectingPair(tree_adj)
    print('draw initial skelenton  \t' + time.asctime(time.localtime(time.time())))
    # write_skelenton_by_connectionPair(orig_sphere_polydata, initial_edges, output_prefix + '_sphere_initial_skelenton.vtk')
    write_skelenton_by_connectionPair(orig_surf_polydata,   initial_edges, output_prefix + '_surf_initial_skelenton.vtk')

    # ---- connect breaks ----
    print('connecting breaks in skelenton  \t' + time.asctime(time.localtime(time.time())))
    final_adj = connect_break_in_circle(tree_adj, point_connect_points_dict_thin_gyri_parts,
                                        deleted_round1_points, father, verbose=False)
    final_edges = create_connectingPair(final_adj)
    # write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, final_edges, output_prefix + '_sphere_skelenton_connect_break_allpoints.vtk')
    write_allPoints_and_skelenton_by_connectionPair(orig_surf_polydata,   final_edges, output_prefix + '_surf_skelenton_connect_break_allpoints.vtk')

    # old code: "clear_empty_point_in_dict" – not needed for LoS; edges already reflect connectivity

    # ---- missing gyri ----
    print('find missing gyri  \t' + time.asctime(time.localtime(time.time())))
    final_adj = create_tree_connection_dict(final_edges)  # LoS from edges for the next stage
    final_adj = create_connection_for_missing_gyri(
        orig_surf_polydata, orig_sphere_polydata, final_adj, original_sulc_data,
        curv_data_delete_thicknessZero, length_thres_of_long_gyri, neighbor_missing_path_smallest_step,
        flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix
    )  # <-- NOTE: update create_connection_for_missing_gyri to accept LoS; return LoS

    all_edges = create_connectingPair(final_adj)
    # write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, all_edges, output_prefix + '_sphere_skelenton_allpoints_final.vtk')
    write_allPoints_and_skelenton_by_connectionPair(orig_surf_polydata,   all_edges, output_prefix + '_surf_skelenton_allpoints_final.vtk')

    all_point_lines_adj = create_tree_connection_dict(all_edges)  # LoS

    # ---- 3-hinge ----
    print('find and draw 3hinge ' + time.asctime(time.localtime(time.time())))
    hinge3_list = create_3hinge(all_point_lines_adj)  # list[int], same as before
    with open(output_prefix + '_3hinge_ids.txt', "w") as f:
        for h in hinge3_list:
            f.write(str(h) + "\n")
    # draw_3hinge_on_surf(orig_sphere_polydata, hinge3_list, output_prefix + '_sphere_3hinge_vertex.vtk')
    draw_3hinge_on_surf(orig_surf_polydata,   hinge3_list, output_prefix + '_surf_3hinge_vertex.vtk')



def los_to_nx(adj_los, *, add_coords=None):
    """
    Convert a list-of-sets adjacency (LoS) to a networkx.Graph.

    Parameters
    ----------
    adj_los : List[Set[int]]
        adj_los[i] is the set of neighbors of vertex i. Missing/empty allowed.
    add_coords : Optional[Union[vtk.vtkPoints, np.ndarray shape (N,3)]]
        If provided, attach node attributes x,y,z.

    Returns
    -------
    G : nx.Graph
    """
    n = len(adj_los)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Edges (add each undirected edge once)
    for u, nbrs in enumerate(adj_los):
        for v in nbrs:
            if u < v:  # avoid duplicates
                G.add_edge(u, v)

    # Optional: attach coordinates from VTK or ndarray
    if add_coords is not None:
        if hasattr(add_coords, 'GetNumberOfPoints'):  # vtk.vtkPoints
            for i in range(n):
                x, y, z = add_coords.GetPoint(i)
                G.nodes[i]['x'] = float(x)
                G.nodes[i]['y'] = float(y)
                G.nodes[i]['z'] = float(z)
        else:
            coords = np.asarray(add_coords)
            if coords.shape[0] != n or coords.shape[1] != 3:
                raise ValueError(f"coords must be (N,3), got {coords.shape}")
            for i in range(n):
                x, y, z = coords[i]
                G.nodes[i]['x'] = float(x)
                G.nodes[i]['y'] = float(y)
                G.nodes[i]['z'] = float(z)

    return G


def nx_to_los(G, *, size=None):
    """
    Convert a networkx.Graph back to list-of-sets adjacency (LoS).

    Parameters
    ----------
    G : nx.Graph
    size : Optional[int]
        If provided, pad the LoS to this number of nodes.
        Defaults to max_node_id+1.

    Returns
    -------
    adj_los : List[Set[int]]
    """
    if size is None:
        size = (max(G.nodes) + 1) if len(G) else 0
    adj_los = [set() for _ in range(size)]
    for u, v in G.edges():
        if u >= size or v >= size:
            raise ValueError("size too small for node ids in G")
        adj_los[u].add(v)
        adj_los[v].add(u)
    return adj_los

def keep_largest_component_inplace(G: nx.Graph) -> None:
    """Remove all nodes not in the largest connected component (modifies G in-place)."""
    if G.number_of_nodes() == 0:
        return
    largest_cc = max(nx.connected_components(G), key=len)
    G.remove_nodes_from(set(G.nodes()) - set(largest_cc))

def keep_largest_component_copy(G: nx.Graph) -> nx.Graph:
    """Return a new graph containing only the largest connected component."""
    if G.number_of_nodes() == 0:
        return G.copy()
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()

def keep_largest_component_edges_inplace(G: nx.Graph):
    """
    Keep all nodes, but remove edges that are not entirely inside
    the largest connected component. This preserves node IDs, so
    downstream arrays (VTK point indices) remain valid.
    """
    if G.number_of_edges() == 0:
        return
    comps = list(nx.connected_components(G))
    if not comps:
        return
    largest = max(comps, key=len)
    G.remove_edges_from([(u, v) for u, v in G.edges()
                         if u not in largest or v not in largest])


def build_surface_graph(polydata):
    """Mesh 1-ring graph with Euclidean edge weights."""
    pts = polydata.GetPoints()
    nbrs = get_connect_points(polydata)  # you already have this
    Gs = nx.Graph()
    Gs.add_nodes_from(range(polydata.GetNumberOfPoints()))
    for u, vs in nbrs.items():
        u = int(u)
        x1, y1, z1 = pts.GetPoint(u)
        for v in vs:
            v = int(v)
            if u < v:
                x2, y2, z2 = pts.GetPoint(v)
                w = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
                Gs.add_edge(u, v, weight=w)
    return Gs

def skeleton_shortest_path(G, a, b):
    """Hop-shortest path on the skeleton."""
    return nx.shortest_path(G, int(a), int(b))

def middle_on_path(path, tie='right'):
    """Middle vertex on a skeleton path (by hops)."""
    L = len(path) - 1
    if L % 2 == 0:
        return int(path[L // 2])
    i1, i2 = (L-1)//2, (L+1)//2
    if tie == 'left': return int(path[i1])
    if tie == 'lower_id': return int(min(path[i1], path[i2]))
    return int(path[i2])

def walk_branch_k_hops(G, hub, first_step, k):
    """Chain [hub, n1, n2, ...] walking away up to k edges; stop at leaf or junction."""
    chain = [int(hub), int(first_step)]
    par, cur = int(hub), int(first_step); steps = 1
    while steps < k:
        nxt = next((w for w in G.neighbors(cur) if w != par), None)
        if nxt is None or G.degree(cur) != 2:
            break
        chain.append(int(nxt))
        par, cur = cur, int(nxt); steps += 1
    return chain

def choose_branch_to_edit(G, hub, corridor_neighbor, k=3):
    """Pick one non-corridor neighbor with the longest k-hop chain (tiebreak lower id)."""
    cand = [v for v in G.neighbors(hub) if v != corridor_neighbor]
    if not cand: return None, None
    best_n, best_chain, best_len = None, None, -1
    for n1 in cand:
        chain = walk_branch_k_hops(G, hub, n1, k)
        clen = len(chain) - 1
        if clen > best_len or (clen == best_len and (best_n is None or n1 < best_n)):
            best_n, best_chain, best_len = n1, chain, clen
    return best_n, best_chain

def cut_chain_edges(G, chain):
    """Remove edges along chain [v0,...,vt]; return cut endpoint t."""
    if not chain or len(chain) < 2: return None
    for i in range(len(chain)-1):
        u, v = chain[i], chain[i+1]
        if G.has_edge(u, v): G.remove_edge(u, v)
    return int(chain[-1])

def skeleton_edges_set(G):
    """Set of undirected edges (min,max) in the skeleton."""
    return {(min(u,v), max(u,v)) for u, v in G.edges()}

def routing_graph_excluding_skeleton(Gsurf, skel_edges, allow_touch_dst=None):
    """Surface routing graph with all current skeleton edges removed.
       If allow_touch_dst is set, keep edges that touch that node."""
    H = Gsurf.copy()
    if allow_touch_dst is None:
        H.remove_edges_from(skel_edges)
    else:
        to_remove = []
        for (u, v) in skel_edges:
            if u == allow_touch_dst or v == allow_touch_dst: continue
            if H.has_edge(u, v): to_remove.append((u, v))
        H.remove_edges_from(to_remove)
    return H


# ---- helpers: canonicalize and safe add/remove --------------------------------

def _canon_edge(u, v):
    u = int(u); v = int(v)
    return (u, v) if u <= v else (v, u)

def _add_edge_if_absent(G, u, v):
    """Canonicalize (u,v), skip self-loops, add only if missing. Returns True if added."""
    u, v = _canon_edge(u, v)
    if u == v:
        return False
    if not G.has_edge(u, v):
        G.add_edge(u, v)
        return True
    return False

def _remove_edge_if_present(G, u, v):
    """Canonicalize (u,v) and remove if present. Returns True if removed."""
    u, v = _canon_edge(u, v)
    if G.has_edge(u, v):
        G.remove_edge(u, v)
        return True
    return False

# (Optional) make sure we always work with canonical skeleton edge set
def skeleton_edges_set(G):
    """Set of undirected edges (min,max) in the skeleton."""
    return {_canon_edge(u, v) for u, v in G.edges()}

# ---- patched rewire -----------------------------------------------------------

def rewire_pair_cut3_surface(G, Gsurf, a, b, *, k_cut=3,
                             allow_last_hop_to_middle=True, middle_tie='right'):
    """
    Protect the corridor a↔b; cut up to k_cut hops on one non-corridor branch
    at each hub; route cut endpoints to the corridor 'mid' via surface-shortest
    paths that avoid walking along the corridor (except the last hop into 'mid').

    Removed edges are NOT blacklisted from the router; they can be reused.
    """
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError("Use networkx.Graph (simple, undirected), not MultiGraph.")

    a, b = int(a), int(b)
    if a == b or not (G.has_node(a) and G.has_node(b)):
        return None

    path = skeleton_shortest_path(G, a, b)  # hop-shortest on skeleton
    if len(path) < 2:
        return None

    mid = middle_on_path(path, tie=middle_tie)   # for d=1, 'left'→a, 'right'→b
    nb_a = path[1]            # corridor neighbor of a
    nb_b = path[-2]           # corridor neighbor of b

    # --- cut a non-corridor branch at each hub, collecting canonical removed edges
    def cut_branch(hub, corridor_neighbor):
        n1, chain = choose_branch_to_edit(G, hub, corridor_neighbor, k=k_cut)
        removed = []
        tip = None
        if chain and len(chain) >= 2:
            for i in range(len(chain) - 1):
                u, v = chain[i], chain[i+1]
                if _remove_edge_if_present(G, u, v):
                    removed.append(_canon_edge(u, v))
            tip = int(chain[-1])
        # de-dup in case chain had repeats
        if removed:
            removed = sorted(set(removed))
        return tip, removed, chain

    tipA, removedA, chainA = cut_branch(a, nb_a)
    tipB, removedB, chainB = cut_branch(b, nb_b)

    L = len(path) - 1
    print(f"[3HG-REWIRE] pair(a={a}, b={b}) L={L} path={path} mid={mid}")
    print(f"[3HG-REWIRE]   cut@a={a}: removed_edges={removedA} tip={tipA}")
    print(f"[3HG-REWIRE]   cut@b={b}: removed_edges={removedB} tip={tipB}")

    if tipA is None and tipB is None:
        return mid

    # Build router graph from surface; drop CURRENT skeleton edges (canonical),
    # allow last hop into mid, and allow using previously removed edges.
    # H = routing_graph_excluding_skeleton(
    #     Gsurf,
    #     skeleton_edges_set(G),
    #     allow_touch_dst=mid if allow_last_hop_to_middle else None
    # )


    print(f"mid = {mid}, tipA = {tipA}, tipB = {tipB}")
    H = Gsurf.copy()
    skeleton_nodes = set(G.nodes())

    keep = {mid}
    if tipA is not None: keep.add(tipA)
    if tipB is not None: keep.add(tipB)


    # --- debug membership BEFORE removing nodes ---
    print(f"[DEBUG] |V(H)|={H.number_of_nodes()}  |V(G)|={len(skeleton_nodes)}")
    print(f"[DEBUG] mid={mid} in H? {H.has_node(mid)}")

    if tipA is None:
        print("[DEBUG] tipA=None (no cut at a)")
    else:
        print(f"[DEBUG] tipA={tipA} in H? {H.has_node(tipA)}")

    if tipB is None:
        print("[DEBUG] tipB=None (no cut at b)")
    else:
        print(f"[DEBUG] tipB={tipB} in H? {H.has_node(tipB)}")

    missing = skeleton_nodes - set(H.nodes())
    print(f"[DEBUG] skeleton nodes missing from H: {len(missing)}")
    if missing:
        # print a small sample
        sample = list(missing)[:20]
        print(f"[DEBUG] sample missing nodes: {sample}")



    H.remove_nodes_from(skeleton_nodes - keep)

    corridor_nodes = set(path) - {mid}  # forbid walking along the corridor itself

    def route_and_splice(src, dst, label):
        if src is None:
            return []
        H2 = H.copy()
        # H2.remove_nodes_from(corridor_nodes)

        try:

            p = nx.shortest_path(H2, src, dst, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            print(f"[3HG-REWIRE]   route{label} NO-PATH src={src} -> dst={dst} (corridor nodes forbidden)")
            return []

        added = []
        for i in range(len(p) - 1):
            u, v = _canon_edge(p[i], p[i+1])
            if _add_edge_if_absent(G, u, v):
                added.append((u, v))
        if added:
            # print unique, canonical
            print(f"[3HG-REWIRE]   route{label} tip={src} -> mid={dst}: path={p} edges_added={added}")
        else:
            print(f"[3HG-REWIRE]   route{label} tip={src} -> mid={dst}: path={p} (no new edges needed)")
        return added

    route_and_splice(tipA, mid, "A")
    route_and_splice(tipB, mid, "B")
    return mid





def rewire_all_pairs(G, Gsurf, *, hop_cutoff=3, k_cut=3,
                     allow_last_hop_to_middle=True, middle_tie='right'):
    """Batch pass: for every 3hg pair within <= hop_cutoff, apply rewiring."""
    three_hg = {v for v in G.nodes() if G.degree(v) >= 3}
    pairs = []
    for u in sorted(three_hg):
        for v, d in nx.single_source_shortest_path_length(G, u, cutoff=hop_cutoff).items():
            if v in three_hg and v > u and d >= 1:
                pairs.append((u, v, d))
    pairs.sort(key=lambda t: (-t[2], t[0], t[1]))  # longer first
    for u, v, _ in pairs:
        if G.degree(u) < 3 or G.degree(v) < 3:  # may have changed
            continue
        print(f"[GRAPH] initial skeleton before rewire_pair_cut3_surface: |V|={G.number_of_nodes()} |E|={G.number_of_edges()}")
        rewire_pair_cut3_surface(G, Gsurf, u, v,
                                 k_cut=k_cut,
                                 allow_last_hop_to_middle=allow_last_hop_to_middle,
                                 middle_tie=middle_tie)



def three_hinge_set(G):
    """Nodes that are 3-hinge (degree ≥ 3)."""
    return {v for v in G.nodes() if G.degree(v) >= 3}

def find_close_3hg_pairs(G, max_hops=3, *, min_hops=1):
    """
    Return all unordered 3-hinge pairs (u<v) with hop distance d in [min_hops, max_hops].
    Distances are computed on the CURRENT skeleton G.
    """
    three_hg = three_hinge_set(G)
    pairs = []
    for u in sorted(three_hg):
        dmap = nx.single_source_shortest_path_length(G, u, cutoff=max_hops)
        for v, d in dmap.items():
            if v in three_hg and v > u and min_hops <= d <= max_hops:
                pairs.append((u, v, d))
    # longest first, deterministic tiebreak
    pairs.sort(key=lambda t: (-t[2], t[0], t[1]))
    return pairs

def summarize_close_3hg_pairs(pairs, *, print_examples=5):
    """
    Print a compact summary: total count, count per distance, a few examples,
    and the minimum distance between any 3-hinge pair if available.
    """
    if not pairs:
        print("[3HG-CHECK] No close 3-hinge pairs found in the requested hop range.")
        return

    by_d = defaultdict(int)
    for _, _, d in pairs:
        by_d[d] += 1

    mins = min(d for *_, d in pairs)
    print(f"[3HG-CHECK] Found {len(pairs)} close 3-hinge pairs; "
          f"min distance among them = {mins} hop(s).")
    for d in sorted(by_d):
        print(f"  - distance {d}: {by_d[d]} pair(s)")

    print(f"  Examples (up to {print_examples}):")
    for (u, v, d) in pairs[:print_examples]:
        print(f"    (u={u}, v={v}, d={d})")


def assert_no_close_3hg(G, *, max_hops=1):
    """
    Raise an AssertionError if ANY 3-hinge pair remains within <= max_hops.
    Useful as a hard guard in a pipeline.
    """
    pairs = find_close_3hg_pairs(G, max_hops=max_hops, min_hops=1)
    assert not pairs, (f"{len(pairs)} 3-hinge pairs remain within <={max_hops} hop(s); "
                       f"example: {pairs[0]}")

def count_close_3hg(G, *, max_hops=3, min_hops=1):
    return len(find_close_3hg_pairs(G, max_hops=max_hops, min_hops=min_hops))

def skeleton_edges_set(G):  # tiny helper
    return set(G.edges())

def clean_adjacent_3hg_pairs(G, Gsurf, *, tries_per_pair=3, target_deg=2, allow_reroute=True):
    pairs = find_close_3hg_pairs(G, max_hops=1, min_hops=1)  # d = 1 only
    for u, v, _ in pairs:
        # Demote by repeated branch cuts
        for _ in range(tries_per_pair):
            if not nx.has_path(G, u, v) or nx.shortest_path_length(G, u, v) != 1:
                break
            # stop once either endpoint is no longer a hinge
            if G.degree(u) <= target_deg or G.degree(v) <= target_deg:
                break
            rewire_pair_cut3_surface(
                G, Gsurf, u, v,
                k_cut=6,  # a bit more aggressive for d=1
                allow_last_hop_to_middle=True,
                middle_tie='right'
            )

        # If still adjacent hinge-hinge, try replacing edge with a longer surface path
        if (allow_reroute and
            nx.has_path(G, u, v) and
            nx.shortest_path_length(G, u, v) == 1 and
            G.degree(u) >= 3 and G.degree(v) >= 3):
            H = routing_graph_excluding_skeleton(Gsurf, skeleton_edges_set(G))
            removed = False
            if G.has_edge(u, v):
                G.remove_edge(u, v); removed = True
            try:
                p = nx.shortest_path(H, u, v, weight='weight')
                for i in range(len(p)-1):
                    G.add_edge(p[i], p[i+1])
            except nx.NetworkXNoPath:
                if removed:
                    G.add_edge(u, v)  # restore if no alternative

def report_close_3hg_pairs(G, *, close_k=3, min_hops=1, label=""):
    """
    Print all 3-hinge pairs (u<v) with hop distance d in [min_hops, close_k],
    plus a per-distance summary. Uses your existing find_close_3hg_pairs().
    """
    pairs = find_close_3hg_pairs(G, max_hops=close_k, min_hops=min_hops)  # (u,v,d)
    print(f"[3HG-REPORT] {label} | cutoff <={close_k} hop(s) | total={len(pairs)}")
    if not pairs:
        return pairs

    # Summary by distance
    dist_counts = Counter(d for _, _, d in pairs)
    by_d = ", ".join(f"d={d}:{dist_counts[d]}" for d in sorted(dist_counts))
    print(f"  by distance -> {by_d}")

    # Detailed list (sorted: longer first, then u,v)
    pairs.sort(key=lambda t: (-t[2], t[0], t[1]))
    print("u,v,d,deg_u,deg_v")
    for u, v, d in pairs:
        print(f"{u},{v},{d},{G.degree(u)},{G.degree(v)}")

    return pairs

def find_skelenton_missing(orig_sphere_polydata, orig_surf_polydata,
                           skeleton_polydata, curv_data_delete_thicknessZero,
                           original_sulc_data, length_thres_of_long_gyri,
                           neighbor_missing_path_smallest_step,
                           flat_threshold_for_convex_gyri, nearest_skeleton_num,
                           island_gyri_length_thres, output_prefix):
    print('================= build skeleton:\t' + time.asctime(time.localtime(time.time())) + ' =======================')
    final_edges = read_connection_of_skelenton_file(skeleton_polydata)
    final_adj = create_tree_connection_dict(final_edges)

    print('find missing gyri  \t' + time.asctime(time.localtime(time.time())))
    final_adj = create_connection_for_missing_gyri(
        orig_surf_polydata, orig_sphere_polydata, final_adj, original_sulc_data,
        curv_data_delete_thicknessZero, length_thres_of_long_gyri,
        neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri,
        nearest_skeleton_num, island_gyri_length_thres, output_prefix
    )  # <-- accept+return LoS

    all_edges = create_connectingPair(final_adj)

    print(f"all_edges count: {len(all_edges)}")
    # write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, all_edges, output_prefix + '_sphere_skelenton_allpoints_final.vtk')
    write_allPoints_and_skelenton_by_connectionPair(orig_surf_polydata,   all_edges, output_prefix + '_surf_skelenton_allpoints_final.vtk')

    all_point_lines_adj = create_tree_connection_dict(all_edges)

    # ---- NEW: edit via NetworkX and convert back ----
    coords = orig_sphere_polydata.GetPoints()  # optional for geometry
    G = los_to_nx(all_point_lines_adj, add_coords=coords)
    G.remove_nodes_from(list(nx.isolates(G)))
    # keep_largest_component_inplace(G)

    print(f"[GRAPH] initial skeleton: |V|={G.number_of_nodes()} |E|={G.number_of_edges()}")

    # >>> Do your edits on G here <<<
    # Example: keep only the largest connected component's edges
    # keep_largest_component_edges_inplace(G)
    # print(f"[GRAPH] initial skeleton after keep_largest_component_edges_inplace: |V|={G.number_of_nodes()} |E|={G.number_of_edges()}")

    
    # BEFORE
    CLOSE_K = 4  # define what "close" means for your check

    print(f"CLOSE_K {CLOSE_K}")


    # MULTI-PASS REWIRE
    # for _ in range(3):
    #     rewire_all_pairs(G, Gsurf, hop_cutoff=3, k_cut=5, allow_last_hop_to_middle=True, middle_tie='right')

    # # D=1 CLEAN-UP
    # clean_adjacent_3hg_pairs(G, Gsurf, tries_per_pair=3, target_deg=2, allow_reroute=True)

    # (You can insert your rewiring passes here and mutate G directly)
    Gsurf = build_surface_graph(orig_sphere_polydata)
    # MULTI-PASS REWIRE to cover the same notion of "close" as reported
    report_close_3hg_pairs(G, close_k=CLOSE_K, label="BEFORE")
    for _ in range(3):
        rewire_all_pairs(
            G, Gsurf,
            hop_cutoff=CLOSE_K,   # was 2; make it match your report
            k_cut=5,
            allow_last_hop_to_middle=True,
            middle_tie='right'
        )
        # AFTER

        # Convert back to LoS, preserving the full point-id space
        n_points = orig_sphere_polydata.GetNumberOfPoints()
        all_point_lines_adj = nx_to_los(G, size=n_points)        
        # (Optional) re-export the edited skeleton to VTK so you can visualize it
        all_edges = create_connectingPair(all_point_lines_adj)
    report_close_3hg_pairs(G, close_k=CLOSE_K, label="AFTER")



    _self = list(nx.selfloop_edges(G))
    if _self:
        print(f"[GRAPH-CHECK] self-loops: {_self}")

    remaining = find_close_3hg_pairs(G, max_hops=CLOSE_K, min_hops=1)
    if remaining:
        u, v, d = remaining[0]
        print(f"[ERROR][3HG-GUARD] Refined skeleton still has close 3HG pairs "
              f"(<= {CLOSE_K} hops). Example: (u={u}, v={v}, d={d}). "
              f"ABORT writing *_final_refine.vtk")

        pairs = find_close_3hg_pairs(G, max_hops=CLOSE_K, min_hops=1)
        pairs = pairs[:50]  # 先限量，避免一次動太多（你可用參數控制）

        ok = 0
        for (u, v, d) in pairs:
            if refine_close_3hg_pair_disjoint(
                G, Gsurf, u, v,
                branches_per_side=2,
                cut_hops=4,                 # 这里才是真正的 "cut 4 hops"
                cut_full_corridor=True,
                enforce_disjoint_routes=True,
                print_debug=True,           # 打印 mid/endpoints/routes/cuts
            ):
                ok += 1
        print("refined pairs ok =", ok)

        remaining_inside = find_close_3hg_pairs(G, max_hops=CLOSE_K, min_hops=1)

        if remaining_inside:
            print(f"[ERROR][3HG-GUARD] After refining, still {len(remaining_inside)} close 3HG pairs remain.")
            # Convert back to LoS, preserving the full point-id space
            n_points = orig_sphere_polydata.GetNumberOfPoints()
            all_point_lines_adj = nx_to_los(G, size=n_points)        
            # (Optional) re-export the edited skeleton to VTK so you can visualize it
            all_edges = create_connectingPair(all_point_lines_adj)
            # write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, all_edges, output_prefix + '_sphere_skelenton_allpoints_final_refine_failed.vtk')
            write_allPoints_and_skelenton_by_connectionPair(orig_surf_polydata,   all_edges, output_prefix + '_surf_skelenton_allpoints_final_refine_failed.vtk')
            print("[INFO][3HG-GUARD] After refining, no close 3HG pairs remain.")            
        else:
            # Convert back to LoS, preserving the full point-id space
            n_points = orig_sphere_polydata.GetNumberOfPoints()
            all_point_lines_adj = nx_to_los(G, size=n_points)        
            # (Optional) re-export the edited skeleton to VTK so you can visualize it
            all_edges = create_connectingPair(all_point_lines_adj)
            write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, all_edges, output_prefix + '_sphere_skelenton_allpoints_final_refine.vtk')
            write_allPoints_and_skelenton_by_connectionPair(orig_surf_polydata,   all_edges, output_prefix + '_surf_skelenton_allpoints_final_refine.vtk')
            print("[INFO][3HG-GUARD] After refining, no close 3HG pairs remain.")

    else: 
        # Convert back to LoS, preserving the full point-id space
        n_points = orig_sphere_polydata.GetNumberOfPoints()
        all_point_lines_adj = nx_to_los(G, size=n_points)        
        # (Optional) re-export the edited skeleton to VTK so you can visualize it
        all_edges = create_connectingPair(all_point_lines_adj)
        # write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, all_edges, output_prefix + '_sphere_skelenton_allpoints_final_refine.vtk')
        write_allPoints_and_skelenton_by_connectionPair(orig_surf_polydata,   all_edges, output_prefix + '_surf_skelenton_allpoints_final_refine.vtk')

    # write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, all_edges, output_prefix + '_sphere_skeleton_allpoints_post_nx.vtk')
    # write_allPoints_and_skelenton_by_connectionPair(orig_surf_polydata,   all_edges, output_prefix + '_surf_skeleton_allpoints_post_nx.vtk')
    # ---- END NEW ----

    print('find and draw 3hinge ' + time.asctime(time.localtime(time.time())))
    hinge3_list = create_3hinge(all_point_lines_adj)

    if len(hinge3_list) < 50:
        print(f"[ERROR][3HG-GUARD] Only {len(hinge3_list)} 3-hinge points found in the final skeleton; likely an error.")
    else:
        print(f"[3HG-GUARD] {len(hinge3_list)} 3-hinge points found in the final skeleton.")
        with open(output_prefix + '_3hinge_ids.txt', "w") as f:
            for h in hinge3_list:
                f.write(str(h) + "\n")
        # draw_3hinge_on_surf(orig_sphere_polydata, hinge3_list, output_prefix + '_sphere_3hinge_vertex.vtk')
        draw_3hinge_on_surf(orig_surf_polydata,   hinge3_list, output_prefix + '_surf_3hinge_vertex.vtk')


def read_connection_of_skelenton_file(skelenten_polydata):
    connected_lines_list = list()
    line_num = skelenten_polydata.GetNumberOfCells()
    for line in range(line_num):
        pointID1 = skelenten_polydata.GetCell(line).GetPointIds().GetId(0)
        pointID2 = skelenten_polydata.GetCell(line).GetPointIds().GetId(1)
        connected_lines_list.append([pointID1, pointID2])
    return connected_lines_list


def find_convex_marginal_points_of_long_gyri(orig_sphere_polydata, final_point_lines_dict, point_neighbor_points_dict, point_connect_points_dict_gyri_parts, marginal_points_gyri, original_sulc_data, curv_data_delete_thicknessZero, point_patchSize_dict_orig, length_thres_of_long_gyri, nearest_skeleton_num, island_gyri_length_thres):
    print('find convex marginal points of long gyri  \t' + time.asctime(time.localtime(time.time())))
    convex_marginal_points_gyri = list()
    for point in marginal_points_gyri:
        first_gyri_neighbor = [first_neighbor for first_neighbor in point_neighbor_points_dict[point] if original_sulc_data[first_neighbor] < 0]
        first_sulc_neighbor = [first_neighbor for first_neighbor in point_neighbor_points_dict[point] if original_sulc_data[first_neighbor] > 0]
        if (len(first_gyri_neighbor) - 2 < len(first_sulc_neighbor)) and curv_data_delete_thicknessZero[point] < 0:
            convex_marginal_points_gyri.append(point)
    convex_marginal_points_length_dict = defaultdict(list)
    long_convax_marginal_points_list = list()
    island_convex_marginal_points_length_dict = defaultdict(list)
    for convex_marginal_point in convex_marginal_points_gyri:
        from_marginal_to_skelenton_father_sons_dict = defaultdict(list)
        from_skelenton_to_marginal_son_fathers_dict = defaultdict(list)
        path = 0
        reach_skelenton = 0
        max_step_after_reach_skelenton = 6
        skelenton_point_set = set()
        current_outer_points_list = list()
        current_outer_points_list.append(convex_marginal_point)
        while max_step_after_reach_skelenton and (not reach_skelenton):
            next_outer_points_list = list()
            for point in current_outer_points_list:
                neighbors = [neighbor for neighbor in point_connect_points_dict_gyri_parts[point] if neighbor not in from_marginal_to_skelenton_father_sons_dict.keys() and neighbor not in current_outer_points_list and curv_data_delete_thicknessZero[neighbor] < 0]
                if point in from_marginal_to_skelenton_father_sons_dict.keys():
                    print('from_marginal_to_skelenton_father_sons_dict error!')
                else:
                    from_marginal_to_skelenton_father_sons_dict[point] = neighbors
                for neighbor in neighbors:
                    if neighbor not in from_skelenton_to_marginal_son_fathers_dict.keys():
                        from_skelenton_to_marginal_son_fathers_dict[neighbor] = [point]
                    else:
                        from_skelenton_to_marginal_son_fathers_dict[neighbor].append(point)

                next_outer_points_list = next_outer_points_list + neighbors

            if len(next_outer_points_list):

                path = path + 1

                # test single point
                # if convex_marginal_point == 71363:
                #     draw_3hinge_on_surf(orig_sphere_polydata, next_outer_points_list,
                #                         output_prefix + '_71363_path_' + str(path) + '.vtk')
                #     print(path)

                skelenton_point_set = skelenton_point_set | (set(next_outer_points_list) & set(final_point_lines_dict.keys()))
                if skelenton_point_set:
                    if max_step_after_reach_skelenton == 6:
                        real_path = path
                        averaged_patchSize = 0
                        for skelenton_point in skelenton_point_set:
                            averaged_patchSize = averaged_patchSize + point_patchSize_dict_orig[skelenton_point]
                        averaged_patchSize = averaged_patchSize / len(skelenton_point_set)
                    max_step_after_reach_skelenton -= 1
                if len(skelenton_point_set) >= nearest_skeleton_num:
                    reach_skelenton = 1
                    if real_path - averaged_patchSize > length_thres_of_long_gyri:
                        convex_marginal_points_length_dict[convex_marginal_point] = [real_path, from_marginal_to_skelenton_father_sons_dict, from_skelenton_to_marginal_son_fathers_dict]
                        long_convax_marginal_points_list.append(convex_marginal_point)
                else:
                    current_outer_points_list = set(next_outer_points_list)
            else:
                if path > island_gyri_length_thres:
                    island_convex_marginal_points_length_dict[convex_marginal_point] = [path, from_marginal_to_skelenton_father_sons_dict, from_skelenton_to_marginal_son_fathers_dict]
                break

    return long_convax_marginal_points_list, convex_marginal_points_length_dict, island_convex_marginal_points_length_dict


def select_convex_endpoint_for_each_missing_gyri(long_path_convex_points_list, convex_marginal_points_length_dict, curv_data_delete_thicknessZero, point_connect_points_dict_gyri_parts, neighbor_missing_path_smallest_step):
    print('select convex endpoint for each missing gyri \t' + time.asctime(time.localtime(time.time())))
    delete_list = list()
    for convex_point in long_path_convex_points_list:
        point_connection = [point for point in convex_marginal_points_length_dict[convex_point][1].keys() if point in long_path_convex_points_list]
        current_point = point_connection[0]
        max_path = convex_marginal_points_length_dict[current_point][0]
        step_n_neighbor_list, _ = find_round_n_neighbor(convex_point, neighbor_missing_path_smallest_step, point_connect_points_dict_gyri_parts)
        for point in point_connection:
            if point in step_n_neighbor_list:
                path = convex_marginal_points_length_dict[point][0]
                if path < max_path:
                    delete_list.append(point)
                elif path > max_path:
                    max_path = path
                    delete_list.append(current_point)
                    current_point = point
                else:
                    if curv_data_delete_thicknessZero[point] < curv_data_delete_thicknessZero[current_point]:
                        delete_list.append(current_point)
                        current_point = point
    delete_list = list(set(delete_list))
    for point in delete_list:
        long_path_convex_points_list.remove(point)
    return long_path_convex_points_list


def build_point_id_graphIndex_mapping(convex_point, from_skelenton_to_marginal_son_fathers_dict):
    index = 0
    pointId_graphIndex_dict = dict()
    graphIndex_pointId_dict = dict()
    pointId_graphIndex_dict[convex_point] = index
    graphIndex_pointId_dict[index] = convex_point
    for point in from_skelenton_to_marginal_son_fathers_dict.keys():
        index += 1
        pointId_graphIndex_dict[point] = index
        graphIndex_pointId_dict[index] = point
    return pointId_graphIndex_dict, graphIndex_pointId_dict


def build_graph_for_convex_point(convex_point, id_graphIndex_dict, from_marginal_to_skelenton_father_sons_dict, curv_data_delete_thicknessZero):
    """
    Build a sparse weighted graph (NetworkX) instead of an NxN dense matrix to avoid OOM.
    Nodes are graph indices; edges connect father-son pairs with symmetric weights.
    """
    G = nx.Graph()
    for father_point in from_marginal_to_skelenton_father_sons_dict.keys():
        for son_point in from_marginal_to_skelenton_father_sons_dict[father_point]:
            weight = 1 / (pow(curv_data_delete_thicknessZero[son_point], 2) + 0.000000001)
            u = id_graphIndex_dict[father_point]
            v = id_graphIndex_dict[son_point]
            G.add_edge(u, v, weight=weight)
    return G


def find_connections_for_long_convex_marginal_point(long_convex_marginal_point, final_point_lines_dict, point_connect_points_dict_gyri_parts, curv_data_delete_thicknessZero):
    from_marginal_to_skelenton_father_sons_dict = defaultdict(list)
    from_skelenton_to_marginal_son_fathers_dict = defaultdict(list)
    connection_dicts_list = list()
    reach_skelenton = 0
    skelenton_point_set = set()
    current_outer_points_list = list()
    current_outer_points_list.append(long_convex_marginal_point)
    while not reach_skelenton:
        next_outer_points_list = list()
        for point in current_outer_points_list:
            neighbors = [neighbor for neighbor in point_connect_points_dict_gyri_parts[point] if
                         neighbor not in from_marginal_to_skelenton_father_sons_dict.keys() and neighbor not in current_outer_points_list and
                         curv_data_delete_thicknessZero[neighbor] < 0]
            if point in from_marginal_to_skelenton_father_sons_dict.keys():
                print('from_marginal_to_skelenton_father_sons_dict error!')
            else:
                from_marginal_to_skelenton_father_sons_dict[point] = neighbors
            for neighbor in neighbors:
                if neighbor not in from_skelenton_to_marginal_son_fathers_dict.keys():
                    from_skelenton_to_marginal_son_fathers_dict[neighbor] = [point]
                else:
                    from_skelenton_to_marginal_son_fathers_dict[neighbor].append(point)

            next_outer_points_list = next_outer_points_list + neighbors

        if len(next_outer_points_list):
            skelenton_point_set = skelenton_point_set | (
                        set(next_outer_points_list) & set(final_point_lines_dict.keys()))
            if len(skelenton_point_set) >= 6:
                reach_skelenton = 1
                connection_dicts_list.append(from_marginal_to_skelenton_father_sons_dict)
                connection_dicts_list.append(from_skelenton_to_marginal_son_fathers_dict)

            else:
                current_outer_points_list = set(next_outer_points_list)
        else:
            break
    return connection_dicts_list


def find_connections_for_long_island_convex_marginal_point(long_convex_marginal_point, point_connect_points_dict_gyri_parts, curv_data_delete_thicknessZero):
    from_marginal_to_skelenton_father_sons_dict = defaultdict(list)
    from_skelenton_to_marginal_son_fathers_dict = defaultdict(list)
    connection_dicts_list = list()
    reach_end = 0
    current_outer_points_list = list()
    current_outer_points_list.append(long_convex_marginal_point)
    while not reach_end:
        next_outer_points_list = list()
        for point in current_outer_points_list:
            neighbors = [neighbor for neighbor in point_connect_points_dict_gyri_parts[point] if
                         neighbor not in from_marginal_to_skelenton_father_sons_dict.keys() and neighbor not in current_outer_points_list and
                         curv_data_delete_thicknessZero[neighbor] < 0]
            if point in from_marginal_to_skelenton_father_sons_dict.keys():
                print('from_marginal_to_skelenton_father_sons_dict error!')
            else:
                from_marginal_to_skelenton_father_sons_dict[point] = neighbors
            for neighbor in neighbors:
                if neighbor not in from_skelenton_to_marginal_son_fathers_dict.keys():
                    from_skelenton_to_marginal_son_fathers_dict[neighbor] = [point]
                else:
                    from_skelenton_to_marginal_son_fathers_dict[neighbor].append(point)

            next_outer_points_list = next_outer_points_list + neighbors

        if len(next_outer_points_list):
            current_outer_points_list = set(next_outer_points_list)
        else:
            connection_dicts_list.append(from_marginal_to_skelenton_father_sons_dict)
            connection_dicts_list.append(from_skelenton_to_marginal_son_fathers_dict)
            connection_dicts_list.append(current_outer_points_list)
            reach_end = 1
    return connection_dicts_list


def build_missing_connection_recurrently(long_path_convex_points_list_sorted, final_point_lines_dict, point_connect_points_dict_gyri_parts, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
    missing_connection_list = list()
    for long_convex_marginal_point in long_path_convex_points_list_sorted:
        connection_dicts_list = find_connections_for_long_convex_marginal_point(long_convex_marginal_point,
                                                                                final_point_lines_dict,
                                                                                point_connect_points_dict_gyri_parts,
                                                                                curv_data_delete_thicknessZero)
        if connection_dicts_list:
            from_marginal_to_skelenton_father_sons_dict = connection_dicts_list[0]
            from_skelenton_to_marginal_son_fathers_dict = connection_dicts_list[1]
            pointId_graphIndex_dict, graphIndex_pointId_dict = build_point_id_graphIndex_mapping(long_convex_marginal_point,
                                                                                                 from_skelenton_to_marginal_son_fathers_dict)
            graph = build_graph_for_convex_point(long_convex_marginal_point, pointId_graphIndex_dict,
                                                 from_marginal_to_skelenton_father_sons_dict,
                                                 curv_data_delete_thicknessZero)
            skelenton_point_list = [point for point in from_skelenton_to_marginal_son_fathers_dict.keys() if
                                    point in final_point_lines_dict.keys()]
            shortest_length = 100000000
            shortest_path = []
            for skelenton_point in skelenton_point_list:
                skelenton_graph_id = pointId_graphIndex_dict[skelenton_point]
                long_convex_marginal_point_id = pointId_graphIndex_dict[long_convex_marginal_point]
                path_list, length = create_shortest_path(graph, long_convex_marginal_point_id, skelenton_graph_id)
                if length < shortest_length:
                    shortest_path = path_list
                    shortest_length = length
            if shortest_path:
                if determine_real_long_path(shortest_path, graphIndex_pointId_dict, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
                    for i in range(len(shortest_path) - 1):
                        if [graphIndex_pointId_dict[shortest_path[i]], graphIndex_pointId_dict[shortest_path[i + 1]]] not in missing_connection_list and [graphIndex_pointId_dict[shortest_path[i + 1]], graphIndex_pointId_dict[shortest_path[i]]] not in missing_connection_list:
                            missing_connection_list.append([graphIndex_pointId_dict[shortest_path[i]], graphIndex_pointId_dict[shortest_path[i + 1]]])

                        if graphIndex_pointId_dict[shortest_path[i]] not in final_point_lines_dict.keys():
                            final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i]]] = {graphIndex_pointId_dict[shortest_path[i + 1]]}
                        else:
                            final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i]]].add(graphIndex_pointId_dict[shortest_path[i + 1]])
                        if graphIndex_pointId_dict[shortest_path[i + 1]] not in final_point_lines_dict.keys():
                            final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i + 1]]] = {graphIndex_pointId_dict[shortest_path[i]]}
                        else:
                            final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i + 1]]].add(graphIndex_pointId_dict[shortest_path[i]])
    return missing_connection_list, final_point_lines_dict


def build_main_skeleton_for_missing_island_gyri(longest_island_convex_point, final_point_lines_dict, point_connect_points_dict_gyri_parts, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
    print('build main skeleton for missing island \t' + time.asctime(time.localtime(time.time())))
    current_island_convex_point = longest_island_convex_point
    connection_dicts_list = find_connections_for_long_island_convex_marginal_point(current_island_convex_point,
                                                                                   point_connect_points_dict_gyri_parts,
                                                                                   curv_data_delete_thicknessZero)
    if connection_dicts_list:
        from_marginal_to_skelenton_father_sons_dict = connection_dicts_list[0]
        from_skelenton_to_marginal_son_fathers_dict = connection_dicts_list[1]
        pointId_graphIndex_dict, graphIndex_pointId_dict = build_point_id_graphIndex_mapping(current_island_convex_point,
                                                                                             from_skelenton_to_marginal_son_fathers_dict)
        graph = build_graph_for_convex_point(current_island_convex_point, pointId_graphIndex_dict,
                                             from_marginal_to_skelenton_father_sons_dict,
                                             curv_data_delete_thicknessZero)
        skelenton_point_list = connection_dicts_list[2]
        shortest_length = 100000000
        shortest_path = []
        for skelenton_point in skelenton_point_list:
            skelenton_graph_id = pointId_graphIndex_dict[skelenton_point]
            long_convex_marginal_point_id = pointId_graphIndex_dict[current_island_convex_point]
            path_list, length = create_shortest_path(graph, long_convex_marginal_point_id, skelenton_graph_id)
            if length < shortest_length:
                shortest_path = path_list
                shortest_length = length
        if determine_real_long_path(shortest_path, graphIndex_pointId_dict, point_patchSize_dict_orig,
                                    curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
            for i in range(len(shortest_path) - 1):
                if graphIndex_pointId_dict[shortest_path[i]] not in final_point_lines_dict.keys():
                    final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i]]] = {graphIndex_pointId_dict[shortest_path[i + 1]]}
                else:
                    final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i]]].add(graphIndex_pointId_dict[shortest_path[i + 1]])
                if graphIndex_pointId_dict[shortest_path[i + 1]] not in final_point_lines_dict.keys():
                    final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i + 1]]] = {graphIndex_pointId_dict[shortest_path[i]]}
                else:
                    final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i + 1]]].add(graphIndex_pointId_dict[shortest_path[i]])
    return final_point_lines_dict, connection_dicts_list


def determine_real_long_path_by_curv(path, graphIndex_pointId_dict, curv_data_delete_thicknessZero):
    curv_of_path = list()
    for index in path:
        point = graphIndex_pointId_dict[index]
        curv_of_path.append(curv_data_delete_thicknessZero[point])
    return curv_of_path


def determine_real_long_path(path, graphIndex_pointId_dict, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
    max_patchsize = 0
    for index in path:
        point = graphIndex_pointId_dict[index]
        if point_patchSize_dict_orig[point] > max_patchsize:
            max_patchsize = point_patchSize_dict_orig[point]
    if len(path) < 2 * max_patchsize:
        curv_of_path = determine_real_long_path_by_curv(path, graphIndex_pointId_dict, curv_data_delete_thicknessZero)
        curv_of_path_array = np.stack(curv_of_path)
        if np.sum(curv_of_path_array[0:6] > - 0.1) >= flat_threshold_for_convex_gyri:
            return False
        else:
            return True
    else:
        return True


def sort_the_long_path_convex_point(long_path_convex_points_list, convex_marginal_points_length_dict):
    long_path_convex_points_pathLength_dict = dict()
    for long_convex_point in long_path_convex_points_list:
        long_path_convex_points_pathLength_dict[long_convex_point] = convex_marginal_points_length_dict[long_convex_point][0]
    long_path_convex_points_pathLength_dict_sorted = sorted(long_path_convex_points_pathLength_dict.items(), key=lambda x: x[1], reverse=True)
    long_path_convex_points_list_sorted = [item[0] for item in long_path_convex_points_pathLength_dict_sorted]
    return long_path_convex_points_list_sorted



def _adj_los_to_dict_sets_nonempty(adj):
    """list[set] -> dict[int->set], dropping empty vertices (matches old dict behavior)."""
    d = {}
    for i, s in enumerate(adj):
        if s:
            d[i] = set(s)
    return d

def _dict_sets_to_adj_los(d, point_num):
    """dict[int->set/list] -> list[set] with length = point_num."""
    adj = [set() for _ in range(point_num)]
    for u, nbrs in d.items():
        su = adj[int(u)]
        for v in nbrs:
            su.add(int(v))
            adj[int(v)].add(int(u))
    return adj


def create_connection_for_missing_gyri(
    orig_surf_polydata,
    orig_sphere_polydata,
    final_point_lines_adj,             # <-- LoS input: list[set]
    original_sulc_data,
    curv_data_delete_thicknessZero,
    length_thres_of_long_gyri,
    neighbor_missing_path_smallest_step,
    flat_threshold_for_convex_gyri,
    nearest_skeleton_num,
    island_gyri_length_thres,
    output_prefix,
):
    """
    Accepts and returns LoS adjacency (list[set]).
    Internally converts to dict-of-sets only for helpers that still expect dicts.
    """

    # Mesh size for conversions back to LoS
    point_num = orig_sphere_polydata.GetNumberOfPoints()

    # Build graph helpers (these already return dict-like neighbor maps)
    point_connect_points_dict_gyri_parts = get_connect_points_gyri_part(orig_sphere_polydata, original_sulc_data)
    point_neighbor_points_dict = get_connect_points(orig_sphere_polydata)

    # Patch sizes (dict[int->int] as in your original)
    point_patchSize_dict_orig = find_the_patchSize_of_gyri_point(original_sulc_data, point_neighbor_points_dict)
    draw_patchSize_colorful_file(orig_sphere_polydata, orig_surf_polydata, point_patchSize_dict_orig, output_prefix, 'orig')

    # Marginals (expects dict neighbors + explicit key list like before)
    _, marginal_points_gyri, _ = find_marginal_point(
        list(point_neighbor_points_dict.keys()),
        point_neighbor_points_dict,
        original_sulc_data,
    )
    # draw_3hinge_on_surf(orig_sphere_polydata, marginal_points_gyri, output_prefix + '_sphere_marginal.vtk')
    draw_3hinge_on_surf(orig_surf_polydata,    marginal_points_gyri, output_prefix + '_surf_marginal.vtk')

    # Convert current LoS skeleton to dict form for helper functions
    final_point_lines_dict = _adj_los_to_dict_sets_nonempty(final_point_lines_adj)

    # 1) Find convex marginal points on long gyri paths
    all_long_path_convex_points_list, convex_marginal_points_length_dict, island_convex_marginal_points_length_dict = \
        find_convex_marginal_points_of_long_gyri(
            orig_sphere_polydata,
            final_point_lines_dict,                  # dict form
            point_neighbor_points_dict,
            point_connect_points_dict_gyri_parts,
            marginal_points_gyri,
            original_sulc_data,
            curv_data_delete_thicknessZero,
            point_patchSize_dict_orig,
            length_thres_of_long_gyri,
            nearest_skeleton_num,
            island_gyri_length_thres,
        )

    # draw_3hinge_on_surf(orig_sphere_polydata, all_long_path_convex_points_list, output_prefix + '_sphere_convex_marginal.vtk')
    draw_3hinge_on_surf(orig_surf_polydata,    all_long_path_convex_points_list, output_prefix + '_surf_convex_marginal.vtk')

    # 2) Choose endpoints and sort by path length
    long_path_convex_points_list = select_convex_endpoint_for_each_missing_gyri(
        all_long_path_convex_points_list,
        convex_marginal_points_length_dict,
        curv_data_delete_thicknessZero,
        point_connect_points_dict_gyri_parts,
        neighbor_missing_path_smallest_step,
    )
    # draw_3hinge_on_surf(orig_sphere_polydata, long_path_convex_points_list, output_prefix + '_sphere_convex_marginal_long.vtk')
    draw_3hinge_on_surf(orig_surf_polydata,    long_path_convex_points_list, output_prefix + '_surf_convex_marginal_long.vtk')

    long_path_convex_points_list_sorted = sort_the_long_path_convex_point(
        long_path_convex_points_list,
        convex_marginal_points_length_dict,
    )

    # 3) Build missing connections (recurrently). Returns: list of edges, and UPDATED dict skeleton.
    missing_connection_list, final_point_lines_dict = build_missing_connection_recurrently(
        long_path_convex_points_list_sorted,
        final_point_lines_dict,                    # dict form
        point_connect_points_dict_gyri_parts,
        point_patchSize_dict_orig,
        curv_data_delete_thicknessZero,
        flat_threshold_for_convex_gyri,
    )

    # Write newly added missing connections
    # write_skelenton_by_connectionPair(orig_sphere_polydata, missing_connection_list, output_prefix + '_sphere_missing_connection.vtk')
    write_skelenton_by_connectionPair(orig_surf_polydata,   missing_connection_list, output_prefix + '_surf_missing_connection.vtk')

    # Convert dict → LoS once here; keep LoS as the working structure going forward
    final_point_lines_adj = _dict_sets_to_adj_los(final_point_lines_dict, point_num)

    # ------------------ Handle island gyri ------------------

    long_path_island_convex_candicate_points_list = list(island_convex_marginal_points_length_dict.keys())

    while long_path_island_convex_candicate_points_list:
        long_path_island_convex_candicate_points_list_sorted = sort_the_long_path_convex_point(
            long_path_island_convex_candicate_points_list,
            island_convex_marginal_points_length_dict,
        )
        current_longest_island_convex_point = long_path_island_convex_candicate_points_list_sorted[0]

        # Convert to dict before calling helper
        final_point_lines_dict = _adj_los_to_dict_sets_nonempty(final_point_lines_adj)

        # This helper returns UPDATED dict skeleton + connection info dicts
        final_point_lines_dict, connection_dicts_list = build_main_skeleton_for_missing_island_gyri(
            current_longest_island_convex_point,
            final_point_lines_dict,                 # dict form
            point_connect_points_dict_gyri_parts,
            point_patchSize_dict_orig,
            curv_data_delete_thicknessZero,
            flat_threshold_for_convex_gyri,
        )

        # Back to LoS working structure
        final_point_lines_adj = _dict_sets_to_adj_los(final_point_lines_dict, point_num)

        from_marginal_to_skelenton_father_sons_dict = connection_dicts_list[0]
        from_skelenton_to_marginal_son_fathers_dict = connection_dicts_list[1]

        # Points that are still candidates AND not yet in the skeleton
        # (In LoS, "in skeleton" == degree > 0)
        island_convex_marginal_point_list = [
            p for p in from_skelenton_to_marginal_son_fathers_dict.keys()
            if p in island_convex_marginal_points_length_dict
            and len(final_point_lines_adj[p]) == 0
        ]

        # Re-scan for islands with the updated skeleton (helpers expect dict)
        final_point_lines_dict = _adj_los_to_dict_sets_nonempty(final_point_lines_adj)

        all_long_path_island_convex_points_list, convex_marginal_island_points_length_dict, _ = \
            find_convex_marginal_points_of_long_gyri(
                orig_sphere_polydata,
                final_point_lines_dict,              # dict form
                point_neighbor_points_dict,
                point_connect_points_dict_gyri_parts,
                island_convex_marginal_point_list,
                original_sulc_data,
                curv_data_delete_thicknessZero,
                point_patchSize_dict_orig,
                length_thres_of_long_gyri,
                nearest_skeleton_num,
                island_gyri_length_thres,
            )

        long_path_island_convex_points_list = select_convex_endpoint_for_each_missing_gyri(
            all_long_path_island_convex_points_list,
            convex_marginal_island_points_length_dict,
            curv_data_delete_thicknessZero,
            point_connect_points_dict_gyri_parts,
            neighbor_missing_path_smallest_step,
        )

        long_path_island_convex_points_list_sorted = sort_the_long_path_convex_point(
            long_path_island_convex_points_list,
            convex_marginal_island_points_length_dict,
        )

        # Build the remaining connections for islands; update dict skeleton
        _, final_point_lines_dict = build_missing_connection_recurrently(
            long_path_island_convex_points_list_sorted,
            final_point_lines_dict,                 # dict form
            point_connect_points_dict_gyri_parts,
            point_patchSize_dict_orig,
            curv_data_delete_thicknessZero,
            flat_threshold_for_convex_gyri,
        )

        # Back to LoS for the loop condition/updates
        final_point_lines_adj = _dict_sets_to_adj_los(final_point_lines_dict, point_num)

        # Remove handled candidates (present in either connection dict)
        finished = [
            p for p in island_convex_marginal_points_length_dict.keys()
            if p in from_marginal_to_skelenton_father_sons_dict
            or p in from_skelenton_to_marginal_son_fathers_dict
        ]
        if finished:
            sfin = set(finished)
            long_path_island_convex_candicate_points_list = [
                p for p in long_path_island_convex_candicate_points_list
                if p not in sfin
            ]
    return final_point_lines_adj


def find_the_max_sulc_point(points_list, original_sulc_data):
    max = -0
    for point in points_list:
        if original_sulc_data[point] < max:
            max = original_sulc_data[point]
            max_sulc_point = point
    return max_sulc_point


def read_thin_sphere_surf(file):
    thin_sphere_polydata = read_vtk_file(file)
    point_num = thin_sphere_polydata.GetNumberOfPoints()
    thin_sulc_data = list()
    for point in range(point_num):
        sulc = thin_sphere_polydata.GetPointData().GetArray('sulc_updated').GetTuple(point)[0]
        thin_sulc_data.append(sulc)
    thin_sulc_data = np.stack(thin_sulc_data)
    return thin_sphere_polydata, thin_sulc_data


def create_tree_connection_dict(connected_lines_list):
    """
    Faster: return list-of-sets adjacency instead of dict.
    (Callers in this file now consume LoS; external outputs remain unchanged.)
    """
    # infer point count from edges
    point_num = 0 if not connected_lines_list else max(max(u, v) for u, v in connected_lines_list) + 1
    return _edges_to_adj_list_of_sets(connected_lines_list, point_num)


def step_n_father(point, n, father_dict):
    point_father_list = list()
    current_father = father_dict[point]
    for step in range(n):
        point_father_list.append(current_father)
        current_father = father_dict[current_father]
    return point_father_list


def get_connection_degree_of_step_n_father(start_point, end_father, father_dict, point_lines_dict):
    connection_degrees = list()
    father = father_dict[start_point]
    while father != end_father:
        degree = len(point_lines_dict[father])
        connection_degrees.append(degree)
        father = father_dict[father]
    degree = len(point_lines_dict[father])
    connection_degrees.append(degree)
    return connection_degrees


def connect_break_in_circle(point_lines_adj,
                            point_connect_points_dict_thin_gyri_parts,
                            deleted_round1_points,
                            father_dict,
                            verbose=False):
    n = len(point_lines_adj)
    father = _to_int_array(father_dict, n)
    nbrs = _to_neighbors_list(point_connect_points_dict_thin_gyri_parts, n)

    # membership mask for deleted_round1_points
    del_mask = np.zeros(n, dtype=bool)
    for d in deleted_round1_points:
        del_mask[int(d)] = True

    for p in range(n):
        if len(point_lines_adj[p]) != 1:
            continue

        found = False

        # Case 1: direct 1-ring endpoint
        for nb in nbrs[p]:
            if len(point_lines_adj[nb]) == 1:
                if not set(step_n_father(p, 10, father)) & set(step_n_father(nb, 10, father)):
                    point_lines_adj[p].add(int(nb))
                    point_lines_adj[int(nb)].add(p)
                    found = True
                    break
                elif verbose:
                    pass  # same-branch; skip noisy prints

        if found:
            continue

        # Case 2: via deleted round1 neighbor
        for nb in nbrs[p]:
            if not del_mask[nb]:
                continue
            for nb2 in nbrs[nb]:
                if len(point_lines_adj[nb2]) == 1 and nb2 != p:
                    if not set(step_n_father(p, 10, father)) & set(step_n_father(nb2, 10, father)):
                        point_lines_adj[p].add(int(nb))
                        point_lines_adj[int(nb)] = {p}
                        point_lines_adj[int(nb)].add(int(nb2))
                        point_lines_adj[int(nb2)].add(int(nb))
                        found = True
                        break
            if found:
                break

        if found:
            continue

        # Case 3: connect via father’s 1-ring endpoint
        f = father[p]
        if f != -1:
            for fn in nbrs[f]:
                if fn != p and len(point_lines_adj[fn]) == 1:
                    if not set(step_n_father(f, 10, father)) & set(step_n_father(fn, 10, father)):
                        point_lines_adj[f].add(int(fn))
                        point_lines_adj[int(fn)].add(int(f))
                        # remove p--f
                        if f in point_lines_adj[p]: point_lines_adj[p].remove(int(f))
                        if p in point_lines_adj[f]: point_lines_adj[f].remove(int(p))
                        found = True
                        break
            if found:
                continue

        # Case 4: 2-step via a non-skeleton neighbor then to a 1-degree vertex
        forbid = {p, father[p]} | ({father[father[p]]} if father[p] != -1 and father[father[p]] != -1 else set())
        pairs = []
        for nb in nbrs[p]:
            if nb in forbid:
                continue
            if len(point_lines_adj[nb]) > 0:
                # neighbor already in skeleton; skip (original printed a note)
                continue
            for nb2 in nbrs[nb]:
                if nb2 in forbid or nb2 == p:
                    continue
                if len(point_lines_adj[nb2]) == 1:
                    pairs.append((int(nb), int(nb2)))

        for nb, nb2 in pairs:
            if not set(step_n_father(p, 10, father)) & set(step_n_father(nb2, 10, father)):
                point_lines_adj[p].add(nb)
                point_lines_adj[nb] = {p}
                point_lines_adj[nb].add(nb2)
                point_lines_adj[nb2].add(nb)
                found = True
                break

        if found:
            continue

        # Case 5: two deleted-round1 steps to a 1-degree vertex
        d1 = [nb for nb in nbrs[p] if del_mask[nb]]
        for nb1 in d1:
            d2 = [nb2 for nb2 in nbrs[nb1] if del_mask[nb2] and nb2 != nb1]
            for nb2 in d2:
                for nb3 in nbrs[nb2]:
                    if len(point_lines_adj[nb3]) == 1:
                        if not set(step_n_father(p, 10, father)) & set(step_n_father(nb3, 10, father)):
                            point_lines_adj[p].add(int(nb1))
                            point_lines_adj[int(nb1)].add(p)
                            point_lines_adj[int(nb1)].add(int(nb2))
                            point_lines_adj[int(nb2)].add(int(nb1))
                            point_lines_adj[int(nb2)].add(int(nb3))
                            point_lines_adj[int(nb3)].add(int(nb2))
                            found = True
                            break
                if found: break
            if found: break

        # If still not found, we silently continue (no spammy prints)
    return point_lines_adj



def clear_empty_point_in_dict(point_lines_adj):
    """
    Convert LoS -> defaultdict(set) but only include non-empty vertices (old behavior).
    """
    out = defaultdict(set)
    for u, nbrs in enumerate(point_lines_adj):
        if nbrs:
            out[u] = set(nbrs)
    return out


def trim_allRound1_long(point_lines_adj,
                        point_patchSize_dict_updated,
                        point_connect_points_dict_thin_gyri_parts,
                        father_dict,
                        inner_gyri_points):

    n_skel = len(point_lines_adj)

    # Arrays that auto-grow to cover their own max keys
    patch  = _to_float_array(point_patchSize_dict_updated, n_skel)
    father = _to_int_array(father_dict, n_skel)

    # Compute the true max index we may touch: skeleton, father, and neighbors
    max_f = int(father[father >= 0].max()) if np.any(father >= 0) else -1
    max_nbr = _max_index_in_neighbors(point_connect_points_dict_thin_gyri_parts)
    point_num_full = max(n_skel, max_f + 1, max_nbr + 1)

    # Build neighbors once with safe size
    nbrs = _to_neighbors_list(point_connect_points_dict_thin_gyri_parts, point_num_full)

    # Build inner mask (accepts bool mask or list of ids)
    if isinstance(inner_gyri_points, np.ndarray) and inner_gyri_points.dtype == bool:
        inner_mask = np.zeros(point_num_full, dtype=bool)
        # copy (truncate/pad) to point_num_full
        upto = min(point_num_full, inner_gyri_points.size)
        inner_mask[:upto] = inner_gyri_points[:upto]
    else:
        inner_mask = np.zeros(point_num_full, dtype=bool)
        for i in inner_gyri_points:
            inner_mask[int(i)] = True

    deleted_round1_points = []

    # ---- delete round-1 short branches ----
    for p in range(n_skel):
        nbr_list = list(point_lines_adj[p])
        if not nbr_list:
            continue
        degs = [len(point_lines_adj[q]) for q in nbr_list]
        long_edges = sum(d > 1 for d in degs)

        short_best = None
        short_best_ps = -np.inf
        for q, d in zip(nbr_list, degs):
            if d == 1:
                ps = patch[q]
                if ps > short_best_ps:
                    short_best_ps = ps
                    short_best = q

        current_deleted = []
        if long_edges < len(nbr_list):
            for q, d in zip(nbr_list, degs):
                if d == 1 and q != short_best:
                    if q in point_lines_adj[p]:
                        point_lines_adj[p].remove(q)
                    if p in point_lines_adj[q]:
                        point_lines_adj[q].remove(p)
                    current_deleted.append(q)
            if long_edges > 1 and short_best is not None:
                if short_best in point_lines_adj[p]:
                    point_lines_adj[p].remove(short_best)
                if p in point_lines_adj[short_best]:
                    point_lines_adj[short_best].remove(p)
                current_deleted.append(short_best)

        if long_edges == 1 and current_deleted:
            deleted_round1_points.extend(current_deleted)

    if deleted_round1_points:
        deleted_round1_points = list(set(deleted_round1_points))

    # ---- delete long branches ----
    endpoints = [u for u in range(n_skel) if len(point_lines_adj[u]) == 1]

    for endpoint in endpoints:
        step_n_father_list = step_n_father(endpoint, 6, father)

        reference_points = []
        # first ring
        for ref in nbrs[endpoint]:
            if ref < n_skel and len(point_lines_adj[ref]) >= 2:
                ok = (ref not in step_n_father_list
                      or (ref in step_n_father_list and
                          max(get_connection_degree_of_step_n_father(endpoint, ref, father, point_lines_adj)) > 2))
                if ok:
                    reference_points.append(int(ref))

        if not reference_points:
            # via inner gyri
            for inner_nb in nbrs[endpoint]:
                if not inner_mask[inner_nb]:
                    continue
                for ref in nbrs[inner_nb]:
                    if ref < n_skel and len(point_lines_adj[ref]) >= 2:
                        ok = (ref not in step_n_father_list
                              or (ref in step_n_father_list and
                                  max(get_connection_degree_of_step_n_father(endpoint, ref, father, point_lines_adj)) > 2))
                        if ok:
                            reference_points.append(int(ref))
            reference_points = list(set(reference_points))

        # same-branch check
        same_branch = False
        if reference_points:
            f_ep = father[endpoint]
            f_f_ep = father[f_ep] if f_ep != -1 else -1
            for ref in reference_points:
                fr = father[ref]
                cond = False
                if f_ep != -1 and f_ep < point_num_full:
                    cond |= (fr in nbrs[f_ep])
                if f_f_ep != -1 and f_f_ep < point_num_full:
                    cond |= (fr in nbrs[f_f_ep])
                if cond:
                    same_branch = True
                    break
                ep_chain = set(step_n_father(endpoint, 12, father))
                ref_chain = set(step_n_father(ref, 12, father))
                if ep_chain & ref_chain:
                    same_branch = True
                    break

        if same_branch:
            cur = endpoint
            while len(point_lines_adj[cur]) == 1:
                nxt = next(iter(point_lines_adj[cur]))
                point_lines_adj[cur].remove(nxt)
                point_lines_adj[nxt].remove(cur)
                cur = nxt

    return point_lines_adj, deleted_round1_points


def create_connectingPair(point_lines_adj):
    """
    Build a de-duplicated undirected edge list from list-of-sets adjacency.
    Return type unchanged: List[List[int]] (pairs).
    """
    return _adj_list_of_sets_to_edges(point_lines_adj)


def create_3hinge(point_lines_adj):
    """
    Return all vertices with degree >= 3. Return type unchanged: List[int].
    """
    return [i for i, nbrs in enumerate(point_lines_adj) if len(nbrs) >= 3]


def draw_3hinge_on_surf(surf_polydata, hinge3_list, output_3hinge_vertex):
    points_new = vtk.vtkPoints()
    vertices_new = vtk.vtkCellArray()
    vertex_num = 0
    for point in hinge3_list:
        coordinate = surf_polydata.GetPoints().GetPoint(point)
        points_new.InsertNextPoint(coordinate)
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, vertex_num)
        vertices_new.InsertNextCell(vertex)
        vertex_num += 1
    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points_new)
    polygonPolyData.SetVerts(vertices_new)
    polygonPolyData.Modified()

    if vtk.VTK_MAJOR_VERSION <= 5:
        polygonPolyData = polygonPolyData.GetProducerPort()
    else:
        polygonPolyData = polygonPolyData

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polygonPolyData)
    writer.SetFileName(output_3hinge_vertex)
    writer.Write()


def create_shortest_path(adj_or_graph, start, end):
    """
    Compute Dijkstra shortest path given either a NetworkX Graph or a dense adjacency matrix.
    Prefers Graph to avoid converting large matrices.
    """
    if isinstance(adj_or_graph, nx.Graph):
        G = adj_or_graph
    else:
        G = nx.Graph(adj_or_graph)
    r_list_numpy_index = nx.dijkstra_path(G, source=start, target=end, weight='weight')
    length = nx.dijkstra_path_length(G, source=start, target=end, weight='weight')
    return r_list_numpy_index, length


def draw_patchSize_colorful_file(*args):
    """
    Compatible wrapper.

    New call:
        draw_patchSize_colorful_file(surface_polydata, point_patchSize_dict, output_prefix, type)

    Old call:
        draw_patchSize_colorful_file(surface_polydata, surf_polydata, point_patchSize_dict, output_prefix, type)

    In no-sphere mode, we only write the surface output.
    """
    if len(args) == 4:
        surface_polydata, point_patchSize_dict, output_prefix, type_name = args

    elif len(args) == 5:
        # Old signature:
        # sphere_polydata, surf_polydata, point_patchSize_dict, output_prefix, type
        # In no-sphere mode, use surf_polydata as the output geometry.
        _, surface_polydata, point_patchSize_dict, output_prefix, type_name = args

    else:
        raise TypeError(
            "draw_patchSize_colorful_file() expects either 4 or 5 positional arguments: "
            "(surface_polydata, point_patchSize_dict, output_prefix, type) or "
            "(sphere_polydata, surf_polydata, point_patchSize_dict, output_prefix, type)"
        )

    patchSize_data = []

    for point in range(surface_polydata.GetNumberOfPoints()):
        patchSize_data.append(point_patchSize_dict[point])

    write_featured_sphere_from_variable_single(
        surface_polydata,
        type_name,
        patchSize_data,
        output_prefix + '_surf_patchSize_' + type_name + '.vtk'
    )

def process_subject(subject, args):
    root = args.root_dir
    hemi_list = args.hemi_list

    out_dir = root + '/' + subject + '/' + args.out_dir

    if args.force and os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    timings = defaultdict(float)
    t_total_start = time.perf_counter()

    log_file = setup_logging(out_dir, name=f"{subject}_log")
    print(f"[LOG] writing all output to {log_file}")

    for hemi in hemi_list:
        print(f"out_dir: {out_dir}")
        print(f"hemi: {hemi}")

        surface_file = root + '/' + subject + '/' + args.input_dir + '/' + hemi + args.surf_file
        output_prefix = out_dir + '/' + hemi

        print(f"surface_file: {surface_file}")

        if not os.path.exists(surface_file):
            raise FileNotFoundError(f"Surface VTK file not found: {surface_file}")

        t0 = time.perf_counter()
        surface_polydata = read_vtk_file(surface_file)
        timings['read_mesh'] += time.perf_counter() - t0

        point_num = surface_polydata.GetNumberOfPoints()
        print(f"Number of points: {point_num}")
        print(f"Number of cells: {surface_polydata.GetNumberOfCells()}")

        required_arrays = ['thickness', 'curv', 'sulc']
        for arr_name in required_arrays:
            arr = surface_polydata.GetPointData().GetArray(arr_name)
            if arr is None:
                raise ValueError(
                    f"Required point data array '{arr_name}' not found in {surface_file}. "
                    f"Please make sure the VTK contains thickness, curv, and sulc arrays."
                )

            if arr.GetNumberOfTuples() != point_num:
                raise ValueError(
                    f"Array length mismatch for '{arr_name}' in {surface_file}: "
                    f"{arr.GetNumberOfTuples()} values, but mesh has {point_num} points."
                )

        t0 = time.perf_counter()

        write_featured_surface_from_surface(
            surface_polydata,
            surface_polydata,
            output_prefix + '_surface_features.vtk'
        )

        timings['export_features'] += time.perf_counter() - t0

        print('create points connection dict:\t' + time.asctime(time.localtime(time.time())))

        t0 = time.perf_counter()

        point_neighbor_points_dict = get_connect_points(surface_polydata)

        timings['build_adjacency'] += time.perf_counter() - t0

        t0 = time.perf_counter()

        updated_sulc_data, original_sulc_data, curv_data_delete_thicknessZero = initialize_sulc_data(
            surface_polydata,
            surface_polydata,
            point_neighbor_points_dict,
            args.inner_sulci_curv_thres,
            args.inner_sulci_round_thres,
            args.outer_gyri_curv_thres,
            args.inner_sulci_neighbor_curv_thres,
            output_prefix
        )

        timings['initialize_sulc'] += time.perf_counter() - t0

        t0 = time.perf_counter()

        thin_sulc_data = write_thin_gyri_on_sphere_point_marginal_sulc_curv(
            surface_polydata,
            surface_polydata,
            point_neighbor_points_dict,
            updated_sulc_data,
            curv_data_delete_thicknessZero,
            args.expend_curv_step_size,
            output_prefix
        )

        timings['thin_gyri'] += time.perf_counter() - t0

        t0 = time.perf_counter()

        point_patchSize_dict_orig = find_the_patchSize_of_gyri_point(
            original_sulc_data,
            point_neighbor_points_dict
        )

        point_patchSize_dict_updated = find_the_patchSize_of_gyri_point(
            thin_sulc_data,
            point_neighbor_points_dict
        )

        timings['patch_size'] += time.perf_counter() - t0

        t0 = time.perf_counter()

        draw_patchSize_colorful_file(
            surface_polydata,
            point_patchSize_dict_orig,
            output_prefix,
            'orig'
        )

        draw_patchSize_colorful_file(
            surface_polydata,
            point_patchSize_dict_updated,
            output_prefix,
            'updated'
        )

        timings['draw_patch_size'] += time.perf_counter() - t0

        t0 = time.perf_counter()

        point_connect_points_dict_thin_gyri_parts = get_connect_points_gyri_part(
            surface_polydata,
            thin_sulc_data
        )

        timings['thin_gyri_adjacency'] += time.perf_counter() - t0

        t0 = time.perf_counter()

        connected_lines_list, father_dict = create_tree(
            surface_polydata,
            surface_polydata,
            point_patchSize_dict_updated,
            point_connect_points_dict_thin_gyri_parts,
            thin_sulc_data,
            output_prefix
        )

        timings['create_tree'] += time.perf_counter() - t0

        t0 = time.perf_counter()

        find_skelenton(
            surface_polydata,
            surface_polydata,
            point_patchSize_dict_updated,
            curv_data_delete_thicknessZero,
            original_sulc_data,
            thin_sulc_data,
            point_neighbor_points_dict,
            point_connect_points_dict_thin_gyri_parts,
            connected_lines_list,
            father_dict,
            args.length_thres_of_long_gyri,
            args.neighbor_missing_path_smallest_step,
            args.flat_threshold_for_convex_gyri,
            args.nearest_skeleton_num,
            args.island_gyri_length_thres,
            output_prefix
        )

        timings['build_skeleton'] += time.perf_counter() - t0

        t0 = time.perf_counter()

        skeleton_file = output_prefix + '_surf_skelenton_connect_break_allpoints.vtk'
        skeleton_polydata = read_vtk_file(skeleton_file)

        timings['read_skeleton_polydata'] += time.perf_counter() - t0

        t0 = time.perf_counter()

        find_skelenton_missing(
            surface_polydata,
            surface_polydata,
            skeleton_polydata,
            curv_data_delete_thicknessZero,
            original_sulc_data,
            args.length_thres_of_long_gyri,
            args.neighbor_missing_path_smallest_step,
            args.flat_threshold_for_convex_gyri,
            args.nearest_skeleton_num,
            args.island_gyri_length_thres,
            output_prefix
        )

        timings['connect_missing'] += time.perf_counter() - t0

    timings['total'] = time.perf_counter() - t_total_start

    try:
        timing_file = os.path.join(out_dir, 'timings.txt')
        with open(timing_file, 'w') as f:
            for k in sorted(timings.keys()):
                f.write(f"{k}\t{timings[k]:.3f}\n")

        print('Timing written to', timing_file)
        print('\n'.join([f"{k}: {timings[k]:.3f}s" for k in sorted(timings.keys())]))

    except Exception as e:
        print('Failed to write timing file:', e)


def main(args):
    root = args.root_dir

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    subjects_list = [
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]

    # Single-subject mode overrides range filtering
    if getattr(args, 'subject_id', None):
        subjects_list = [str(args.subject_id)] if os.path.isdir(os.path.join(root, str(args.subject_id))) else []
        print(os.path.isdir(os.path.join(root, str(args.subject_id))))
        print(os.path.join(root, str(args.subject_id)))
        print(f"Single-subject mode: processing subject {args.subject_id}")
        print("Subjects to process:", subjects_list)
    # Single-subject mode overrides everything
    if args.subject_id:
        subjects_list = [args.subject_id] if os.path.isdir(os.path.join(root, args.subject_id)) else []
        print(f"Single-subject mode: processing subject {args.subject_id}")
        print("Subjects to process:", subjects_list)

    subjects_list.sort()
    # print(subjects_list)
    # print("root dir:", root)
    # print(f"args.subject_id: {args.subject_id}")
    if getattr(args, 'subject_id', None):
        for subject in subjects_list:
            process_subject(subject, args)
    elif getattr(args, 'num_workers', 1) and args.num_workers > 1:
        print('Running in parallel with workers:', args.num_workers)
        global EXECUTOR
        EXECUTOR = ProcessPoolExecutor(max_workers=args.num_workers)
        try:
            with EXECUTOR as executor:
                futures = [executor.submit(process_subject, subject, args) for subject in subjects_list]
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        print('Subject failed with exception:', e)
        except KeyboardInterrupt:
            print('KeyboardInterrupt received in main, cancelling futures...')
            _terminate_children_and_exit()
        finally:
            EXECUTOR = None
    else:
        for subject in subjects_list:
            process_subject(subject, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GyralNet creation by expending algorithm")
    parser.add_argument('-root_dir', '--root_dir', type=str, default='/work/users/m/i/minhengc/HCP_cerebellum', help='root for input data')

    parser.add_argument('-subject_list_start_id', '--subject_list_start_id', type=int, default=100206, help='subjects list start and end ids')
    parser.add_argument('-subject_list_end_id', '--subject_list_end_id', type=int, default=9967782, help='subjects list start and end ids')

    parser.add_argument('-input_dir', '--input_dir', type=str, default='surf', help='input dir within each subject')
    parser.add_argument('-out_dir', '--out_dir', type=str, default='gyralnet_output', help='out dir within each subject')
    parser.add_argument(
        '-hemi_list',
        '--hemi_list',
        nargs='+',
        default=['lh', 'rh'],
        help='hemispheres to process, e.g., lh rh'
    )

    parser.add_argument(
        '-surf_file',
        '--surf_file',
        type=str,
        default='.white.vtk',
        help='surface VTK file name suffix, e.g., .white.vtk'
    )
    # The following feature files are now read from surface point data instead of external files.
    parser.add_argument('-curv_file', '--curv_file', type=str, default='.curv', help='curv_file name')
    parser.add_argument('-sulc_file', '--sulc_file', type=str, default='.sulc', help='sulc_file name')
    parser.add_argument('-thickness_file', '--thickness_file', type=str, default='.thickness', help='thickness_file name')

    parser.add_argument('-inner_sulci_curv_thres', '--inner_sulci_curv_thres', type=float, default=0.06, help='the curv threshold to identify sulci in the gyri part')
    parser.add_argument('-inner_sulci_neighbor_curv_thres', '--inner_sulci_neighbor_curv_thres', type=float, default=0.006, help='the curv threshold to change the neighbor to sulci')
    parser.add_argument('-inner_sulci_round_thres', '--inner_sulci_round_thres', type=int, default=6, help='the patch size threshold to identify sulci in the gyri part')
    parser.add_argument('-outer_gyri_curv_thres', '--outer_gyri_curv_thres', type=float, default=-0.03, help='the curv threshold to identify gyri in the sulci part')
    parser.add_argument('-expend_curv_step_size', '--expend_curv_step_size', type=float, default=0.02, help='during expending, the edges with small curv will be erosen first')
    parser.add_argument('-length_thres_of_long_gyri', '--length_thres_of_long_gyri', type=int, default=6, help='length threshold to select missing long gyri')
    parser.add_argument('-neighbor_missing_path_smallest_step', '--neighbor_missing_path_smallest_step', type=int, default=15, help='length threshold to select missing long gyri')
    parser.add_argument('-flat_threshold_for_convex_gyri', '--flat_threshold_for_convex_gyri', type=int, default=4, help='to evaluate if the long path represent a convex gyri')
    parser.add_argument('-nearest_skeleton_num', '--nearest_skeleton_num', type=int, default=6, help='the number of nearest skeleton points to each marginal convex point')
    parser.add_argument('-island_gyri_length_thres', '--island_gyri_length_thres', type=int, default=20, help='the lower bound the the length of island gyri to be connected')
    parser.add_argument('-num_workers', '--num_workers', type=int, default=20, help='number of parallel workers over subjects')
    parser.add_argument('-subject_id', '--subject_id', type=str, default='', help='run a single subject id (overrides range and parallel)')
    parser.add_argument('-f', '--force', action='store_true', help='force remove and recreate per-subject output directory')

    args = parser.parse_args()
    print(args)
    main(args)