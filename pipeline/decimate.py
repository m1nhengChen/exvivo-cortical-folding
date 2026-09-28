#!/usr/bin/env python3
"""Decimate a mesh to a target face count via pymeshlab quadric edge collapse.

Reads a VTK PolyData, collapses to `target` faces (boundary, normal, and topology
preserved), and writes the result back out.
"""
import sys, numpy as np, vtk, pymeshlab
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

src, dst, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = vtk.vtkPolyDataReader(); r.SetFileName(src); r.Update()
pd = r.GetOutput()
v = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
f = vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1, 4)[:, 1:]

ms = pymeshlab.MeshSet()
ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f))
ms.meshing_decimation_quadric_edge_collapse(
    targetfacenum=target, preserveboundary=True, preservenormal=True,
    preservetopology=True, planarquadric=True)
m = ms.mesh(0)
v2, f2 = m.vertex_matrix(), m.face_matrix()

out = vtk.vtkPolyData()
pts = vtk.vtkPoints(); pts.SetData(numpy_to_vtk(np.ascontiguousarray(v2)))
out.SetPoints(pts)
ca = vtk.vtkCellArray()
ca.SetCells(f2.shape[0], numpy_to_vtk(np.insert(f2, 0, 3, axis=1).ravel(),
                                      array_type=vtk.VTK_ID_TYPE))
out.SetPolys(ca)
w = vtk.vtkPolyDataWriter(); w.SetFileName(dst); w.SetInputData(out); w.SetFileVersion(42)
w.Update()
print(f"  {len(v)} 顶点 / {len(f)} 面  ->  {len(v2)} 顶点 / {len(f2)} 面")
