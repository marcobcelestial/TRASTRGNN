import os
import numpy as np
import torch
from ansys.mapdl.core import launch_mapdl

# ─────────────────────────────────────────────
# CONFIGURATION — tune these two values first
# ─────────────────────────────────────────────
ELEMENT_SIZE   = 50      # mm — increase if still over license limit
SMRTSIZE_LEVEL = 8        # 1=finest, 10=coarsest; start at 8 for limited licenses
LICENSE_LIMIT  = 32000    # set to your actual license cap (32k / 512k / unlimited)
step_file_path = r"D:\WU15\TRASTRGNN\Data\Version 1\Structure-1.igs"

# ─────────────────────────────────────────────
# HELPER — safe get() that never crashes on error strings
# ─────────────────────────────────────────────
def safe_get(mapdl, *args):
    try:
        val = mapdl.get(*args)
        return float(val)
    except Exception:
        return 0.0

# ─────────────────────────────────────────────
# 1. LAUNCH
# ─────────────────────────────────────────────
print("Launching Ansys MAPDL...")
mapdl = launch_mapdl()
mapdl.clear()

# ─────────────────────────────────────────────
# 2. IMPORT GEOMETRY
# ─────────────────────────────────────────────
print("Importing geometry...")
mapdl.aux15()
mapdl.igesin(step_file_path)
mapdl.finish()

# ─────────────────────────────────────────────
# 3. GEOMETRY CLEANUP
# ─────────────────────────────────────────────
mapdl.prep7()
print("Cleaning geometry...")

try:
    mapdl.vglue("ALL")
    print("  VGLUE OK")
except Exception as e:
    print(f"  VGLUE failed ({e}), trying VOVLAP...")
    try:
        mapdl.vovlap("ALL")
        print("  VOVLAP OK")
    except Exception as e2:
        print(f"  VOVLAP also failed ({e2}). Using raw geometry.")

mapdl.nummrg("KP", 0.001)
mapdl.numcmp("ALL")

# ─────────────────────────────────────────────
# 4. MATERIAL & ELEMENT TYPE
# ─────────────────────────────────────────────
mapdl.et(1, 285)
mapdl.mp("EX",   1, 200000)
mapdl.mp("PRXY", 1, 0.3)
mapdl.mp("DENS", 1, 7.85e-9)

# ─────────────────────────────────────────────
# 5. MESH — coarse enough to stay under license limit
# ─────────────────────────────────────────────
print(f"Meshing with ESIZE={ELEMENT_SIZE}...")

# 1. THE FIX: Kill SmartSize and Shape Checking
mapdl.run("SHPP, OFF")      # Force Ansys to ignore ugly element shapes at the joints
mapdl.run("SMRTSIZE, OFF")  # Stop Ansys from auto-refining the mesh at overlaps

mapdl.mshape(1, "3D")
mapdl.mshkey(0)

# 2. Now ESIZE is the absolute law
mapdl.esize(ELEMENT_SIZE)

try:
    mapdl.vmesh("ALL")
    print("  VMESH OK")
except Exception as e:
    raise RuntimeError(
        f"Meshing failed: {e}\n"
        f"→ Try increasing ELEMENT_SIZE (currently {ELEMENT_SIZE})"
    )
# ─────────────────────────────────────────────
# 6. ELEMENT COUNT GATE
# ─────────────────────────────────────────────
n_elem = int(safe_get(mapdl, "NUM_ELEM", "ELEM", 0, "COUNT"))
n_node = int(safe_get(mapdl, "NUM_NODE", "NODE", 0, "COUNT"))
print(f"Mesh stats → Elements: {n_elem}  |  Nodes: {n_node}")

if n_elem == 0:
    raise RuntimeError(
        "Mesh is empty after VMESH. "
        "Check geometry import — volumes may not have been created."
    )

if n_elem > LICENSE_LIMIT:
    raise RuntimeError(
        f"Element count {n_elem} exceeds license limit {LICENSE_LIMIT}.\n"
        f"→ Increase ELEMENT_SIZE above {ELEMENT_SIZE} and re-run.\n"
        f"   Rule of thumb: doubling ELEMENT_SIZE cuts element count by ~8x."
    )

print(f"  Element count {n_elem} is within license limit ({LICENSE_LIMIT}).")

# ─────────────────────────────────────────────
# 7. POST-MESH NODE MERGE
# ─────────────────────────────────────────────
mapdl.nummrg("NODE", 0.001)
mapdl.numcmp("NODE")

# ─────────────────────────────────────────────
# 8. BOUNDARY CONDITIONS
# ─────────────────────────────────────────────
print("Applying boundary conditions...")
mapdl.run("/SOLU")
mapdl.antype("TRANS")
mapdl.time(0.1)
mapdl.nsubst(10)
mapdl.outres("ALL", "ALL")

# Fixed base
mapdl.nsel("S", "LOC", "Y", -5, 5)
n_fixed = int(safe_get(mapdl, "N_FIXED", "NODE", 0, "COUNT"))
print(f"  Fixed BC nodes: {n_fixed}")
if n_fixed == 0:
    print("  WARNING: No nodes in Y=[-5,5]. Widening to Y=[-50,50]...")
    mapdl.nsel("S", "LOC", "Y", -50, 50)
mapdl.d("ALL", "ALL")
mapdl.allsel()

# Load at top
max_y = safe_get(mapdl, "MAX_Y", "NODE", 0, "MXLOC", "Y")
print(f"  Top face Y = {max_y:.2f} mm")
mapdl.nsel("S", "LOC", "Y", max_y - 10, max_y + 10)
n_loaded = int(safe_get(mapdl, "N_LOADED", "NODE", 0, "COUNT"))
print(f"  Loaded nodes: {n_loaded}")
if n_loaded == 0:
    print("  WARNING: No nodes at top face. Widening tolerance to ±50 mm...")
    mapdl.nsel("S", "LOC", "Y", max_y - 50, max_y + 50)
mapdl.f("ALL", "FX", 5000)
mapdl.allsel()

# ─────────────────────────────────────────────
# 9. SOLVE
# ─────────────────────────────────────────────
print("Solving transient time-response...")
try:
    mapdl.solve()
    print("  SOLVE OK")
except Exception as e:
    raise RuntimeError(f"Solver crashed during execution: {e}")

# ─────────────────────────────────────────────
# 10. GNN DATA EXTRACTION (/POST1)
# ─────────────────────────────────────────────
print("Extracting physical data for GNN...")

mapdl.post1()

# A. Extract Graph Nodes (Vertices)
# Returns an array of node coordinates
nodes = mapdl.mesh.nodes 

# B. Extract Graph Connectivity (Edges)
# Returns the raw element table showing which nodes make up which elements
elements = mapdl.mesh.elem 

# C. Extract Time-Series Labels (Ground Truth)
time_series = []
for step in range(1, 11):  # Loop through your 10 substeps (0.01s to 0.1s)
    mapdl.set(1, step)
    
    # Extract the X, Y, Z displacement for every node at this specific time step
    disp = mapdl.post_processing.nodal_displacement("ALL")
    time_series.append(disp)



# ─────────────────────────────────────────────
# 11. PYTORCH DATASET PACKAGING & CLEANUP
# ─────────────────────────────────────────────
print("Packaging into PyTorch tensors...")

# Convert the Ansys arrays into PyTorch tensors
# Using float32 for positions/displacements and int64 (long) for indices
dataset_item = {
    "pos": torch.tensor(nodes, dtype=torch.float32),          # Shape: [N, 3]
    "elements": torch.tensor(elements, dtype=torch.long),     # Raw connectivity
    "y": torch.tensor(np.array(time_series), dtype=torch.float32) # Shape: [10, N, 3]
}

# Save the .pt file directly next to the original .igs file
out_file = step_file_path.replace(".igs", ".pt")
torch.save(dataset_item, out_file)
print(f"✅ Success! GNN dataset saved to: {out_file}")



# CRITICAL: Always exit MAPDL at the end of the script to free up 
# your RAM and the gRPC port for the next structure in your batch.
mapdl.exit()
