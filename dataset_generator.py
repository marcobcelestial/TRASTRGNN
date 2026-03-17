import os
import glob
import traceback

import numpy as np
import torch
import gmsh
import pyvista as pv
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# CONFIGURATION

# PATH
STEP_SOURCE = r"D:\WU15\TRASTRGNN\Data\Version 2"   

MAX_FILES     = 1      # Max structures to process per run (None = ALL)
SKIP_EXISTING = False    # Skip structures whose .pt output already exists

GMSH_CHAR_LEN_MIN = 40      # mm — minimum element characteristic length
GMSH_CHAR_LEN_MAX = 80     # mm — maximum element characteristic length
GMSH_VERBOSE      = True   # True = show Gmsh terminal output

MAX_NODES = 3_000_000     # Rule of thumb: 20k nodes ≈ 2–4 GB RAM per solve

TOTAL_TIME    = 5.0     # seconds — total simulation duration
N_SUBSTEPS    = 50     # number of equal time steps to extract
NEWMARK_BETA  = 0.25    # 0.25 = constant-average acceleration (stable)
NEWMARK_GAMMA = 0.5     # 0.5  = no numerical damping

GRAVITY_Y    = -9810.0  # mm/s²  (use -9.81 for m-models)
BODY_FORCE_X =  0.0     # optional lateral body force mm/s² (0 = gravity only)

MAT_EX   = 200_000.0    # MPa — Young's modulus
MAT_PRXY = 0.3          # Poisson's ratio
MAT_DENS = 7.85e-9      # tonne/mm³


SHOW_MESH        = True    # PyVista mesh preview after Gmsh meshing
SHOW_TIME_SERIES = True    # Plot displacement magnitude at each time step

def find_step_files(folder):
    files = []
    for pat in ["*.step", "*.stp", "*.STEP", "*.STP"]:
        files.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(set(files))

def output_path(step_file):
    return step_file.rsplit(".", 1)[0] + ".pt"

def tmp_msh_path(step_file):
    return step_file.rsplit(".", 1)[0] + "_tmp.msh"

def cleanup_tmp(msh_file):
    try:
        if os.path.exists(msh_file):
            os.remove(msh_file)
    except Exception:
        pass

def mesh_with_gmsh(step_file, msh_file):
    print(f"  [Gmsh] Meshing: {os.path.basename(step_file)}")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", int(GMSH_VERBOSE))
        gmsh.model.add("structure")
        gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", GMSH_CHAR_LEN_MIN)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", GMSH_CHAR_LEN_MAX)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        # Physical groups — needed to read node/element data back out
        volumes  = gmsh.model.getEntities(dim=3)
        surfaces = gmsh.model.getEntities(dim=2)
        if not volumes:
            raise RuntimeError("No 3-D volumes found in STEP file.")

        gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes],  tag=1)
        gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=2)

        gmsh.model.mesh.generate(3)
        gmsh.write(msh_file)

        if SHOW_MESH:
            print("  [Gmsh] Mesh viewer — close window to continue...")
            gmsh.fltk.run()

        print(f"  [Gmsh] Written: {os.path.basename(msh_file)}")
        return True

    except Exception as e:
        print(f"  [Gmsh] ERROR: {e}")
        traceback.print_exc()
        return False
    finally:
        gmsh.finalize()

def read_msh(msh_file):
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(msh_file)

        # Node coordinates
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = np.array(coords, dtype=np.float64).reshape(-1, 3)

        # Build a tag → 0-based index map
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        # Tet4 elements (element type 4 in Gmsh)
        _, _, node_tags_elem = gmsh.model.mesh.getElements(dim=3)
        if len(node_tags_elem) == 0 or len(node_tags_elem[0]) == 0:
            raise RuntimeError("No 3-D elements found in mesh.")

        raw = np.array(node_tags_elem[0], dtype=np.int64).reshape(-1, 4)
        # Map global tags to 0-based indices
        elements = np.vectorize(tag_to_idx.get)(raw).astype(np.int64)

        return nodes, elements

    finally:
        gmsh.finalize()

# Tet4 shape-function gradient (constant per element)

def tet4_shape_grad(coords):
    """
    coords : [4, 3] node coordinates of one tet element
    Returns:
        B    : [6, 12] strain-displacement matrix
        vol  : scalar element volume
    """
    x = coords
    # Edge vectors from node 0
    J = (x[1:] - x[0]).T                       # [3, 3] Jacobian
    vol = np.linalg.det(J) / 6.0
    if vol <= 0:
        vol = abs(vol)                          # tolerate inverted elements

    inv_J = np.linalg.inv(J)

    # Shape function gradients w.r.t. global coords
    dN_loc = np.array([[-1, -1, -1],
                       [ 1,  0,  0],
                       [ 0,  1,  0],
                       [ 0,  0,  1]], dtype=np.float64)   # [4, 3]
    dN = dN_loc @ inv_J.T                                  # [4, 3]

    # Build B matrix [6, 12]
    B = np.zeros((6, 12))
    for i in range(4):
        col = i * 3
        B[0, col]   = dN[i, 0]
        B[1, col+1] = dN[i, 1]
        B[2, col+2] = dN[i, 2]
        B[3, col]   = dN[i, 1];  B[3, col+1] = dN[i, 0]
        B[4, col+1] = dN[i, 2];  B[4, col+2] = dN[i, 1]
        B[5, col]   = dN[i, 2];  B[5, col+2] = dN[i, 0]

    return B, vol


def build_elasticity_matrix(E, nu):
    """3-D isotropic linear elastic constitutive matrix [6, 6]."""
    c = E / ((1 + nu) * (1 - 2 * nu))
    a = 1 - nu
    b = nu
    s = (1 - 2 * nu) / 2
    return c * np.array([
        [a, b, b, 0, 0, 0],
        [b, a, b, 0, 0, 0],
        [b, b, a, 0, 0, 0],
        [0, 0, 0, s, 0, 0],
        [0, 0, 0, 0, s, 0],
        [0, 0, 0, 0, 0, s],
    ])


# Global assembly

def assemble_KM(nodes, elements, E, nu, rho):
    """
    Assemble global stiffness K and consistent mass matrix M.
    Both are returned as scipy csr_matrix, shape [3N, 3N].
    """
    n_nodes = len(nodes)
    n_dof   = 3 * n_nodes
    n_elem  = len(elements)

    D = build_elasticity_matrix(E, nu)

    # Lumped mass per node (simpler + more stable for explicit-like Newmark)
    mass_vec = np.zeros(n_dof)

    K = lil_matrix((n_dof, n_dof))

    print(f"  [FEM]  Assembling K & M for {n_elem} elements, {n_nodes} nodes...")

    for e_idx, conn in enumerate(elements):
        if e_idx % max(1, n_elem // 10) == 0:
            pct = 100 * e_idx // n_elem
            print(f"         {pct}% ...", end="\r")

        coords = nodes[conn]               # [4, 3]
        B, vol = tet4_shape_grad(coords)
        Ke     = vol * (B.T @ D @ B)      # [12, 12]

        # Element mass (lumped: quarter of total element mass to each node)
        elem_mass = rho * vol / 4.0

        # Scatter into global K and mass_vec
        dofs = np.array([conn[i]*3 + j for i in range(4) for j in range(3)])
        for li, gi in enumerate(dofs):
            for lj, gj in enumerate(dofs):
                K[gi, gj] += Ke[li, lj]
            mass_vec[gi] += elem_mass

    print(f"         100% — assembly done.        ")

    K = K.tocsr()

    # Diagonal mass matrix as sparse
    M = csr_matrix((mass_vec, (np.arange(n_dof), np.arange(n_dof))),
                   shape=(n_dof, n_dof))

    return K, M


# Apply fixed-base Dirichlet BC (penalty method)

def apply_fixed_base(K, M, nodes, tol_frac=0.02):
    """
    Zero out rows/cols for DOFs at Y ≈ Y_min (penalty elimination).
    Returns modified K_bc, M_bc, and the list of fixed DOF indices.
    """
    y_min  = nodes[:, 1].min()
    tol    = max(abs(y_min) * tol_frac, 1.0)
    fixed_nodes = np.where(nodes[:, 1] <= y_min + tol)[0]
    fixed_dofs  = np.array([n*3 + d for n in fixed_nodes for d in range(3)])
    print(f"  [FEM]  Fixed BC: {len(fixed_nodes)} nodes, {len(fixed_dofs)} DOFs at Y ≤ {y_min+tol:.2f}")

    # Convert to LIL for efficient row/col ops
    K_bc = K.tolil()
    M_bc = M.tolil()

    PENALTY = 1e30
    for dof in fixed_dofs:
        K_bc[dof, :] = 0
        K_bc[:, dof] = 0
        K_bc[dof, dof] = PENALTY
        M_bc[dof, :] = 0
        M_bc[:, dof] = 0
        M_bc[dof, dof] = 1.0   # neutral mass — avoids singular M

    return K_bc.tocsr(), M_bc.tocsr(), fixed_dofs


# Body force vector 

def build_force_vector(nodes, elements, rho, gx, gy):
    """Consistent nodal body force from gravity (and optional lateral load)."""
    n_dof = 3 * len(nodes)
    f = np.zeros(n_dof)
    for conn in elements:
        coords   = nodes[conn]
        _, vol   = tet4_shape_grad(coords)
        node_f   = rho * vol / 4.0          # quarter of element body force per node
        for ni in conn:
            f[ni*3 + 0] += node_f * gx
            f[ni*3 + 1] += node_f * gy
    return f


# Newmark-β transient solve

def newmark_solve(K, M, f, beta, gamma, dt, n_steps):
    """
    Newmark-β time integration.
    Returns list of displacement arrays, one per step: each shape [n_dof].
    """
    n_dof = K.shape[0]

    # Effective stiffness:  K_eff = M/(β dt²) + K
    K_eff = M / (beta * dt**2) + K

    u = np.zeros(n_dof)
    v = np.zeros(n_dof)
    a = np.zeros(n_dof)   # initial acceleration from static equilibrium

    # Initial acceleration:  M a0 = f - K u0
    rhs0 = f - K @ u
    # Simple diagonal-mass solve
    M_diag = M.diagonal()
    M_diag_safe = np.where(M_diag > 0, M_diag, 1.0)
    a = rhs0 / M_diag_safe

    displacements = []

    print(f"  [FEM]  Newmark-β solve: {n_steps} steps, dt = {dt:.5f} s")
    for step in range(1, n_steps + 1):
        # Predictor
        u_pred = u + dt * v + dt**2 * (0.5 - beta) * a
        v_pred = v + dt * (1.0 - gamma) * a

        # Effective RHS
        rhs = f + M @ u_pred / (beta * dt**2)

        # Solve
        u_new = spsolve(K_eff, rhs)

        # Corrector
        a_new = (u_new - u_pred) / (beta * dt**2)
        v_new = v_pred + dt * gamma * a_new

        u, v, a = u_new, v_new, a_new

        disp_3d = u.reshape(-1, 3)
        max_d   = np.linalg.norm(disp_3d, axis=1).max()
        print(f"         step {step:02d}/{n_steps}  |u|_max = {max_d:.4e} mm")
        displacements.append(disp_3d.copy())

    return displacements

#Von Mises for visualization

def compute_von_mises(nodes, elements, displacements_flat, E, nu):
    """
    Compute von Mises stress per NODE by averaging from surrounding elements.
    Returns array of shape [N_nodes] for one time step.
    """
    D = build_elasticity_matrix(E, nu)
    n_nodes = len(nodes)
    
    stress_sum   = np.zeros((n_nodes, 6))  # 6 stress components per node
    stress_count = np.zeros(n_nodes)

    for conn in elements:
        coords   = nodes[conn]
        B, vol   = tet4_shape_grad(coords)

        # Element displacement vector [12]
        u_elem = displacements_flat[conn].flatten()

        # Stress vector [6]: σ = D · B · u
        strain = B @ u_elem
        stress = D @ strain

        # Scatter to nodes (average over contributing elements)
        for ni in conn:
            stress_sum[ni]   += stress
            stress_count[ni] += 1

    # Avoid divide by zero
    stress_count = np.where(stress_count > 0, stress_count, 1)
    avg_stress = stress_sum / stress_count[:, None]

    # Von Mises from 6-component stress [sx,sy,sz,sxy,syz,sxz]
    sx  = avg_stress[:, 0]
    sy  = avg_stress[:, 1]
    sz  = avg_stress[:, 2]
    sxy = avg_stress[:, 3]
    syz = avg_stress[:, 4]
    sxz = avg_stress[:, 5]

    von_mises = np.sqrt(0.5 * ((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2
                               + 6*(sxy**2 + syz**2 + sxz**2)))
    return von_mises

def run_fem_pipeline(msh_file):
    try:
        # Load mesh
        nodes, elements = read_msh(msh_file)
        n_nodes = len(nodes)
        n_elem  = len(elements)
        print(f"  [FEM]  Nodes: {n_nodes}   Elements: {n_elem}")

        if MAX_NODES and n_nodes > MAX_NODES:
            raise RuntimeError(
                f"Node count {n_nodes} > MAX_NODES={MAX_NODES}. "
                f"Increase GMSH_CHAR_LEN_MIN/MAX or raise MAX_NODES."
            )

        if SHOW_MESH:
            print("  [FEM]  Mesh viewer — close window to continue...")
            grid = pv.UnstructuredGrid(
                {pv.CellType.TETRA: elements},
                nodes
            )
            pl = pv.Plotter(title="Mesh Preview")
            pl.add_mesh(grid, show_edges=True, color="lightblue")
            pl.add_axes()
            pl.show()

        # Assemble
        K, M = assemble_KM(nodes, elements, MAT_EX, MAT_PRXY, MAT_DENS)

        # Apply BCs
        K_bc, M_bc, _ = apply_fixed_base(K, M, nodes)

        # Body force
        f = build_force_vector(nodes, elements, MAT_DENS, BODY_FORCE_X, GRAVITY_Y)

        # Solve
        dt = TOTAL_TIME / N_SUBSTEPS
        time_series = newmark_solve(K_bc, M_bc, f, NEWMARK_BETA, NEWMARK_GAMMA, dt, N_SUBSTEPS)

        if SHOW_TIME_SERIES:
            for step, disp in enumerate(time_series, 1):

                # Auto-scale warp
                bbox_size      = nodes.max(axis=0) - nodes.min(axis=0)
                structure_size = np.linalg.norm(bbox_size)
                max_disp       = np.linalg.norm(disp, axis=1).max()
                WARP_SCALE     = (structure_size * 0.05) / max_disp if max_disp > 0 else 1.0

                warped_nodes = nodes + disp * WARP_SCALE

                # Compute von Mises stress
                von_mises = compute_von_mises(nodes, elements, disp, MAT_EX, MAT_PRXY)
                print(f"    Step {step} | von Mises: min={von_mises.min():.2e}  max={von_mises.max():.2e} MPa")

                grid = pv.UnstructuredGrid({pv.CellType.TETRA: elements}, warped_nodes)
                grid.point_data["von_mises"] = von_mises

                pl = pv.Plotter(title=f"Von Mises Stress — step {step}/{N_SUBSTEPS}")
                pl.add_mesh(
                    grid,
                    scalars="von_mises",
                    cmap="jet",
                    show_edges=False,
                    clim=[np.percentile(von_mises, 5),   # ignore bottom 5% outliers
                        np.percentile(von_mises, 95)]   # ignore top 5% outliers
                )
                pl.add_scalar_bar("Von Mises [MPa]", fmt="%.2e")
                pl.show()

        dataset = {
            "pos":      torch.tensor(nodes,                          dtype=torch.float32),
            "elements": torch.tensor(elements,                       dtype=torch.long),
            "y":        torch.tensor(np.array(time_series),          dtype=torch.float32),
        }
        # Shapes: pos[N,3]  elements[C,4]  y[T,N,3]
        return dataset

    except Exception as e:
        print(f"  [FEM]  ERROR: {e}")
        traceback.print_exc()
        return None


# MAIN LOOP

def main():
    print(" TRASTRGNN DATASET GENERATOR (SciPy sparse FEM backend)")
    print(f" Source folder   : {STEP_SOURCE}")
    print(f" Mesh size       : {GMSH_CHAR_LEN_MIN} – {GMSH_CHAR_LEN_MAX} mm")
    print(f" Substeps        : {N_SUBSTEPS}  over  {TOTAL_TIME} s")
    print(f" Gravity Y       : {GRAVITY_Y} mm/s²")
    print(f" Max files       : {MAX_FILES if MAX_FILES else 'ALL'}")
    print(f" Skip existing   : {SKIP_EXISTING}")

    all_files = find_step_files(STEP_SOURCE)
    if not all_files:
        print(f"[ERROR] No .step/.stp files found in:\n  {STEP_SOURCE}")
        return

    print(f"Found {len(all_files)} STEP file(s).")

    if SKIP_EXISTING:
        pending = [f for f in all_files if not os.path.exists(output_path(f))]
        skipped = len(all_files) - len(pending)
        if skipped:
            print(f"Skipping {skipped} already-processed file(s).")
    else:
        pending = list(all_files)

    if MAX_FILES is not None:
        pending = pending[:MAX_FILES]

    print(f"Processing {len(pending)} file(s) this run.\n")

    results = {"ok": 0, "failed": 0, "files": []}

    for idx, step_file in enumerate(pending, 1):
        name     = os.path.basename(step_file)
        msh_file = tmp_msh_path(step_file)
        out_file = output_path(step_file)

        print(f"[{idx}/{len(pending)}] {name}")

        try:
            if not mesh_with_gmsh(step_file, msh_file):
                raise RuntimeError("Gmsh meshing failed.")

            dataset = run_fem_pipeline(msh_file)
            if dataset is None:
                raise RuntimeError("FEM pipeline returned no data.")

            torch.save(dataset, out_file)
            print(f"Saved: {out_file}")
            print(f"pos      : {tuple(dataset['pos'].shape)}")
            print(f"elements : {tuple(dataset['elements'].shape)}")
            print(f"y: {tuple(dataset['y'].shape)}")
            results["ok"] += 1
            results["files"].append({"file": name, "status": "OK",
                                     "nodes": dataset["pos"].shape[0]})

        except Exception as e:
            print(f"FAILED: {e}")
            results["failed"] += 1
            results["files"].append({"file": name, "status": f"FAILED: {e}"})

        finally:
            cleanup_tmp(msh_file)

    print(" SUMMARY")
    print(f"Successful : {results['ok']}")
    print(f"Failed     : {results['failed']}")

if __name__ == "__main__":
    main()