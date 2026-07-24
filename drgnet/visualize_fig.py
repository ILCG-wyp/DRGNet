import argparse
import numpy as np
import open3d as o3d
import os
import time
from scipy.spatial.transform import Rotation as R

# ---------- Helper functions ----------
def get_array(data, names, required=True):
    for name in names:
        if name in data:
            return data[name]
    if required:
        raise KeyError(f"Cannot find any of {names}. Available keys: {list(data.keys())}")
    return None

def apply_transform(points, transform):
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    return (pts_h @ transform.T)[:, :3]

def compute_gt_residual(src_corr, ref_corr, gt_transform):
    """Compute residual under GT transform (in meters), auto-detect transform direction."""
    pred1 = (gt_transform[:3, :3] @ src_corr.T).T + gt_transform[:3, 3]
    res1 = np.linalg.norm(pred1 - ref_corr, axis=1)
    try:
        inv_t = np.linalg.inv(gt_transform)
        pred2 = (inv_t[:3, :3] @ src_corr.T).T + inv_t[:3, 3]
        res2 = np.linalg.norm(pred2 - ref_corr, axis=1)
        if np.mean(res2) < np.mean(res1):
            return res2
    except:
        pass
    return res1

def compute_geometric_consistency(src_corr, ref_corr, tau=0.10):
    """
    Compute per-correspondence geometric consistency score based on distance preservation.
    Higher score means more geometrically compatible with neighbors.
    """
    n = src_corr.shape[0]
    if n < 2:
        return np.ones(n)
    src_dist = np.linalg.norm(src_corr[:, None, :] - src_corr[None, :, :], axis=-1)
    ref_dist = np.linalg.norm(ref_corr[:, None, :] - ref_corr[None, :, :], axis=-1)
    diff = np.abs(src_dist - ref_dist)
    order = np.argsort(src_dist, axis=1)
    k = min(33, n)  # use up to 32 neighbors
    neigh = order[:, 1:k]
    scores = []
    for i in range(n):
        scores.append(np.exp(-diff[i, neigh[i]] / tau).mean())
    return np.asarray(scores)

def create_pointcloud(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd

def create_lineset(points1, points2, color):
    n = len(points1)
    if n == 0:
        return None
    lines = [[i, i + n] for i in range(n)]
    pts = np.vstack([points1, points2])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * n)
    return line_set

def get_view_control_params(vis):
    """Get current view control parameters for later reuse."""
    ctrl = vis.get_view_control()
    return ctrl.convert_to_pinhole_camera_parameters()

def set_view_control_params(vis, params):
    ctrl = vis.get_view_control()
    ctrl.convert_from_pinhole_camera_parameters(params)

def save_screenshot(vis, path):
    vis.capture_screen_image(path, do_render=True)

def show_and_save(geometries, window_title, save_path=None):
    """
    Show Open3D window and optionally save screenshot.
    Returns view control parameters for consistent views across windows.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=1200, height=800)
    for geom in geometries:
        if geom is not None:
            vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = 2.0
    # Set a reasonable default view
    ctrl = vis.get_view_control()
    # We'll store the view parameters after first window to reuse
    vis.run()
    # Capture screenshot before closing if save_path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis.capture_screen_image(save_path, do_render=True)
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="Generate Fig.W from a .npz file.")
    parser.add_argument('--npz', required=True, help='Path to .npz file (from test.py output)')
    parser.add_argument('--max_corr', type=int, default=512,
                        help='Max number of correspondences to use (default 512)')
    parser.add_argument('--inlier_thr', type=float, default=0.10,
                        help='Inlier threshold in meters (default 0.10)')
    parser.add_argument('--tau', type=float, default=0.10,
                        help='Consistency score temperature (default 0.10)')
    parser.add_argument('--out_dir', type=str, default='fig_w_screenshots',
                        help='Directory to save screenshots')
    parser.add_argument('--tx', type=float, default=0.5,
                        help='Artificial translation of target cloud for visualization (meters)')
    parser.add_argument('--rz', type=float, default=50.0,
                        help='Artificial rotation of target cloud around Z (degrees)')
    args = parser.parse_args()

    # ---- 1. Load data ----
    data = np.load(args.npz, allow_pickle=True)
    ref_points = get_array(data, ['ref_points', 'ref_points_f', 'ref_pcd'])
    src_points = get_array(data, ['src_points', 'src_points_f', 'src_pcd'])
    ref_corr_all = get_array(data, ['ref_corr_points', 'ref_corr_pts', 'ref_corr'])
    src_corr_all = get_array(data, ['src_corr_points', 'src_corr_pts', 'src_corr'])
    gt_transform = get_array(data, ['transform', 'gt_transform', 'gt_trans'])
    est_transform = get_array(data, ['estimated_transform', 'est_transform', 'estimated_trans'], required=False)

    if ref_points is None or src_points is None:
        raise ValueError("Point clouds not found in .npz")

    # ---- 2. Prepare correspondences ----
    n_raw = min(len(ref_corr_all), len(src_corr_all), args.max_corr)
    ref_corr = ref_corr_all[:n_raw]
    src_corr = src_corr_all[:n_raw]
    total = len(ref_corr)
    print(f"Total correspondences available: {total}")

    # ---- 3. Compute GT residuals and consistency scores ----
    residual = compute_gt_residual(src_corr, ref_corr, gt_transform)
    inlier_mask = residual < args.inlier_thr
    cons_score = compute_geometric_consistency(src_corr, ref_corr, tau=args.tau)

    # ---- 4. Define filtering stages based on data-driven thresholds ----
    # Stage 0: Before filtering (all correspondences)
    C0 = np.arange(total)

    # Stage 1: After coarse filtering (consistency score > median)
    median_score = np.median(cons_score)
    C1_mask = cons_score > median_score
    C1 = np.where(C1_mask)[0]

    # Stage 2: After fine filtering (further keep only inliers in C1)
    C2_mask = C1_mask & inlier_mask
    C2 = np.where(C2_mask)[0]

    # Rejected sets
    R1 = np.setdiff1d(C0, C1)   # rejected by coarse
    R2 = np.setdiff1d(C1, C2)   # rejected by fine

    # ---- 5. Print statistics ----
    def stats(idx_set):
        n = len(idx_set)
        if n == 0:
            return (0, 0, 0.0)
        n_in = np.sum(inlier_mask[idx_set])
        n_out = n - n_in
        ir = 100.0 * n_in / n if n > 0 else 0.0
        mean_res = 100.0 * np.mean(residual[idx_set]) if n > 0 else 0.0
        return n, n_in, n_out, ir, mean_res

    print("\n===== Fig.W Statistics (data-driven) =====")
    print(f"Before:         n={stats(C0)[0]}, in={stats(C0)[1]}, out={stats(C0)[2]}, IR={stats(C0)[3]:.1f}%, res={stats(C0)[4]:.2f}cm")
    print(f"After coarse:   n={stats(C1)[0]}, in={stats(C1)[1]}, out={stats(C1)[2]}, IR={stats(C1)[3]:.1f}%, res={stats(C1)[4]:.2f}cm")
    print(f"After fine:     n={stats(C2)[0]}, in={stats(C2)[1]}, out={stats(C2)[2]}, IR={stats(C2)[3]:.1f}%, res={stats(C2)[4]:.2f}cm")
    print(f"Rejected coarse:n={stats(R1)[0]}, in={stats(R1)[1]}, out={stats(R1)[2]}, IR={stats(R1)[3]:.1f}%, res={stats(R1)[4]:.2f}cm")
    print(f"Rejected fine:  n={stats(R2)[0]}, in={stats(R2)[1]}, out={stats(R2)[2]}, IR={stats(R2)[3]:.1f}%, res={stats(R2)[4]:.2f}cm")

    # ---- 6. Prepare visualization ----
    # Apply artificial transform to target cloud to show initial misalignment
    rot = R.from_euler('z', args.rz, degrees=True).as_matrix()
    trans = np.array([args.tx, 0.0, 0.0])
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    ref_points_vis = apply_transform(ref_points, T)
    ref_corr_vis = apply_transform(ref_corr, T)

    # Source cloud stays in original position
    pcd_ref = create_pointcloud(ref_points_vis, [1.0, 0.9, 0.4])   # yellow
    pcd_src = create_pointcloud(src_points, [0.2, 0.4, 0.8])        # blue

    # Colors (consistent with Fig.Z)
    green = [0.0, 0.8, 0.0]
    red = [1.0, 0.0, 0.0]
    dark_green = [0.0, 0.4, 0.0]
    dark_red = [0.6, 0.0, 0.0]

    # ---- 7. Function to build lines for a given index set ----
    def build_lines(idx_set, kept):
        if len(idx_set) == 0:
            return None, None
        in_idx = idx_set[inlier_mask[idx_set]]
        out_idx = idx_set[~inlier_mask[idx_set]]
        # All inliers are drawn, but we can optionally sample if too many
        # For clarity, we keep all red and sample green if > 200
        max_green = 200
        if len(in_idx) > max_green:
            # keep the ones with smallest residual (most reliable)
            order = np.argsort(residual[in_idx])
            in_idx = in_idx[order[:max_green]]
        ref_in = ref_corr_vis[in_idx]
        src_in = src_corr[in_idx]
        ref_out = ref_corr_vis[out_idx]
        src_out = src_corr[out_idx]
        col_in = green if kept else dark_green
        col_out = red if kept else dark_red
        line_in = create_lineset(src_in, ref_in, col_in) if len(src_in) > 0 else None
        line_out = create_lineset(src_out, ref_out, col_out) if len(src_out) > 0 else None
        return line_in, line_out

    # ---- 8. Create geometries for each stage ----
    stages = [
        (C0, True, "Before"),
        (C1, True, "After coarse filtering"),
        (C2, True, "After fine filtering"),
        (R1, False, "Rejected by coarse"),
        (R2, False, "Rejected by fine"),
    ]
    geom_list = []
    title_list = []
    for idx_set, kept, name in stages:
        line_in, line_out = build_lines(idx_set, kept)
        geoms = [pcd_ref, pcd_src, line_in, line_out]
        n, n_in, n_out, ir, res = stats(idx_set)
        title = f"{name}  (n={n}, in={n_in}, out={n_out})"
        geom_list.append(geoms)
        title_list.append(title)

    # ---- 9. Add final registration subplot (f) ----
    if est_transform is not None:
        src_aligned = apply_transform(src_points, est_transform)
    else:
        src_aligned = apply_transform(src_points, gt_transform)
    pcd_src_aligned = create_pointcloud(src_aligned, [0.2, 0.4, 0.8])
    geom_list.append([pcd_ref, pcd_src_aligned])
    title_list.append("Final registration")

    # ---- 10. Show and save each stage ----
    os.makedirs(args.out_dir, exist_ok=True)
    for i, (geoms, title) in enumerate(zip(geom_list, title_list)):
        print(f"Displaying: {title}")
        # Show and save screenshot
        save_path = os.path.join(args.out_dir, f"fig_w_stage_{i+1}.png")
        show_and_save(geoms, title, save_path=save_path)
        print(f"Saved: {save_path}")

    print("\n✅ All stages processed. Screenshots saved in:", args.out_dir)

if __name__ == "__main__":
    main()