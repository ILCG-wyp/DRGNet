import numpy as np
import open3d as o3d
import argparse
import glob
import os
import sys

# ============================================
# Project root detection
# ============================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# ============================================
# Helper functions
# ============================================

def create_point_cloud(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd


def create_sphere_mesh(points, color, radius=0.03):
    mesh = o3d.geometry.TriangleMesh()
    for pt in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(pt)
        sphere.paint_uniform_color(color)
        mesh += sphere
    return mesh


def create_correspondence_lines(src_points, tgt_points, correspondences,
                                inlier_mask=None, transform=None, threshold=0.1):
    """
    Build colored line set for correspondences.
    Green: inlier, Red: outlier.
    """
    if len(correspondences) == 0:
        return None, None

    src_idx = correspondences[:, 0]
    tgt_idx = correspondences[:, 1]

    # Safety clipping
    src_idx = src_idx[src_idx < len(src_points)]
    tgt_idx = tgt_idx[tgt_idx < len(tgt_points)]
    min_len = min(len(src_idx), len(tgt_idx))
    src_idx = src_idx[:min_len]
    tgt_idx = tgt_idx[:min_len]

    src_corr = src_points[src_idx]
    tgt_corr = tgt_points[tgt_idx]

    # Determine inliers
    if transform is not None:
        R = transform[:3, :3]
        t = transform[:3, 3]
        src_trans = (R @ src_corr.T).T + t
        errors = np.linalg.norm(src_trans - tgt_corr, axis=1)
        is_inlier = errors < threshold
    elif inlier_mask is not None:
        is_inlier = inlier_mask[:min_len]
    else:
        is_inlier = np.ones(min_len, dtype=bool)

    # Build line set
    points_list = []
    lines_list = []
    colors_list = []
    for i in range(min_len):
        points_list.append(src_corr[i])
        points_list.append(tgt_corr[i])
        lines_list.append([2*i, 2*i+1])
        colors_list.append([0, 1, 0] if is_inlier[i] else [1, 0, 0])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points_list))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines_list))
    line_set.colors = o3d.utility.Vector3dVector(np.array(colors_list))

    return line_set, is_inlier


def create_render_option(background='white', point_size=3.0, line_width=2.0):
    opt = o3d.visualization.RenderOption()
    if background == 'white':
        opt.background_color = np.array([1.0, 1.0, 1.0])
    else:  # black
        opt.background_color = np.array([0.0, 0.0, 0.0])
    opt.point_size = point_size
    opt.line_width = line_width
    opt.light_on = True
    opt.mesh_show_back_face = False
    return opt


# ============================================
# Main visualization function
# ============================================

def visualize_drg_style(npz_path, save=False, output_dir='figures',
                        show_nodes=False, show_original=False,
                        background='white', mode='default'):
    """
    DRG-Net paper-style visualization with white background and colored correspondences.
    """
    print("=" * 70)
    print("DRG-Net Registration Visualization")
    print("=" * 70)

    # Load data
    print(f"Loading: {npz_path}")
    data = np.load(npz_path)

    ref_points = data['ref_points_f']
    src_points = data['src_points_f']
    ref_points_c = data['ref_points_c']
    src_points_c = data['src_points_c']

    # Transformation
    if 'estimated_transform' in data:
        est_T = data['estimated_transform']
        R_est = est_T[:3, :3]
        t_est = est_T[:3, 3]
        src_transformed = (R_est @ src_points.T).T + t_est
        print("Using estimated transform")
    else:
        est_T = None
        src_transformed = src_points
        print("No estimated transform found, using raw points")

    gt_T = data['transform'] if 'transform' in data else None

    # Correspondences
    if 'ref_node_corr_indices' in data and 'src_node_corr_indices' in data:
        ref_idx = data['ref_node_corr_indices']
        src_idx = data['src_node_corr_indices']
        corr = np.column_stack([ref_idx, src_idx])
        print(f"Correspondences: {len(corr)} pairs")
    else:
        corr = np.array([])
        print("No correspondences found")

    print(f"Source points: {src_points.shape[0]}, Target points: {ref_points.shape[0]}")
    print(f"Source nodes: {src_points_c.shape[0]}, Target nodes: {ref_points_c.shape[0]}")

    # Build geometries
    target_pcd = create_point_cloud(ref_points, [0.7, 0.7, 1.0])   # light blue
    source_pcd = create_point_cloud(src_transformed, [1.0, 0.9, 0.4])  # yellow
    source_orig = create_point_cloud(src_points, [0.2, 0.4, 0.8]) if show_original else None

    # Nodes
    source_nodes = create_sphere_mesh(src_points_c, [0.8, 0.2, 0.8], radius=0.03)
    target_nodes = create_sphere_mesh(ref_points_c, [0.2, 0.8, 0.8], radius=0.03)

    # Correspondences lines
    corr_lines = None
    inlier_mask = None
    inlier_ratio = 0.0
    if len(corr) > 0:
        transform_use = gt_T if gt_T is not None else est_T
        corr_lines, inlier_mask = create_correspondence_lines(
            src_points_c, ref_points_c, corr,
            transform=transform_use, threshold=0.1
        )
        if inlier_mask is not None:
            inlier_ratio = np.mean(inlier_mask)
            print(f"Inlier ratio: {inlier_ratio:.2%} ({np.sum(inlier_mask)}/{len(inlier_mask)})")

    # Collect objects
    objects = [target_pcd, source_pcd]
    if show_original and source_orig is not None:
        objects.append(source_orig)
    if show_nodes:
        objects.extend([source_nodes, target_nodes])
    if corr_lines is not None:
        objects.append(corr_lines)

    # Interactive selection for additional options (if not using args)
    print("\n" + "=" * 70)
    print("Visualization controls:")
    print("  Mouse drag: rotate, Wheel: zoom, Ctrl+drag: pan")
    print("  R: reset view, C: center, S: save screenshot (if window focused)")
    print("  Q / ESC: exit")

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"DRG-Net: {os.path.basename(npz_path)}",
                      width=1600, height=900, visible=True)

    for obj in objects:
        vis.add_geometry(obj)

    # Set render options
    render_opt = create_render_option(background=background)
    vis.get_render_option().background_color = render_opt.background_color
    vis.get_render_option().point_size = render_opt.point_size
    vis.get_render_option().line_width = render_opt.line_width

    # Auto view
    try:
        all_pts = np.vstack([ref_points, src_transformed])
        center = np.mean(all_pts, axis=0)
        ctrl = vis.get_view_control()
        ctrl.set_front([0, -1, -0.5])
        ctrl.set_up([0, -0.5, 1])
        ctrl.set_lookat(center)
        ctrl.set_zoom(0.8)
    except:
        pass

    vis.run()
    vis.destroy_window()

    # Save if requested
    if save:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(npz_path))[0]

        vis2 = o3d.visualization.Visualizer()
        vis2.create_window(width=1600, height=900, visible=False)
        for obj in objects:
            vis2.add_geometry(obj)
        vis2.get_render_option().background_color = render_opt.background_color
        ctrl2 = vis2.get_view_control()
        try:
            ctrl2.set_front([0, -1, -0.5])
            ctrl2.set_up([0, -0.5, 1])
            ctrl2.set_lookat(center)
            ctrl2.set_zoom(0.8)
        except:
            pass
        vis2.poll_events()
        vis2.update_renderer()

        img_path = os.path.join(output_dir, f"{base}_registration.png")
        vis2.capture_screen_image(img_path, do_render=True)
        vis2.destroy_window()
        print(f"Screenshot saved: {img_path}")

        # Also save point clouds as PLY
        o3d.io.write_point_cloud(os.path.join(output_dir, f"{base}_target.ply"), target_pcd)
        o3d.io.write_point_cloud(os.path.join(output_dir, f"{base}_source.ply"), source_pcd)
        print(f"Point clouds saved as PLY in {output_dir}")


# ============================================
# Command line interface
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description='DRG-Net registration visualization with white background and colored correspondences'
    )
    parser.add_argument('--npz_path', required=True, help='Path to .npz file')
    parser.add_argument('--save', '-s', action='store_true', help='Save screenshot and PLY files')
    parser.add_argument('--output_dir', default='figures', help='Output directory for saved files')
    parser.add_argument('--show_nodes', action='store_true', help='Display superpoint nodes as spheres')
    parser.add_argument('--show_original', action='store_true', help='Show original (untransformed) source point cloud')
    parser.add_argument('--background', choices=['white', 'black'], default='white', help='Background color')
    parser.add_argument('--mode', choices=['default', 'error_map'], default='default',
                        help='Visualization mode (error_map not yet implemented)')

    args = parser.parse_args()

    if not os.path.exists(args.npz_path):
        print(f"File not found: {args.npz_path}")
        # Try to find a sample
        sample_pattern = '../../output/3DMatch/features/3DMatch/**/*.npz'
        files = glob.glob(sample_pattern, recursive=True)
        if files:
            print(f"Found {len(files)} .npz files. Try one of:")
            for f in files[:5]:
                print(f"  {f}")
        return

    visualize_drg_style(
        npz_path=args.npz_path,
        save=args.save,
        output_dir=args.output_dir,
        show_nodes=args.show_nodes,
        show_original=args.show_original,
        background=args.background,
        mode=args.mode
    )


if __name__ == "__main__":
    main()