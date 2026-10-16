"""NeRF spiral trajectories, retaining the original 4DGS convention."""

import numpy as np


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate a mean camera pose from [N, 3, 4] camera-to-world matrices.
    Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    center = poses[..., 3].mean(0)
    z = normalize(poses[..., 2].mean(0))
    y_ = poses[..., 1].mean(0)
    x = normalize(np.cross(z, y_))
    y = np.cross(x, z)
    pose_avg = np.stack([x, y, z, center], 1)
    return pose_avg


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zrate, rotations=2, n_views=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    for theta in np.linspace(0.0, 2.0 * np.pi * rotations, n_views + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]) * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, n_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    c2w = average_poses(c2ws_all)
    up = normalize(c2ws_all[:, :3, 1].sum(0))
    dt = 0.75
    close_depth, inf_depth = (near_fars.min() * 0.9, near_fars.max() * 5.0)
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zrate=0.5, n_views=n_views)
    return np.stack(render_poses)
