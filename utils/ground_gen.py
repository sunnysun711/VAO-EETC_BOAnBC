# utils/ground_gen.py
"""Ground profile generation utilities."""

import numpy as np


def get_random_seed(title: str = '', display: bool = True) -> int:
    """Generate a random seed.

    Args:
        title: Optional title for display
        display: Whether to print the seed

    Returns:
        Random seed value
    """
    seed = np.random.randint(10000000)
    if display:
        print(f'RANDOM-{title}', f"numpy random seed is {seed}")
    return seed


def generate_random_ground_points(
    num_eval_points: int,
    *,
    seed: int | None = None,
    ds: float = 50.0,
    platform_intervals: int = 1,
    smooth_scale: float = 1.0,
    flat_grad_max: float = 0.03,
) -> np.ndarray:
    """Generate random ground profile points.

    Generates a (num_eval_points+1, 2) array with first column as x-coordinates
    and second column as y-coordinates (elevations).

    Note: The first and last `platform_intervals` points are set to flat platforms.

    Args:
        num_eval_points: Number of evaluation points
        seed: Random seed (if None, generates one)
        ds: interval length (default 50.0)
        platform_intervals: Number of platform intervals at start/end
        smooth_scale: Scale factor for terrain smoothness (default 1.0)
        flat_grad_max: Maximum flat gradient (default 0.03)

    Returns:
        (num_eval_points+1, 2) ndarray with [x, y] columns
    """
    seed = get_random_seed(title='gen_rand_ground') if seed is None else seed
    np.random.seed(seed)

    # Generate random walk terrain
    terrain = np.cumsum(np.random.randn(num_eval_points + 1) * 5)
    e6g = dict(enumerate(terrain))
    x, y = list(e6g.keys()), list(e6g.values())
    
    # Normalize: shift y to make min height 0
    y = np.array(y) - min(y)
    y = y * smooth_scale
    
    h1, h2 = y[0], y[-1]  # Heights at start and end
    cur_grad = abs(h1 - h2) / (ds * (num_eval_points-2*platform_intervals))
    if cur_grad > flat_grad_max:
        y = y * flat_grad_max / cur_grad
    
    # Set platform intervals (flat areas at start and end)
    for i in range(platform_intervals + 1):
        y[i] = y[0]
        y[-i - 1] = y[-1]
    
    return np.array([x, y]).T


if __name__ == '__main__':
    ground_points = generate_random_ground_points(
        num_eval_points=100,
        seed=None,
        ds=50.0,
        platform_intervals=4,
        smooth_scale=1.0,
        flat_grad_max=0.015,
    )
    print(ground_points)
    
    import matplotlib.pyplot as plt
    plt.plot(ground_points[:, 0], ground_points[:, 1], marker='o')
    plt.show()