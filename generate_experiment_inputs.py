#!/usr/bin/env python3
"""
Generates a sweep of .in files for flood simulation experiments.
Edit the parameter lists below to adjust the sweep.
"""
import itertools
import os

# Parameter sweeps
rows_cols = [
    (32,32), (64, 64), (128, 128), (512, 512), (1024, 1024)
]
scenarios = ['M', 'V', 'D', 'd']
num_clouds_list = [4, 32, 256, 1024]
ex_factors = [10]
thresholds = [0.000001]
num_minutes = [1000]
cloud_max_radius = [50]
cloud_max_intensity = [80]
cloud_max_speed = [60]
cloud_max_angle = [45]
cloud_seed = 12345

# Fixed cloud front parameters (can be randomized if desired)
front_distance_factor = 0.2  # Fraction of min(rows, cols)
front_width_factor = 0.3     # Fraction of cols
front_depth_factor = 0.1     # Fraction of min(rows, cols)
front_direction = 240 # Direction of the cloud front (degrees)

def make_filename(params):
    return "exp_{rows}x{cols}_{scen}_c{clouds}_ex{ex}_t{thresh}_m{mins}.in".format(
        rows=params['rows'], cols=params['cols'], scen=params['scenario'], clouds=params['num_clouds'],
        ex=params['ex_factor'], thresh=str(params['threshold']).replace('.', 'p'), mins=params['num_minutes']
    )

def main():
    outdir = "test_files"
    os.makedirs(outdir, exist_ok=True)
    sweep = itertools.product(
        rows_cols, scenarios, num_clouds_list, ex_factors, thresholds, num_minutes,
        cloud_max_radius, cloud_max_intensity, cloud_max_speed, cloud_max_angle
    )
    for (rows, cols), scenario, num_clouds, ex_factor, threshold, mins, r, inten, speed, angle in sweep:
        # Scale front parameters based on scenario size
        front_distance = int(round(front_distance_factor * min(rows, cols)))
        front_width = int(round(front_width_factor * cols))
        front_depth = max(1, int(round(front_depth_factor * min(rows, cols))))
        params = dict(
            rows=rows, cols=cols, scenario=scenario, threshold=threshold, num_minutes=mins,
            ex_factor=ex_factor, front_distance=front_distance, front_width=front_width, front_depth=front_depth,
            front_direction=front_direction, num_clouds=num_clouds, r=r, inten=inten, speed=speed, angle=angle,
            seed=cloud_seed
        )
        args = [
            str(params['rows']), str(params['cols']), params['scenario'], f"{params['threshold']:.8f}",
            str(params['num_minutes']), str(params['ex_factor']), str(params['front_distance']),
            str(params['front_width']), str(params['front_depth']), str(params['front_direction']),
            str(params['num_clouds']), str(params['r']), str(params['inten']), str(params['speed']),
            str(params['angle']), str(params['seed'])
        ]
        fname = os.path.join(outdir, make_filename(params))
        with open(fname, 'w') as f:
            f.write(' '.join(args) + '\n')
    print(f"Generated {len(list(itertools.product(rows_cols, scenarios, num_clouds_list, ex_factors, thresholds, num_minutes, cloud_max_radius, cloud_max_intensity, cloud_max_speed, cloud_max_angle)))} input files in {outdir}/")

if __name__ == "__main__":
    main()
