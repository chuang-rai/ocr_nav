from pathlib import Path
import numpy as np
import yaml


def main():
    yaml_output_path = Path("/tmp/yaml_output.yaml")
    waypoints = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.7071, 0.0, 0.7071, 0.0],
            [1.0, 1.0, 0.0, 0.7071, 0.0, 0.7071, 0.0],
        ]
    )

    waypoints_to_yaml(yaml_output_path, waypoints)


if __name__ == "__main__":
    main()
