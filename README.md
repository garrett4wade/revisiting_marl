# A repository based on MAPPO for fast idea implementation

+ Write config in `configs/your_config.yaml`.
+ Register the config in `configs/config.py'
+ If a new environment is required,
  + write the environment class like `environment/gym_mujoco/`;
  + register the environment;
  + import the environment in `environment/__init__.py`.
+ Run the code using `python main.py --config your_registered_config_name`.
