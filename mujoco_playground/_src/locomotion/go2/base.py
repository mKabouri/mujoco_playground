from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env


@dataclass
class BaseEnvConfig:
    action_scale: float
    ctrl_dt: float
    sim_dt: float
    task_name: str = "default"
    randomize_tasks: bool = False
    kp: float = 30.0
    kd: float = 1.0
    debug: bool = False
    leg_control: str = "torque"


class BaseEnv(mjx_env.MjxEnv):
    def __init__(self, config: BaseEnvConfig):
        # Validate timesteps
        assert jnp.allclose(config.ctrl_dt % config.sim_dt, 0.0), (
            "sim_dt must divide evenly into ctrl_dt"
        )

        self._config = config
        self._ctrl_dt = config.ctrl_dt
        self._sim_dt = config.sim_dt

        # Create models
        self._mj_model = self.make_mj_model(config)
        self._mjx_model = mjx.put_model(self._mj_model)

        # Joint limit definitions
        self.physical_joint_range = jnp.array(self._mj_model.jnt_range[1:])
        self.joint_range = self.physical_joint_range
        self.joint_torque_range = jnp.array(self._mj_model.actuator_ctrlrange)

        # Number of degrees of freedom
        self._nv = self._mj_model.nv
        self._nq = self._mj_model.nq

    def make_mj_model(self, config: BaseEnvConfig) -> mujoco.MjModel:
        raise NotImplementedError

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def dt(self) -> float:
        return self._ctrl_dt

    @property
    def sim_dt(self) -> float:
        return self._sim_dt

    @property
    def n_substeps(self) -> int:
        return int(round(self.dt / self.sim_dt))

    def make_data(
        self,
        qpos: jax.Array | None = None,
        qvel: jax.Array | None = None,
        ctrl: jax.Array | None = None,
    ) -> mjx.Data:
        """Initialize MJX Data with optional state values."""
        data = mjx.make_data(self._mjx_model)
        if qpos is not None:
            data = data.replace(qpos=qpos)
        if qvel is not None:
            data = data.replace(qvel=qvel)
        if ctrl is not None:
            data = data.replace(ctrl=ctrl)
        return data

    def step_physics(self, data: mjx.Data, ctrl: jax.Array) -> mjx.Data:
        """Step the physics simulation n_substeps times."""
        def single_step(d, _):
            d = d.replace(ctrl=ctrl)
            d = mjx.step(self._mjx_model, d)
            return d, None

        return jax.lax.scan(single_step, data, (), self.n_substeps)[0]

    @partial(jax.jit, static_argnums=(0,))
    def act2joint(self, act: jax.Array) -> jax.Array:
        """
        Convert action to joint targets using DELTA control.
        Target = Default_Pose + Action * Scale
        """
        # Use the default standing pose extracted in __init__
        joint_targets = self._default_pose + act*self._config.action_scale

        # Clip to physical limits for safety
        joint_targets = jnp.clip(
            joint_targets,
            self.physical_joint_range[:, 0],
            self.physical_joint_range[:, 1],
        )
        return joint_targets

    @partial(jax.jit, static_argnums=(0,))
    def act2tau(self, act: jax.Array, data: mjx.Data) -> jax.Array:
        """Convert action to joint torques using PD control."""
        joint_target = self.act2joint(act)

        # Get joint positions and velocities
        q = data.qpos[7:] if self._mj_model.nq > len(joint_target) else data.qpos
        q = q[: len(joint_target)]
        qd = data.qvel[6:] if self._mj_model.nv > len(joint_target) else data.qvel
        qd = qd[: len(joint_target)]

        # PD control
        q_err = joint_target - q
        tau = self._config.kp * q_err - self._config.kd * qd

        # Only clip when actuator ctrl range is actually specified (max > min).
        # mujoco sets ctrlrange=[0,0] to mean "unconstrained"; clipping to [0,0]
        # would zero all torques and cause the robot to collapse.
        tau_min = self.joint_torque_range[:, 0]
        tau_max = self.joint_torque_range[:, 1]
        has_range = tau_max > tau_min
        tau = jnp.where(has_range, jnp.clip(tau, tau_min, tau_max), tau)
        return tau

    def reset(self, rng: jax.Array):
        """Reset the environment. Must be implemented by subclasses."""
        raise NotImplementedError

    def step(self, state, action: jax.Array):
        """Step the environment. Must be implemented by subclasses."""
        raise NotImplementedError
