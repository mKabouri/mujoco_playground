from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2.base import BaseEnv, BaseEnvConfig

_ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "go2" / "unitree_go2"


def global_to_body_velocity(v, q):
    from brax import math
    return math.inv_rotate(v, q)


def body_to_global_velocity(v, q):
    from brax import math
    return math.rotate(v, q)


@jax.jit
def get_foot_step(duty_ratio, cadence, amplitude, phases, time):
    def step_height(t, footphase, duty_ratio):
        angle = (t + jnp.pi - footphase) % (2 * jnp.pi) - jnp.pi
        angle = jnp.where(duty_ratio < 1, angle * 0.5 / (1 - duty_ratio), angle)
        clipped_angle = jnp.clip(angle, -jnp.pi / 2, jnp.pi / 2)
        value = jnp.where(duty_ratio < 1, jnp.cos(clipped_angle), 0)
        final_value = jnp.where(jnp.abs(value) >= 1e-6, jnp.abs(value), 0.0)
        return final_value

    h_steps = amplitude * jax.vmap(step_height, in_axes=(None, 0, None))(
        time * 2 * jnp.pi * cadence + jnp.pi,
        2 * jnp.pi * phases,
        duty_ratio,
    )
    return h_steps


@dataclass
class UnitreeGo2EnvConfig(BaseEnvConfig):
    default_vx: float = 1.0
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ramp_up_time: float = 2.0
    gait: str = "walk"


class UnitreeGo2Env(BaseEnv):

    def __init__(
        self,
        config: config_dict.ConfigDict,
        config_overrides: dict[str, Any] = None,
    ):
        # ConfigDict doesn't have .copy()
        self._full_config = config_dict.ConfigDict(config)
        if config_overrides:
            for key, value in config_overrides.items():
                self._full_config[key] = value

        base_config = BaseEnvConfig(
            task_name=self._full_config.get("task_name", "walk"),
            randomize_tasks=self._full_config.get("randomize_tasks", False),
            kp=self._full_config.get("kp", 80.0),
            kd=self._full_config.get("kd", 2.0),
            debug=self._full_config.get("debug", False),
            ctrl_dt=self._full_config["ctrl_dt"],
            sim_dt=self._full_config["sim_dt"],
            leg_control=self._full_config["leg_control"],
            action_scale=self._full_config["action_scale"],
        )

        super().__init__(base_config)

        self._foot_radius = 0.0175
        self._gait = self._full_config.get("gait", "walk")

        self._gait_phase = {
            "stand": jnp.zeros(4),
            "walk": jnp.array([0.0, 0.5, 0.75, 0.25]),
            "trot": jnp.array([0.0, 0.5, 0.5, 0.0]),
            "canter": jnp.array([0.0, 0.33, 0.33, 0.66]),
            "gallop": jnp.array([0.0, 0.05, 0.4, 0.35]),
        }
        self._gait_params = {
            # duty_ratio, cadence, amplitude
            "stand": jnp.array([1.0, 1.0, 0.0]),
            "walk": jnp.array([0.75, 1.0, 0.08]),
            "trot": jnp.array([0.45, 2, 0.08]),
            "canter": jnp.array([0.4, 4, 0.06]),
            "gallop": jnp.array([0.3, 3.5, 0.10]),
        }

        self._torso_idx = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "base"
        )
        self._init_q = jnp.array(self._mj_model.keyframe("home").qpos)
        self._default_pose = self._mj_model.keyframe("home").qpos[7:]

        self.joint_range = jnp.array(
            [
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -0.85],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -0.85],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -1.3],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -1.3],
            ]
        )

        feet_site = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        feet_site_id = [
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), "Site not found."
        self._feet_site_id = jnp.array(feet_site_id)

    def make_mj_model(self, config: BaseEnvConfig) -> mujoco.MjModel:
        model = mujoco.MjModel.from_xml_path(
            (_ROOT_PATH / "mjx_scene_position.xml").as_posix()
        )
        model.opt.timestep = config.sim_dt
        if config.leg_control == "position":
            # gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kd
            model.actuator_gainprm[:, 0] = config.kp
            model.actuator_biasprm[:, 1] = -config.kp
            model.actuator_biasprm[:, 2] = -config.kd
        return model

    @property
    def xml_path(self) -> str:
        return (_ROOT_PATH / "mjx_scene_position.xml").as_posix()

    @property
    def action_size(self) -> int:
        return 12

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, key = jax.random.split(rng)

        data = self.make_data(qpos=self._init_q, qvel=jnp.zeros(self._nv))
        data = mjx.forward(self._mjx_model, data)

        # settle the robot on the ground before the episode starts
        def _settle_step(d, _):
            d = d.replace(ctrl=self._default_pose)
            return mjx.step(self._mjx_model, d), None

        data, _ = jax.lax.scan(_settle_step, data, None, length=50)
        data = data.replace(qvel=jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "pos_tar": jnp.array([0.282, 0.0, 0.3]),
            "vel_tar": jnp.array([0.0, 0.0, 0.0]),
            "ang_vel_tar": jnp.array([0.0, 0.0, 0.0]),
            "yaw_tar": 0.0,
            "step": 0,
            "z_feet": jnp.zeros(4),
            "z_feet_tar": jnp.zeros(4),
            "randomize_target": self._config.randomize_tasks,
            "last_contact": jnp.zeros(4, dtype=jnp.bool),
            "feet_air_time": jnp.zeros(4),
        }

        obs = self._get_obs(data, state_info)
        reward, done = jnp.zeros(2)
        metrics = {}

        return mjx_env.State(data, obs, reward, done, metrics, state_info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        if self._config.leg_control == "position":
            ctrl = self.act2joint(action)
        elif self._config.leg_control == "torque":
            ctrl = self.act2tau(action, state.data)
        else:
            raise ValueError(f"Unknown leg_control: {self._config.leg_control}")

        data = self.step_physics(state.data, ctrl)

        def dont_randomize():
            return (
                jnp.array([
                    self._full_config.get("default_vx", 1.0),
                    self._full_config.get("default_vy", 0.0),
                    0.0,
                ]),
                jnp.array([0.0, 0.0, self._full_config.get("default_vyaw", 0.0)]),
            )

        def randomize():
            return self.sample_command(cmd_rng)

        vel_tar, ang_vel_tar = jax.lax.cond(
            (state.info["randomize_target"]) & (state.info["step"] % 500 == 0),
            randomize,
            dont_randomize,
        )

        ramp_up_time = self._full_config.get("ramp_up_time", 0.0)
        ramp_factor = jnp.where(
            ramp_up_time <= 0.0,
            1.0,
            jnp.minimum(state.info["step"] * self.dt / ramp_up_time, 1.0),
        )
        state.info["vel_tar"] = vel_tar * ramp_factor
        state.info["ang_vel_tar"] = ang_vel_tar * ramp_factor

        # _compute_reward also updates contact tracking in state.info
        reward = self._compute_reward(data, state.info)
        done = self._check_done(data)
        obs = self._get_obs(data, state.info)

        state.info["step"] += 1
        state.info["rng"] = rng

        return mjx_env.State(data, obs, reward, done, {}, state.info)

    def _get_obs(self, data: mjx.Data, state_info: dict[str, Any]) -> jax.Array:
        # qpos: [x, y, z, qw, qx, qy, qz, joint×12]
        pos = data.qpos[0:3]
        quat = data.qpos[3:7]

        # qvel: [vx, vy, vz, wx, wy, wz, joint_vel×12]
        vel = data.qvel[0:3]
        ang = data.qvel[3:6]

        vb = global_to_body_velocity(vel, quat)
        ab = global_to_body_velocity(ang, quat)

        obs = jnp.concatenate([
            state_info["vel_tar"],      # 3
            state_info["ang_vel_tar"],  # 3
            data.ctrl,                  # 12
            data.qpos,                  # 19
            vb,                         # 3
            ab,                         # 3
            data.qvel[6:],              # 12
        ])
        return obs

    def _compute_reward(self, data: mjx.Data, state_info: dict[str, Any]) -> jax.Array:
        from brax import math

        pos = data.qpos[0:3]
        quat = data.qpos[3:7]
        vel = data.qvel[0:3]
        ang = data.qvel[3:6]

        z_feet = data.site_xpos[self._feet_site_id][:, 2]
        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]
        z_feet_tar = get_foot_step(
            duty_ratio, cadence, amplitude, phases, state_info["step"] * self.dt
        )
        # gallop has a wider tolerance so the gait penalty doesn't swamp the alive bonus
        gait_tol = 0.10 if self._gait == "gallop" else 0.05
        reward_gaits = -jnp.sum(((z_feet_tar - z_feet) / gait_tol) ** 2)

        foot_pos = data.site_xpos[self._feet_site_id]
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3
        contact_filt_mm = contact | state_info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state_info["last_contact"]
        first_contact = (state_info["feet_air_time"] > 0) * contact_filt_mm

        state_info["feet_air_time"] += self.dt
        reward_air_time = jnp.sum((state_info["feet_air_time"] - 0.1) * first_contact)

        pos_tar = state_info["pos_tar"] + state_info["vel_tar"] * self.dt * state_info["step"]
        R = math.quat_to_3x3(quat)
        head_vec = jnp.array([0.285, 0.0, 0.0])
        head_pos = pos + jnp.dot(R, head_vec)
        reward_pos = -jnp.sum((head_pos - pos_tar) ** 2)

        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, quat)
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))

        yaw_tar = (
            state_info["yaw_tar"] + state_info["ang_vel_tar"][2] * self.dt * state_info["step"]
        )
        yaw = math.quat_to_euler(quat)[2]
        d_yaw = yaw - yaw_tar
        reward_yaw = -jnp.square(jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw)))

        vb = global_to_body_velocity(vel, quat)
        ab = global_to_body_velocity(ang, quat)
        reward_vel = -jnp.sum((vb[:2] - state_info["vel_tar"][:2]) ** 2)
        reward_ang_vel = -jnp.sum((ab[2] - state_info["ang_vel_tar"][2]) ** 2)

        reward_height = -jnp.sum((pos[2] - state_info["pos_tar"][2]) ** 2)
        reward_energy = -jnp.sum(jnp.maximum(data.ctrl * data.qvel[6:] / 160.0, 0.0) ** 2)

        done = self._check_done(data)
        reward_alive = 1.0 - done

        if self._gait == "gallop":
            reward = (
                reward_gaits    * 0.5
                + reward_air_time * 0.5
                + reward_pos      * 0.0
                + reward_upright  * 0.5
                + reward_yaw      * 0.1
                + reward_vel      * 1.0
                + reward_ang_vel  * 0.2
                + reward_height   * 0.3
                + reward_energy   * 0.001
                + reward_alive    * 2.0
            )
        else:
            reward = (
                reward_gaits    * 0.1
                + reward_air_time * 0.0
                + reward_pos      * 0.0
                + reward_upright  * 0.5
                + reward_yaw      * 0.3
                + reward_vel      * 1.0
                + reward_ang_vel  * 1.0
                + reward_height   * 1.0
                + reward_energy   * 0.0
                + reward_alive    * 2.0
            )

        state_info["feet_air_time"] *= ~contact_filt_mm
        state_info["last_contact"] = contact
        state_info["z_feet"] = z_feet
        state_info["z_feet_tar"] = z_feet_tar

        return reward

    def _check_done(self, data: mjx.Data) -> jax.Array:
        from brax import math

        pos = data.qpos[0:3]
        quat = data.qpos[3:7]
        joint_angles = data.qpos[7:]

        up = jnp.array([0.0, 0.0, 1.0])
        done = jnp.dot(math.rotate(up, quat), up) < 0.5
        done |= jnp.any(joint_angles < self.joint_range[:, 0])
        done |= jnp.any(joint_angles > self.joint_range[:, 1])
        done |= pos[2] < 0.18

        return done.astype(jnp.float32)

    def sample_command(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        lin_vel_x = [-1.5, 1.5]   # [m/s]
        lin_vel_y = [-0.5, 0.5]   # [m/s]
        ang_vel_yaw = [-1.5, 1.5] # [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1])
        lin_vel_y = jax.random.uniform(key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1])
        ang_vel_yaw = jax.random.uniform(key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1])

        new_lin_vel_cmd = jnp.array([lin_vel_x[0], lin_vel_y[0], 0.0])
        new_ang_vel_cmd = jnp.array([0.0, 0.0, ang_vel_yaw[0]])
        return new_lin_vel_cmd, new_ang_vel_cmd


def default_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.task_name = "walk"
    config.randomize_tasks = False
    config.default_vx = 1.0
    config.default_vy = 0.0
    config.default_vyaw = 0.0
    config.ramp_up_time = 2.0
    config.gait = "trot"  # stand, walk, trot, canter, gallop

    # MJX's explicit integrator needs higher kp than Brax (30) to hold the
    # standing pose above the height termination threshold
    config.kp = 80.0
    config.kd = 2.0
    config.leg_control = "position"
    config.action_scale = 0.25

    config.ctrl_dt = 0.02  # 20ms
    config.sim_dt = 0.005  # 5ms

    return config


def gallop_default_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.task_name = "gallop"
    config.randomize_tasks = False
    config.default_vx = 1.5
    config.default_vy = 0.0
    config.default_vyaw = 0.0
    config.ramp_up_time = 3.0
    config.gait = "gallop"

    config.kp = 80.0
    config.kd = 3.0  # stiffer for high-impact gallop
    config.leg_control = "position"
    config.action_scale = 0.5

    config.ctrl_dt = 0.02
    config.sim_dt = 0.005

    return config
