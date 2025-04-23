import os
import time
import numpy as np
import torch as th
from typing import Tuple, List

from .config import Config

TEN = th.Tensor


class Evaluator:
    def __init__(self, cwd: str, env, args: Config, if_tensorboard: bool = False):
        self.cwd = cwd  # current working directory to save model
        self.env = env  # the env for Evaluator, `eval_env = env` in default
        self.agent_id = args.gpu_id
        self.total_step = 0  # the total training step
        self.start_time = time.time()  # `used_time = time.time() - self.start_time`
        self.eval_times = (
            args.eval_times
        )  # number of times that get episodic cumulative return
        self.eval_per_step = args.eval_per_step  # evaluate the agent per training steps
        self.eval_step_counter = (
            -self.eval_per_step
        )  # `self.total_step > self.eval_step_counter + self.eval_per_step`

        self.save_gap = args.save_gap
        self.save_counter = 0
        self.if_keep_save = args.if_keep_save
        self.if_over_write = args.if_over_write

        self.recorder_path = f"{cwd}/recorder.npy"
        self.recorder = []  # total_step, r_avg, r_std, critic_value, ...
        self.recorder_step = (
            args.eval_record_step
        )  # start recording after the exploration reaches this step.
        self.max_r = -np.inf
        print(
            "| Evaluator:"
            "\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
            "\n| `time`: Time spent from the start of training to this moment."
            "\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
            "\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
            "\n| `avgS`: Average of steps in an episode."
            "\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
            "\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
            f"\n{'#' * 80}\n"
            f"{'ID':<3}{'Step':>8}{'Time':>8} |"
            f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
            f"{'expR':>8}{'objC':>7}{'objA':>7}{'etc.':>7}",
            flush=True,
        )

        if getattr(env, "num_envs", 1) == 1:  # get attribute
            self.get_cumulative_rewards_and_step = (
                self.get_cumulative_rewards_and_step_single_env
            )
        else:  # vectorized environment
            self.get_cumulative_rewards_and_step = (
                self.get_cumulative_rewards_and_step_vectorized_env
            )

        if if_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard = SummaryWriter(f"{cwd}/tensorboard")
        else:
            self.tensorboard = None

    def evaluate_and_save(
        self, actor: th.nn.Module, steps: int, exp_r: float, logging_tuple: tuple
    ):  # Added type hint for actor
        self.total_step += steps  # update total training steps

        if self.total_step < self.recorder_step:
            return
        if self.total_step < self.eval_step_counter + self.eval_per_step:
            return

        self.eval_step_counter = self.total_step

        rewards_step_ten = self.get_cumulative_rewards_and_step(actor)

        # Handle empty or malformed tensor to prevent IndexError
        if (
            rewards_step_ten is None
            or not isinstance(rewards_step_ten, th.Tensor)
            or rewards_step_ten.ndim < 2
            or rewards_step_ten.shape[0] == 0
        ):
            # Log a warning or handle as appropriate, maybe skip evaluation?
            print(
                f"| Evaluator: Warning! No evaluation episodes completed or tensor is malformed ({rewards_step_ten.shape if isinstance(rewards_step_ten, th.Tensor) else type(rewards_step_ten)}). Skipping save/log for step {self.total_step}.",
                flush=True,
            )
            # Set default/invalid values to avoid downstream errors but indicate failure
            avg_r, std_r, avg_s, std_s = -np.inf, 0.0, 0.0, 0.0
            # If we return early, logging and saving logic below is skipped for this eval step
            # Consider if returning here is better than proceeding with invalid stats
            # return # Option to return early
        else:
            # Original logic assumes 2D tensor [n_episodes, 2]
            try:
                returns = rewards_step_ten[:, 0]  # episodic cumulative returns
                steps_in_episode = rewards_step_ten[
                    :, 1
                ]  # episodic step number (renamed from 'steps' to avoid confusion with func arg)
                avg_r = returns.mean().item()
                std_r = returns.std().item()
                avg_s = steps_in_episode.mean().item()
                std_s = steps_in_episode.std().item()
            except IndexError as e:
                print(
                    f"| Evaluator: ERROR during stats calculation! Tensor shape: {rewards_step_ten.shape}. Error: {e}",
                    flush=True,
                )
                avg_r, std_r, avg_s, std_s = -np.inf, 0.0, 0.0, 0.0

        train_time = int(time.time() - self.start_time)
        # Ensure logging_tuple has at least one element before accessing [-1]
        logging_str = ""
        value_tuple = []
        if logging_tuple:
            value_tuple = [
                v for v in logging_tuple[:-1] if isinstance(v, (int, float))
            ]  # Exclude last element for logging_str
            if isinstance(logging_tuple[-1], str):
                logging_str = logging_tuple[-1]
            else:  # Handle case where last element isn't a string
                value_tuple.append(
                    logging_tuple[-1]
                )  # Assume it's a value if not string

        """record the training information"""
        # Only record if evaluation was successful (avg_r is not -inf)
        if avg_r > -np.inf:
            self.recorder.append(
                (self.total_step, avg_r, std_r, exp_r, *value_tuple)
            )  # update recorder
            if (
                self.tensorboard and len(value_tuple) >= 2
            ):  # Check value_tuple has enough elements
                try:
                    self.tensorboard.add_scalar(
                        "info/critic_loss_sample", value_tuple[0], self.total_step
                    )
                    self.tensorboard.add_scalar(
                        "info/actor_obj_sample", -1 * value_tuple[1], self.total_step
                    )  # Assuming 2nd value is actor_obj
                    self.tensorboard.add_scalar(
                        "reward/avg_reward_sample", avg_r, self.total_step
                    )
                    self.tensorboard.add_scalar(
                        "reward/std_reward_sample", std_r, self.total_step
                    )
                    self.tensorboard.add_scalar(
                        "reward/exp_reward_sample", exp_r, self.total_step
                    )

                    self.tensorboard.add_scalar(
                        "info/critic_loss_time", value_tuple[0], train_time
                    )
                    self.tensorboard.add_scalar(
                        "info/actor_obj_time", -1 * value_tuple[1], train_time
                    )
                    self.tensorboard.add_scalar(
                        "reward/avg_reward_time", avg_r, train_time
                    )
                    self.tensorboard.add_scalar(
                        "reward/std_reward_time", std_r, train_time
                    )
                    self.tensorboard.add_scalar(
                        "reward/exp_reward_time", exp_r, train_time
                    )
                except Exception as e:
                    print(f"| Evaluator: Error logging to TensorBoard: {e}", flush=True)

        """print some information to Terminal"""
        prev_max_r = self.max_r
        # Only update max_r if evaluation was successful
        if avg_r > -np.inf:
            self.max_r = max(self.max_r, avg_r)  # update max average cumulative rewards
        print(
            f"{self.agent_id:<3}{self.total_step:8.2e}{train_time:8.0f} |"
            f"{avg_r:8.2f}{std_r:7.1f}{avg_s:7.0f}{std_s:6.0f} |"
            f"{exp_r:8.2f}{''.join(f'{n:7.2f}' for n in value_tuple)} {logging_str}",
            flush=True,
        )

        # Only save if evaluation was successful and improved performance
        if_save = avg_r > -np.inf and avg_r > prev_max_r
        if if_save:
            # Avoid saving curve if recorder is empty or only contains invalid entries
            if self.recorder and self.recorder[-1][1] > -np.inf:
                self.save_training_curve_jpg()
        if not self.if_keep_save:
            return

        self.save_counter += 1
        actor_path = None
        if if_save:  # save checkpoint with the highest episode return
            save_filename = (
                f"actor__{self.total_step:012}_{self.max_r:09.3f}.pt"
                if not self.if_over_write
                else "actor.pt"
            )
            actor_path = f"{self.cwd}/{save_filename}"

        elif (
            self.save_counter >= self.save_gap
        ):  # Use >= in case save_gap is 1 or counter skips
            self.save_counter = 0
            save_filename = (
                f"actor__{self.total_step:012}.pt"
                if not self.if_over_write
                else "actor.pt"
            )
            actor_path = f"{self.cwd}/{save_filename}"

        if actor_path:
            try:
                # It's generally recommended to save the state_dict for portability
                th.save(actor.state_dict(), actor_path)
                print(
                    f"| Evaluator: Saved model checkpoint to {actor_path}", flush=True
                )
                # Save curve after successful model save
                if self.recorder and self.recorder[-1][1] > -np.inf:
                    self.save_training_curve_jpg()
            except Exception as e:
                print(
                    f"| Evaluator: ERROR saving model to {actor_path}: {e}", flush=True
                )

    def save_or_load_recoder(self, if_save: bool):
        if if_save:
            if not self.recorder:  # Avoid saving empty recorder
                print(
                    "| Evaluator: Warning! Recorder is empty, not saving.", flush=True
                )
                return
            try:
                recorder_ary = np.array(self.recorder)
                np.save(self.recorder_path, recorder_ary)
                print(
                    f"| Evaluator: Saved recorder to {self.recorder_path}", flush=True
                )
            except Exception as e:
                print(
                    f"| Evaluator: ERROR saving recorder to {self.recorder_path}: {e}",
                    flush=True,
                )

        elif os.path.exists(self.recorder_path):
            try:
                recorder = np.load(self.recorder_path)
                if recorder.size == 0:  # Check if loaded array is empty
                    print(
                        f"| Evaluator: Warning! Loaded recorder file '{self.recorder_path}' is empty.",
                        flush=True,
                    )
                    self.recorder = []
                else:
                    self.recorder = [
                        tuple(i) for i in recorder
                    ]  # convert numpy to list
                    # Ensure loaded data is valid before accessing last element
                    if self.recorder:
                        self.total_step = self.recorder[-1][0]
                        # Recalculate max_r from loaded data, handling potential -inf
                        valid_rewards = [
                            rec[1]
                            for rec in self.recorder
                            if len(rec) > 1
                            and isinstance(rec[1], (int, float))
                            and rec[1] > -np.inf
                        ]
                        self.max_r = np.max(valid_rewards) if valid_rewards else -np.inf
                        print(
                            f"| Evaluator: Loaded recorder from {self.recorder_path}. Resuming from step {self.total_step}, max_r: {self.max_r:.2f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"| Evaluator: Loaded recorder from {self.recorder_path}, but it contained no valid data.",
                            flush=True,
                        )

            except Exception as e:
                print(
                    f"| Evaluator: Error loading recorder file '{self.recorder_path}': {e}",
                    flush=True,
                )
                self.recorder = []  # Reset recorder on load error

    def get_cumulative_rewards_and_step_single_env(
        self, actor
    ) -> TEN | None:  # Return None on major error
        # Added try-except block for robustness during evaluation runs
        rewards_steps_list = []
        try:
            actor.eval()  # Set actor to evaluation mode
            for _ in range(self.eval_times):
                result = get_rewards_and_steps(self.env, actor)
                if result is not None:  # Only append valid results
                    rewards_steps_list.append(result)
            actor.train()  # Set actor back to training mode
        except Exception as e:
            import traceback

            print(
                f"| Evaluator: Error during single-env evaluation runs: {e}", flush=True
            )
            print(traceback.format_exc(), flush=True)
            actor.train()  # Ensure actor is set back to train mode on error
            return None  # Return None on major error

        if not rewards_steps_list:
            print(
                "| Evaluator: Warning! No valid results from single-env evaluation.",
                flush=True,
            )
            return th.empty((0, 2), dtype=th.float32)  # Return empty 2D tensor

        try:
            rewards_steps_ten = th.tensor(rewards_steps_list, dtype=th.float32)
            # Ensure it's 2D even if only one result
            if rewards_steps_ten.ndim == 1 and rewards_steps_ten.shape[0] == 2:
                rewards_steps_ten = rewards_steps_ten.unsqueeze(0)
            elif rewards_steps_ten.ndim != 2:
                print(
                    f"| Evaluator: Warning! Tensor created with unexpected dimensions {rewards_steps_ten.ndim} in single-env. Shape: {rewards_steps_ten.shape}",
                    flush=True,
                )
                return th.empty(
                    (0, 2), dtype=th.float32
                )  # Return empty 2D tensor if shape is wrong
            return rewards_steps_ten
        except Exception as e:
            print(
                f"| Evaluator: Error converting results to tensor in single-env: {e}",
                flush=True,
            )
            return None

    def get_cumulative_rewards_and_step_vectorized_env(
        self, actor
    ) -> TEN | None:  # Return None on major error
        # Added try-except block for robustness during evaluation runs
        rewards_step_list_of_lists = []
        try:
            num_envs = getattr(self.env, "num_envs", 1)
            if num_envs <= 0:
                num_envs = 1  # Safety check
            eval_runs = max(1, self.eval_times // num_envs)
            actor.eval()  # Set actor to evaluation mode
            for _ in range(eval_runs):
                result = get_cumulative_rewards_and_step_from_vec_env(self.env, actor)
                if result is not None:  # Only append valid results
                    rewards_step_list_of_lists.append(result)
            actor.train()  # Set actor back to training mode
        except Exception as e:
            import traceback

            print(
                f"| Evaluator: Error during vectorized-env evaluation runs: {e}",
                flush=True,
            )
            print(traceback.format_exc(), flush=True)
            actor.train()  # Ensure actor is set back to train mode on error
            return None  # Return None on major error

        # Flatten the list of lists and filter out potential None or invalid entries
        flat_rewards_step_list = []
        for lst in rewards_step_list_of_lists:
            # lst should be a list of tuples (float, int)
            if isinstance(lst, list):
                flat_rewards_step_list.extend(
                    [item for item in lst if isinstance(item, tuple) and len(item) == 2]
                )

        if not flat_rewards_step_list:
            print(
                "| Evaluator: Warning! No valid results from vectorized-env evaluation.",
                flush=True,
            )
            return th.empty((0, 2), dtype=th.float32)  # Return empty 2D tensor

        try:
            rewards_step_ten = th.tensor(flat_rewards_step_list, dtype=th.float32)
            # Ensure it's 2D even if only one result
            if rewards_step_ten.ndim == 1 and rewards_step_ten.shape[0] == 2:
                rewards_step_ten = rewards_step_ten.unsqueeze(0)
            elif rewards_step_ten.ndim != 2:
                print(
                    f"| Evaluator: Warning! Tensor created with unexpected dimensions {rewards_step_ten.ndim} in vec-env. Shape: {rewards_step_ten.shape}",
                    flush=True,
                )
                return th.empty(
                    (0, 2), dtype=th.float32
                )  # Return empty 2D tensor if shape is wrong
            return rewards_step_ten
        except Exception as e:
            print(
                f"| Evaluator: Error converting results to tensor in vec-env: {e}",
                flush=True,
            )
            return None

    def save_training_curve_jpg(self):
        if not self.recorder:
            print(
                "| Evaluator: Warning! Recorder is empty, cannot save learning curve.",
                flush=True,
            )
            return
        try:
            recorder = np.array(self.recorder)
            # Filter out invalid entries (-inf avg_r or non-numeric)
            valid_mask = np.array(
                [
                    isinstance(rec[1], (int, float)) and rec[1] > -np.inf
                    for rec in recorder
                    if len(rec) > 1
                ]
            )
            if not np.any(valid_mask):
                print(
                    "| Evaluator: Warning! No valid data points in recorder to save learning curve.",
                    flush=True,
                )
                return
            valid_recorder = recorder[valid_mask]

            if valid_recorder.shape[0] < 2:  # Need at least 2 points to plot
                print(
                    "| Evaluator: Warning! Not enough valid data points (>=2) in recorder to save learning curve.",
                    flush=True,
                )
                return

            train_time = int(time.time() - self.start_time)
            total_step = int(valid_recorder[-1, 0])
            # Use max valid reward for title
            max_valid_r = valid_recorder[:, 1].max()
            fig_title = (
                f"step_time_maxR_{total_step}_{int(train_time)}_{max_valid_r:.3f}"
            )

            draw_learning_curve(
                recorder=valid_recorder,
                fig_title=fig_title,
                save_path=f"{self.cwd}/LearningCurve.jpg",
            )
            # Save the original recorder (including invalid steps) only if plotting was successful
            np.save(
                self.recorder_path, recorder
            )  # Save original recorder regardless of plotting success? Debatable. Saving here.
            print(
                f"| Evaluator: Saved learning curve to {self.cwd}/LearningCurve.jpg",
                flush=True,
            )

        except Exception as e:
            import traceback

            print(
                f"| Evaluator: Error drawing or saving learning curve: {e}", flush=True
            )
            print(traceback.format_exc(), flush=True)


"""util"""


def get_rewards_and_steps(
    env, actor, if_render: bool = False
) -> Tuple[float, int] | None:  # Added Optional return type
    """Run one episode in the environment and return cumulative reward and steps."""
    max_step = getattr(
        env, "max_step", getattr(env, "_max_episode_steps", 10000)
    )  # Try both attributes
    if max_step is None or max_step <= 0:
        max_step = 10000  # Default if invalid

    device = next(actor.parameters()).device  # Assume actor has parameters

    try:
        # Newer gym environments return obs, info
        # Older gym environments return obs
        reset_output = env.reset()
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            state, info_dict = reset_output
        else:
            state = reset_output
            info_dict = {}  # Initialize empty info dict for older gym envs
        # Ensure state is numpy array for consistency before converting to tensor
        if not isinstance(state, np.ndarray):
            # Attempt conversion if possible (e.g., list), warn otherwise
            try:
                state = np.array(state, dtype=np.float32)
            except Exception as conv_e:
                print(
                    f"| get_rewards_and_steps: WARNING - could not convert initial state to numpy array: {conv_e}",
                    flush=True,
                )
                # Try to proceed, might fail later
    except Exception as e:
        import traceback

        print(f"| get_rewards_and_steps: ERROR during env.reset(): {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return None  # Return None on error

    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    terminated = False
    truncated = False
    try:
        with th.no_grad():  # Ensure gradients are not calculated during evaluation
            for episode_steps in range(max_step):
                try:
                    # Ensure state is float32 numpy array before converting
                    if not isinstance(state, np.ndarray) or state.dtype != np.float32:
                        state = np.array(state, dtype=np.float32)
                    tensor_state = th.as_tensor(
                        state, dtype=th.float32, device=device
                    ).unsqueeze(0)
                    tensor_action = actor(tensor_state)
                    action = tensor_action.detach().cpu().numpy()[0]
                except Exception as e:
                    print(
                        f"| get_rewards_and_steps: ERROR during action selection at step {episode_steps}: {e}",
                        flush=True,
                    )
                    # Optionally return None or try to continue with a default action? Returning None is safer.
                    return None

                try:
                    # Newer gym returns obs, reward, terminated, truncated, info
                    # Older gym returns obs, reward, done, info
                    step_output = env.step(action)
                    if len(step_output) == 5:
                        state, reward, terminated, truncated, info_dict = step_output
                        done = terminated or truncated
                    elif len(step_output) == 4:
                        state, reward, done, info_dict = step_output
                        terminated = done  # Assume 'done' means terminated in older gym for consistency
                        truncated = False  # Older gym doesn't have truncated explicitly
                    else:
                        print(
                            f"| get_rewards_and_steps: WARNING - Unexpected number of return values from env.step(): {len(step_output)}",
                            flush=True,
                        )
                        # Try to unpack assuming standard order if possible, otherwise fail
                        if len(step_output) >= 4:
                            state, reward, done, info_dict = step_output[:4]
                            terminated = done
                            truncated = False
                        else:
                            return None  # Cannot proceed

                    cumulative_returns += reward

                    if if_render:
                        # Handle different render modes
                        render_mode = getattr(env, "render_mode", None)
                        if render_mode == "human":
                            env.render()
                        elif hasattr(env, "render") and not render_mode:  # Older gym?
                            env.render()

                    if done:
                        break
                except Exception as e:
                    import traceback

                    print(
                        f"| get_rewards_and_steps: ERROR during env.step() at step {episode_steps}: {e}",
                        flush=True,
                    )
                    print(traceback.format_exc(), flush=True)
                    return None  # Return None on error during step
            else:  # Loop completed without break (terminated/truncated)
                # This means max_step was reached
                truncated = True  # Set truncated flag if max_step is reached
                print(
                    f"| get_rewards_and_steps: WARNING. Episode reached max_step {max_step}",
                    flush=True,
                )

        # Use final cumulative returns from env info dict if available (e.g., from Monitor wrapper)
        # Look for 'episode' key in the final info_dict
        final_info = info_dict if done else {}  # Use last info if done, otherwise empty
        final_cumulative_returns = final_info.get("episode", {}).get(
            "r", cumulative_returns
        )
        # Use steps from info dict if available, otherwise use counted steps + 1
        final_steps = final_info.get("episode", {}).get("l", episode_steps + 1)

        return float(final_cumulative_returns), int(
            final_steps
        )  # Cast to float/int for consistency

    except Exception as e:
        import traceback

        print(f"| get_rewards_and_steps: UNEXPECTED ERROR: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return None


def get_cumulative_rewards_and_step_from_vec_env(
    env, actor
) -> List[Tuple[float, int]] | None:  # Added Optional return type
    """Get cumulative returns and step counts from a vectorized environment."""
    device = getattr(
        actor, "device", getattr(env, "device", "cpu")
    )  # Get device preferably from actor
    env_num = getattr(env, "num_envs", 1)
    max_step = getattr(env, "max_step", getattr(env, "_max_episode_steps", 10000))
    if max_step is None or max_step <= 0:
        max_step = 10000
    if env_num <= 0:
        env_num = 1

    returns_step_list = []

    try:
        # Reset env - state shape should be (env_num, obs_dim)
        reset_output = env.reset()
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            state, info_dict_array = reset_output  # Newer gym VecEnv
        else:
            state = reset_output  # Older gym VecEnv
            info_dict_array = [{} for _ in range(env_num)]  # Placeholder

        # Ensure state is a tensor on the correct device
        if isinstance(state, np.ndarray):
            state = th.as_tensor(state, dtype=th.float32, device=device)
        elif isinstance(state, th.Tensor):
            if state.device != device:
                state = state.to(device)
            if state.dtype != th.float32:
                state = state.to(th.float32)  # Ensure correct dtype
        else:
            print(
                "| vec_env: ERROR - Initial state is not Tensor or ndarray.", flush=True
            )
            return None

        # Check initial state shape
        if state.shape[0] != env_num:
            print(
                f"| vec_env: ERROR - Initial state batch size ({state.shape[0]}) != num_envs ({env_num}).",
                flush=True,
            )
            return None

        # Track ongoing returns and steps per environment
        ep_returns = th.zeros(env_num, dtype=th.float32, device=device)
        ep_steps = th.zeros(env_num, dtype=th.int32, device=device)
        # active = th.ones(env_num, dtype=th.bool, device=device) # Not needed if using auto-resetting VecEnv

        with th.no_grad():  # Ensure no gradients during evaluation
            for t in range(max_step):
                # Get action - state should be (env_num, obs_dim) -> action (env_num, act_dim) or (env_num,)
                try:
                    action = actor(state)
                except Exception as e:
                    print(
                        f"| vec_env: ERROR during action selection at step {t}: {e}",
                        flush=True,
                    )
                    return None  # Cannot proceed without action

                # Step env - expect state(N,obs), reward(N,), term(N,), trunc(N,), info(list[dict, N])
                try:
                    step_output = env.step(action)
                    if len(step_output) == 5:  # Newer gym VecEnv standard
                        state, reward, terminated, truncated, info_dict_array = (
                            step_output
                        )
                        done = th.logical_or(terminated, truncated)
                    elif len(step_output) == 4:  # Older gym VecEnv?
                        state, reward, done, info_dict_array = step_output
                        terminated = done  # Assume done is terminated
                        truncated = th.zeros_like(
                            done, dtype=th.bool
                        )  # Assume no truncation
                    else:
                        print(
                            f"| vec_env: WARNING - Unexpected number of return values from env.step(): {len(step_output)}",
                            flush=True,
                        )
                        return None  # Cannot reliably proceed

                    # Ensure outputs are tensors on the correct device
                    if isinstance(state, np.ndarray):
                        state = th.as_tensor(state, dtype=th.float32, device=device)
                    elif state.device != device or state.dtype != th.float32:
                        state = state.to(device, dtype=th.float32)
                    if isinstance(reward, np.ndarray):
                        reward = th.as_tensor(reward, dtype=th.float32, device=device)
                    elif reward.device != device:
                        reward = reward.to(device)
                    if isinstance(terminated, np.ndarray):
                        terminated = th.as_tensor(
                            terminated, dtype=th.bool, device=device
                        )
                    elif terminated.device != device:
                        terminated = terminated.to(device)
                    if isinstance(truncated, np.ndarray):
                        truncated = th.as_tensor(
                            truncated, dtype=th.bool, device=device
                        )
                    elif truncated.device != device:
                        truncated = truncated.to(device)
                    if isinstance(done, np.ndarray):
                        done = th.as_tensor(done, dtype=th.bool, device=device)
                    elif done.device != device:
                        done = done.to(device)

                    # Check shapes after step
                    if (
                        state.shape[0] != env_num
                        or reward.shape != (env_num,)
                        or done.shape != (env_num,)
                    ):
                        print(
                            f"| vec_env: ERROR - Shape mismatch after step {t}. State:{state.shape}, Reward:{reward.shape}, Done:{done.shape}",
                            flush=True,
                        )
                        return None

                    # Update ongoing returns and steps
                    ep_returns += reward
                    ep_steps += 1

                    # Process finished environments using info dict (robust way)
                    # Newer gym VecEnv puts final info in info['_final_observation'] and info['_final_info']
                    # Older gym might put it directly in info or use Monitor wrapper style
                    finished_mask = (
                        done.cpu().numpy()
                    )  # Use CPU numpy mask for easier indexing

                    if np.any(finished_mask):
                        if isinstance(
                            info_dict_array, dict
                        ):  # Check if using newer VecEnv info structure
                            final_infos = info_dict_array.get(
                                "_final_info", [None] * env_num
                            )  # Get list of final infos
                            valid_final_infos = info_dict_array.get(
                                "final_info", [None] * env_num
                            )  # Fallback? SB3 style?

                            for i in np.where(finished_mask)[0]:
                                final_info = (
                                    final_infos[i]
                                    if final_infos[i] is not None
                                    else valid_final_infos[i]
                                )
                                if final_info is not None and "episode" in final_info:
                                    final_return = final_info["episode"]["r"]
                                    final_steps = final_info["episode"]["l"]
                                    returns_step_list.append(
                                        (float(final_return), int(final_steps))
                                    )
                                else:
                                    # If Monitor wrapper info not found, use accumulated values just before reset
                                    # Note: VecEnv auto-resets, so state[i] is already the *new* state.
                                    # We use the accumulated ep_returns[i] and ep_steps[i] *before* they reset below.
                                    # This might be slightly off if reset happens mid-step logic, but often best available.
                                    returns_step_list.append(
                                        (
                                            float(ep_returns[i].item()),
                                            int(ep_steps[i].item()),
                                        )
                                    )

                        else:  # Assume older style or simple list of dicts
                            for i in np.where(finished_mask)[0]:
                                info = (
                                    info_dict_array[i]
                                    if i < len(info_dict_array)
                                    else {}
                                )
                                if (
                                    "episode" in info
                                ):  # Check for Monitor wrapper style info
                                    final_return = info["episode"]["r"]
                                    final_steps = info["episode"]["l"]
                                    returns_step_list.append(
                                        (float(final_return), int(final_steps))
                                    )
                                else:
                                    # Fallback to accumulated values
                                    returns_step_list.append(
                                        (
                                            float(ep_returns[i].item()),
                                            int(ep_steps[i].item()),
                                        )
                                    )

                        # Reset accumulated returns and steps for finished envs
                        ep_returns[done] = 0
                        ep_steps[done] = 0

                except Exception as e:
                    import traceback

                    print(
                        f"| vec_env: ERROR during env.step() processing at step {t}: {e}",
                        flush=True,
                    )
                    print(traceback.format_exc(), flush=True)
                    return None  # Cannot reliably continue

            # After loop, handle environments that were still active (hit max_step)
            # For auto-resetting VecEnvs, this shouldn't happen unless max_step is very low
            # If using non-auto-resetting, need to record final accumulated values
            # Assuming auto-resetting for now. If issues arise, add handling for non-resetting case.
            # Check if any envs finished exactly on the last step
            # (This logic is now handled within the loop)

    except Exception as e:
        import traceback

        print(
            f"| vec_env: UNEXPECTED ERROR during vectorized evaluation: {e}", flush=True
        )
        print(traceback.format_exc(), flush=True)
        return None  # Return None on error

    return returns_step_list


def draw_learning_curve(
    recorder: np.ndarray = None,
    fig_title: str = "learning_curve",
    save_path: str = "learning_curve.jpg",
):
    # Check if recorder is provided and has enough data
    if (
        recorder is None
        or not isinstance(recorder, np.ndarray)
        or recorder.ndim != 2
        or recorder.shape[0] < 2
    ):
        print(
            "| draw_learning_curve: Warning! Invalid or insufficient data provided for plotting.",
            flush=True,
        )
        return

    # Check recorder dimensions - expecting at least 6 columns for standard plot
    min_expected_cols = 6
    actual_cols = recorder.shape[1]
    if actual_cols < 2:  # Need at least steps and avg_r
        print(
            f"| draw_learning_curve: Warning! Recorder has only {actual_cols} columns. Need at least 2 (steps, avg_r).",
            flush=True,
        )
        return

    # Define column indices safely
    col_steps = 0
    col_avg_r = 1
    col_std_r = 2 if actual_cols > 2 else -1  # Use -1 if column doesn't exist
    col_exp_r = 3 if actual_cols > 3 else -1
    col_obj_c = 4 if actual_cols > 4 else -1
    col_obj_a = 5 if actual_cols > 5 else -1
    other_cols_start = 6 if actual_cols > 6 else -1

    # Extract data, handling potential errors if data isn't numeric
    try:
        steps = recorder[:, col_steps].astype(float)
        r_avg = recorder[:, col_avg_r].astype(float)
        r_std = (
            recorder[:, col_std_r].astype(float)
            if col_std_r != -1
            else np.zeros_like(r_avg)
        )
        r_exp = (
            recorder[:, col_exp_r].astype(float)
            if col_exp_r != -1
            else np.zeros_like(r_avg)
        )
        obj_c = (
            recorder[:, col_obj_c].astype(float)
            if col_obj_c != -1
            else np.zeros_like(r_avg)
        )
        obj_a = (
            recorder[:, col_obj_a].astype(float)
            if col_obj_a != -1
            else np.zeros_like(r_avg)
        )
    except ValueError as e:
        print(
            f"| draw_learning_curve: Error converting recorder data to float: {e}. Skipping plot.",
            flush=True,
        )
        return

    """plot subplots"""
    try:
        import matplotlib as mpl

        mpl.use("Agg")  # Use Agg backend for non-interactive plotting
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(
            2, figsize=(12, 8), sharex=True
        )  # Share x-axis, adjusted figure size

        # --- Top Subplot (Rewards) ---
        ax00 = axs[0]
        color0 = "lightcoral"
        ax00.set_ylabel("Episode Return", color=color0)
        (line0,) = ax00.plot(
            steps, r_avg, label="Avg Episode Return", color=color0, linewidth=2
        )
        if col_std_r != -1:
            ax00.fill_between(
                steps,
                r_avg - r_std,
                r_avg + r_std,
                facecolor=color0,
                alpha=0.2,
                label="Std Dev",
            )
        ax00.tick_params(axis="y", labelcolor=color0)
        ax00.grid(True, linestyle="--", alpha=0.6)

        # Plot explore reward on twin axis if available
        if col_exp_r != -1:
            ax01 = ax00.twinx()
            color01 = "darkcyan"
            ax01.set_ylabel("Explore AvgReward (Smoothed)", color=color01)
            # Simple moving average to smooth explore reward (window size 5)
            window_size = min(5, len(r_exp) // 2)  # Adjust window if few points
            if window_size > 0:
                exp_r_smooth = np.convolve(
                    r_exp, np.ones(window_size) / window_size, mode="valid"
                )
                # Adjust steps for smoothed data
                steps_smooth = steps[
                    window_size // 2 : len(steps) - (window_size - 1) // 2
                ][
                    : len(exp_r_smooth)
                ]  # Center alignment approx
                if len(steps_smooth) == len(exp_r_smooth):  # Ensure lengths match
                    (line1,) = ax01.plot(
                        steps_smooth,
                        exp_r_smooth,
                        color=color01,
                        alpha=0.7,
                        linestyle=":",
                        label="Smoothed Explore Reward",
                    )
                    ax01.tick_params(axis="y", labelcolor=color01)
                    # Combine legends from both axes
                    lines, labels = ax00.get_legend_handles_labels()
                    lines2, labels2 = ax01.get_legend_handles_labels()
                    ax01.legend(lines + lines2, labels + labels2, loc="upper left")
                else:
                    print(
                        "| draw_learning_curve: Warning! Smoothed explore reward length mismatch. Skipping plot.",
                        flush=True,
                    )
                    ax00.legend(loc="upper left")  # Legend for main axis only
            else:
                ax00.legend(loc="upper left")  # Legend for main axis only
        else:
            ax00.legend(loc="upper left")  # Legend for main axis only

        # --- Bottom Subplot (Objectives) ---
        ax10 = axs[1]
        color10 = "royalblue"
        ax10.set_xlabel("Total Steps")
        ax10.grid(True, linestyle="--", alpha=0.6)
        lines10, labels10 = [], []

        # Plot Actor Objective (objA) if available
        if col_obj_a != -1:
            ax10.set_ylabel("Actor Objective (objA)", color=color10)
            (line_a,) = ax10.plot(
                steps,
                obj_a,
                label="Actor Objective (objA)",
                color=color10,
                linewidth=1.5,
            )
            ax10.tick_params(axis="y", labelcolor=color10)
            lines10.append(line_a)
            labels10.append(line_a.get_label())

        # Plot other objectives if they exist
        if other_cols_start != -1:
            for plot_i in range(other_cols_start, actual_cols):
                try:
                    other_data = recorder[:, plot_i].astype(float)
                    # Simple check to avoid plotting constant zero columns unless it's objA/objC
                    if np.any(other_data):
                        (line_other,) = ax10.plot(
                            steps,
                            other_data,
                            label=f"Other {plot_i}",
                            color="grey",
                            alpha=0.6,
                            linestyle="-.",
                            linewidth=1,
                        )
                        lines10.append(line_other)
                        labels10.append(line_other.get_label())
                except ValueError:
                    print(
                        f"| draw_learning_curve: Warning! Could not plot additional column {plot_i} due to non-numeric data.",
                        flush=True,
                    )
                except IndexError:
                    break  # Stop if index goes out of bounds

        # Plot Critic Loss (objC) on twin axis if available and objA was plotted
        if col_obj_c != -1:
            ax11 = ax10.twinx()
            color11 = "forestgreen"  # Changed color
            ax11.set_ylabel("Critic Loss (objC)", color=color11)
            # Use fill_between from 0 for loss
            ax11.fill_between(
                steps,
                0,
                obj_c,
                facecolor=color11,
                alpha=0.2,
                label="Critic Loss (objC)",
            )
            ax11.tick_params(axis="y", labelcolor=color11)
            # Combine legends for bottom plot
            lines11, labels11 = ax11.get_legend_handles_labels()
            ax11.legend(
                lines10 + lines11,
                labels10 + labels11,
                loc="center left",
                bbox_to_anchor=(1.15, 0.5),
            )  # Adjust legend position
        elif lines10:  # If only objA etc. were plotted
            ax10.legend(lines10, labels10, loc="center left", bbox_to_anchor=(1.1, 0.5))

        """plot save"""
        fig.suptitle(fig_title, fontsize=16)
        fig.tight_layout(
            rect=[0, 0, 0.9, 0.96]
        )  # Adjust layout rect to make space for legend and title
        plt.savefig(save_path, dpi=150, bbox_inches="tight")  # Use bbox_inches='tight'
        plt.close(fig)  # Close the specific figure instance

    except ImportError:
        print(
            "| draw_learning_curve: Warning! Matplotlib not found. Skipping plot generation.",
            flush=True,
        )
    except Exception as e:
        import traceback

        print(f"| draw_learning_curve: Error during plotting: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        # Ensure figure is closed even if error occurs during plotting
        if "plt" in locals() and plt.get_fignums():
            plt.close("all")


# Keep demo/run functions as they were, assuming they are not the source of the main error
# Add imports needed by these functions if not already present globally
import sys
import gym

# Note: The following imports depend on the ElegantRL structure and might need adjustment
try:
    from elegantrl.agents.AgentPPO import AgentPPO  # Example agent
    from elegantrl.train.config import Config, build_env
    from elegantrl.envs.CustomGymEnv import PendulumEnv  # Example custom env
except ImportError as e:
    print(
        f"| WARNING: Could not import ElegantRL components for demo functions: {e}",
        flush=True,
    )

    # Define dummy classes/functions if needed for the script to not crash immediately
    class AgentPPO:
        pass

    class Config:
        pass

    def build_env(env_class, env_args):
        return None

    class PendulumEnv:
        pass


def demo_evaluator_actor_pth():
    print("| Running demo_evaluator_actor_pth...", flush=True)
    gpu_id = -1  # Use CPU for demo

    # Check if AgentPPO was imported
    if "AgentPPO" not in globals() or AgentPPO is None:
        print(
            "| demo_evaluator_actor_pth: AgentPPO not available. Skipping.", flush=True
        )
        return -np.inf, 0

    agent_class = AgentPPO

    try:
        env_class = gym.make
        env_args = {"id": "LunarLanderContinuous-v2"}  # Use 'id' for gym.make
        # Create a dummy env to get dims if build_env fails
        try:
            _dummy_env = env_class(env_args["id"])
            state_dim = _dummy_env.observation_space.shape[0]
            action_dim = _dummy_env.action_space.shape[0]
            max_step = getattr(_dummy_env, "_max_episode_steps", 1000)
            _dummy_env.close()
        except Exception:
            print(
                "| demo_evaluator_actor_pth: Could not create dummy env for dims. Using defaults.",
                flush=True,
            )
            state_dim = 8
            action_dim = 2
            max_step = 1000

        full_env_args = {
            "num_envs": 1,
            "env_name": "LunarLanderContinuous-v2",
            "max_step": max_step,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "if_discrete": False,
            "target_return": 200,
            "id": "LunarLanderContinuous-v2",
        }
    except Exception as e:
        print(
            f"| demo_evaluator_actor_pth: Error setting up env args: {e}. Skipping.",
            flush=True,
        )
        return -np.inf, 0

    eval_times = 4
    net_dim = 2**7  # Example network dimension

    """init"""
    # Check if Config and build_env were imported
    if (
        "Config" not in globals()
        or "build_env" not in globals()
        or Config is None
        or build_env is None
    ):
        print(
            "| demo_evaluator_actor_pth: Config or build_env not available. Skipping.",
            flush=True,
        )
        return -np.inf, 0

    try:
        args = Config(
            agent_class=agent_class, env_class=env_class, env_args=full_env_args
        )
        # Pass state/action dim explicitly if build_env needs them
        args.state_dim = state_dim
        args.action_dim = action_dim
        env = build_env(env_class=args.env_class, env_args=args.env_args)
        if env is None:
            raise ValueError("build_env returned None")
        # If build_env doesn't set max_step, set it manually
        if not hasattr(env, "max_step"):
            env.max_step = max_step

        act = agent_class(net_dim, state_dim, action_dim, gpu_id=gpu_id, args=args).act
    except Exception as e:
        import traceback

        print(
            f"| demo_evaluator_actor_pth: Error initializing agent/env: {e}", flush=True
        )
        print(traceback.format_exc(), flush=True)
        if "env" in locals() and hasattr(env, "close"):
            env.close()
        return -np.inf, 0

    # actor_path = './LunarLanderContinuous-v2_PPO_0/actor.pt' # Example path
    # if os.path.exists(actor_path):
    #     print(f"| Loading actor from: {actor_path}", flush=True)
    #     try:
    #         act.load_state_dict(th.load(actor_path, map_location=lambda storage, loc: storage))
    #     except Exception as e:
    #          print(f"| Failed to load actor: {e}", flush=True)
    # else:
    #      print(f"| Actor path not found: {actor_path}. Using untrained actor.", flush=True)

    """evaluate"""
    r_s_list = []
    print(f"| Evaluating {eval_times} times...", flush=True)
    for i in range(eval_times):
        result = get_rewards_and_steps(env, act)
        if result is not None:
            r_s_list.append(result)
            print(
                f"| Eval {i+1}/{eval_times}: Reward={result[0]:.2f}, Steps={result[1]}",
                flush=True,
            )
        else:
            print(f"| Eval {i+1}/{eval_times}: Failed.", flush=True)

    # Close env
    if hasattr(env, "close"):
        env.close()

    if not r_s_list:
        print(
            "| demo_evaluator_actor_pth: No valid evaluation results obtained.",
            flush=True,
        )
        return -np.inf, 0

    r_s_ary = np.array(r_s_list, dtype=np.float32)
    r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step

    print(f"| Final Avg Reward: {r_avg:.2f}, Avg Steps: {s_avg:.1f}", flush=True)
    return r_avg, s_avg


def demo_evaluate_actors(
    dir_path: str, gpu_id: int, agent, env_args: dict, eval_times=2, net_dim=128
):
    print(f"\n| Running demo_evaluate_actors in directory: {dir_path}", flush=True)
    # Check if AgentPPO was imported
    if "AgentPPO" not in globals() or AgentPPO is None:
        print(
            "| demo_evaluate_actors: Agent class not available. Skipping.", flush=True
        )
        return np.empty((0, 3))

    # Setup Env
    try:
        env_class = gym.make
        # Ensure essential args are present
        env_id = env_args.get("id", env_args.get("env_name"))
        if not env_id:
            raise ValueError("Missing env id/name in env_args")

        # Create dummy env for dims
        try:
            _dummy_env = env_class(env_id)
            state_dim = _dummy_env.observation_space.shape[0]
            action_dim = (
                _dummy_env.action_space.shape[0]
                if isinstance(_dummy_env.action_space, gym.spaces.Box)
                else _dummy_env.action_space.n
            )
            max_step = getattr(_dummy_env, "_max_episode_steps", 1000)
            _dummy_env.close()
        except Exception as e:
            print(
                f"| demo_evaluate_actors: Could not create dummy env '{env_id}' for dims: {e}. Using defaults.",
                flush=True,
            )
            state_dim = env_args.get("state_dim", None)
            action_dim = env_args.get("action_dim", None)
            max_step = env_args.get("max_step", 1000)
            if state_dim is None or action_dim is None:
                print(
                    "| demo_evaluate_actors: Missing state_dim or action_dim. Cannot proceed.",
                    flush=True,
                )
                return np.empty((0, 3))

        full_env_args = env_args.copy()
        full_env_args.update(
            {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_step": max_step,
                "id": env_id,  # Ensure id is present
            }
        )

        if (
            "Config" not in globals()
            or "build_env" not in globals()
            or Config is None
            or build_env is None
        ):
            print(
                "| demo_evaluate_actors: Config or build_env not available. Skipping env creation.",
                flush=True,
            )
            return np.empty((0, 3))

        args_obj = Config(
            agent_class=agent, env_class=env_class, env_args=full_env_args
        )
        args_obj.state_dim = state_dim  # Pass dims explicitly
        args_obj.action_dim = action_dim
        env = build_env(env_class=env_class, env_args=args_obj.env_args)
        if env is None:
            raise ValueError("build_env returned None")
        if not hasattr(env, "max_step"):
            env.max_step = max_step  # Set max_step if needed

        # Init Actor (only the structure, weights loaded later)
        act = agent(net_dim, state_dim, action_dim, gpu_id=gpu_id, args=args_obj).act

    except Exception as e:
        import traceback

        print(
            f"| demo_evaluate_actors: Error initializing env/actor structure: {e}",
            flush=True,
        )
        print(traceback.format_exc(), flush=True)
        if "env" in locals() and hasattr(env, "close"):
            env.close()
        return np.empty((0, 3))

    """evaluate saved models"""
    step_epi_r_s_ary = []
    try:
        # Find actor files like actor*.pt
        actor_files = [
            name
            for name in os.listdir(dir_path)
            if name.startswith("actor")
            and name.endswith(".pt")
            and os.path.isfile(os.path.join(dir_path, name))
        ]
        actor_files.sort()  # Sort for consistent order
        print(f"| Found {len(actor_files)} actor files to evaluate.", flush=True)

        for act_name in actor_files:
            act_path = os.path.join(dir_path, act_name)
            print(f"|-- Evaluating: {act_name}", end=" ", flush=True)
            try:
                # Load state dict
                map_location = f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"
                act.load_state_dict(th.load(act_path, map_location=map_location))
            except Exception as e:
                print(f"Failed loading: {e}", flush=True)
                continue  # Skip this actor

            # Evaluate multiple times
            r_s_list = []
            for i in range(eval_times):
                result = get_rewards_and_steps(env, act)
                if result is not None:
                    r_s_list.append(result)
                # else: # Optional: print failure for each run
                #      print(f"\n   Eval run {i+1} failed.", flush=True)

            if not r_s_list:
                print(f"No valid runs.", flush=True)
                continue  # Skip if all runs failed

            r_s_ary = np.array(r_s_list, dtype=np.float32)
            r_avg, s_avg = r_s_ary.mean(axis=0)
            print(
                f"AvgR={r_avg:.2f}, AvgS={s_avg:.1f} ({len(r_s_list)} valid runs)",
                flush=True,
            )

            # Extract step number from filename (e.g., actor__000123456_...pt or actor__000123456.pt)
            try:
                parts = act_name[:-3].split("_")  # Remove .pt and split
                # Find the first long digit sequence, likely the step count
                step_str = "0"
                for part in reversed(parts):
                    if part.isdigit() and len(part) > 5:  # Heuristic for step count
                        step_str = part
                        break
                step = int(step_str)
            except (ValueError, IndexError):
                print(f"| Could not parse step from '{act_name}'. Using 0.", flush=True)
                step = 0

            step_epi_r_s_ary.append((step, r_avg, s_avg))

    except Exception as e:
        import traceback

        print(
            f"\n| demo_evaluate_actors: Error during evaluation loop: {e}", flush=True
        )
        print(traceback.format_exc(), flush=True)

    finally:
        # Ensure env is closed
        if "env" in locals() and hasattr(env, "close"):
            env.close()
            print("| Environment closed.", flush=True)

    if not step_epi_r_s_ary:
        print("| demo_evaluate_actors: No actors evaluated successfully.", flush=True)
        return np.empty((0, 3))

    step_epi_r_s_ary = np.array(step_epi_r_s_ary, dtype=np.float32)
    # Sort by step number
    step_epi_r_s_ary = step_epi_r_s_ary[step_epi_r_s_ary[:, 0].argsort()]
    return step_epi_r_s_ary


def demo_load_pendulum_and_render():
    print("\n| Running demo_load_pendulum_and_render...", flush=True)
    gpu_id = -1  # Use CPU for demo

    # Check dependencies
    if (
        "AgentPPO" not in globals()
        or "Config" not in globals()
        or "build_env" not in globals()
        or AgentPPO is None
        or Config is None
        or build_env is None
    ):
        print(
            "| demo_load_pendulum_and_render: Missing dependencies (AgentPPO, Config, build_env). Skipping.",
            flush=True,
        )
        return

    agent_class = AgentPPO
    env = None  # Initialize env to None for finally block

    try:
        # Try custom env first, fallback to gym
        try:
            if "PendulumEnv" not in globals() or PendulumEnv is None:
                raise ImportError
            env_class = PendulumEnv
            env_args = {"env_name": "Pendulum-v1", "max_step": 200}
            print("| Using custom PendulumEnv.", flush=True)
        except ImportError:
            print(
                "| Custom PendulumEnv not found. Using gym.make('Pendulum-v1').",
                flush=True,
            )
            env_class = gym.make
            env_args = {
                "id": "Pendulum-v1",
                "render_mode": "human",
                "max_episode_steps": 200,
            }  # Add render_mode

        # Create dummy env for dims
        try:
            _dummy_env = env_class(env_args.get("id", "Pendulum-v1"))
            state_dim = _dummy_env.observation_space.shape[0]
            action_dim = _dummy_env.action_space.shape[0]
            max_step = getattr(_dummy_env, "_max_episode_steps", 200)
            _dummy_env.close()
        except Exception as e:
            print(
                f"| demo_load_pendulum_and_render: Could not create dummy env for dims: {e}. Using defaults.",
                flush=True,
            )
            state_dim = 3
            action_dim = 1
            max_step = 200

        full_env_args = env_args.copy()
        full_env_args.update(
            {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_step": max_step,
                "id": env_args.get("id", "Pendulum-v1"),  # Ensure id
            }
        )

        # Actor Path (adjust if needed)
        actor_path = "./Pendulum-v1_PPO_0/actor.pt"
        net_dim = 2**7

        """init"""
        args = Config(
            agent_class=agent_class, env_class=env_class, env_args=full_env_args
        )
        args.state_dim = state_dim
        args.action_dim = action_dim

        # Build real env for rendering
        render_env_args = full_env_args.copy()
        render_env_args["render_mode"] = "human"  # Ensure render mode for gym.make
        env = build_env(env_class=env_class, env_args=render_env_args)
        if env is None:
            raise ValueError("build_env returned None")
        if not hasattr(env, "max_step"):
            env.max_step = max_step

        act = agent_class(net_dim, state_dim, action_dim, gpu_id=gpu_id, args=args).act

        if not os.path.exists(actor_path):
            print(
                f"| Actor path not found: {actor_path}. Skipping evaluation and render.",
                flush=True,
            )
            return

        print(f"| Loading actor from: {actor_path}", flush=True)
        act.load_state_dict(
            th.load(actor_path, map_location=lambda storage, loc: storage)
        )
        act.eval()  # Set to evaluation mode

        """render"""
        max_render_steps = getattr(env, "max_step", max_step)
        device = next(act.parameters()).device

        state, info = env.reset()
        steps_count = 0
        returns = 0.0
        terminated = False
        truncated = False
        print("| Starting render loop...", flush=True)

        with th.no_grad():
            for steps_count in range(max_render_steps):
                if not isinstance(state, np.ndarray) or state.dtype != np.float32:
                    state = np.array(state, dtype=np.float32)
                s_tensor = th.as_tensor(
                    state, dtype=th.float32, device=device
                ).unsqueeze(0)
                a_tensor = act(s_tensor)
                # Pendulum action space requires scaling (typically * max_torque, which is 2)
                action = a_tensor.detach().cpu().numpy()[0] * 2.0

                step_output = env.step(action)
                if len(step_output) == 5:
                    state, reward, terminated, truncated, info = step_output
                elif len(step_output) == 4:
                    state, reward, terminated, info = step_output
                    truncated = False  # Older gym
                else:
                    raise ValueError("Unexpected step output length")

                returns += reward
                env.render()  # Call render

                if terminated or truncated:
                    break
            else:
                print(
                    f"| Render loop reached max_steps: {max_render_steps}", flush=True
                )

        final_returns = info.get("episode", {}).get(
            "r", returns
        )  # Get final return from info if available
        final_steps = info.get("episode", {}).get("l", steps_count + 1)

        print(f"\n| Render finished.", flush=True)
        print(f"| Final Cumulative Return: {final_returns:.3f}", flush=True)
        print(f"| Episode Steps: {final_steps}", flush=True)

    except Exception as e:
        import traceback

        print(f"\n| demo_load_pendulum_and_render: ERROR: {e}", flush=True)
        print(traceback.format_exc(), flush=True)

    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
            print("| Environment closed.", flush=True)


def run():
    print("\n| Running combined evaluation and plotting (run function)...", flush=True)
    # --- Configuration ---
    flag_id = (
        0  # Selects which env_args to use (0 for LunarLander, 1 for BipedalWalker)
    )
    default_gpu_id = -1  # Default to CPU if selection fails
    default_net_dim = 256  # Default network dim for loading actors

    # Check dependencies
    if "AgentPPO" not in globals() or AgentPPO is None:
        print("| run: AgentPPO class not available. Exiting.", flush=True)
        return
    agent_class = AgentPPO

    # --- Environment Setup ---
    env_args_list = [
        # LunarLanderContinuous-v2
        {
            "id": "LunarLanderContinuous-v2",
            "env_name": "LunarLanderContinuous-v2",
            "num_envs": 1,
            "target_return": 200,
            "eval_times": 16,
        },
        # BipedalWalker-v3
        {
            "id": "BipedalWalker-v3",
            "env_name": "BipedalWalker-v3",
            "num_envs": 1,
            "target_return": 300,
            "eval_times": 8,
        },
    ]
    try:
        selected_env_args = env_args_list[flag_id]
        env_name = selected_env_args["env_name"]
    except IndexError:
        print(
            f"| run: Error! flag_id {flag_id} is out of range for env_args list. Exiting.",
            flush=True,
        )
        return

    # --- GPU Setup ---
    gpu_list = [-1, 0, 1, 2, 3]  # Example list of available GPUs/CPU
    try:
        gpu_id = gpu_list[flag_id + 1]  # Offset example, adjust as needed
    except IndexError:
        print(
            f"| run: Warning! flag_id {flag_id} leads to out of range GPU selection. Using default GPU {default_gpu_id}.",
            flush=True,
        )
        gpu_id = default_gpu_id

    print("| Run Settings:", flush=True)
    print(f"| Environment: {env_name}", flush=True)
    print(f"| Agent Class: {agent_class.__name__}", flush=True)
    print(f"| GPU ID: {gpu_id}", flush=True)
    print(f'| Eval Times per Actor: {selected_env_args["eval_times"]}', flush=True)
    print(f"| Net Dim: {default_net_dim}", flush=True)

    # --- Evaluate Saved Models ---
    all_results = []
    cwd_path = "."
    # Find directories matching env_name convention (e.g., LunarLanderContinuous-v2_PPO_*)
    potential_dirs = [
        name
        for name in os.listdir(cwd_path)
        if env_name in name and os.path.isdir(os.path.join(cwd_path, name))
    ]
    print(
        f"\n| Found {len(potential_dirs)} potential model directories: {potential_dirs}",
        flush=True,
    )

    for dir_name in potential_dirs:
        dir_path = os.path.join(cwd_path, dir_name)
        # Run evaluation for this directory
        results_ary = demo_evaluate_actors(
            dir_path,
            gpu_id,
            agent_class,
            selected_env_args,
            eval_times=selected_env_args["eval_times"],
            net_dim=default_net_dim,
        )

        if results_ary is not None and results_ary.shape[0] > 0:
            all_results.append(results_ary)
            # Optionally save results per directory
            # save_path = f"{dir_path}-step_epi_r_s_ary.txt"
            # np.savetxt(save_path, results_ary, fmt='%.4f')
            # print(f"| Saved results for {dir_name} to {save_path}", flush=True)
        else:
            print(f"| No results generated for directory: {dir_name}", flush=True)

    # --- Process and Plot Combined Results ---
    if not all_results:
        print(
            "\n| No evaluation results found across all directories. Cannot plot.",
            flush=True,
        )
        return

    # Combine results and sort by step
    combined_results = np.vstack(all_results)
    combined_results = combined_results[
        combined_results[:, 0].argsort()
    ]  # Sort by step (column 0)
    print(
        f"\n| Combined evaluation results shape: {combined_results.shape}", flush=True
    )
    if combined_results.shape[0] == 0:
        print("| No combined results to plot after merging.", flush=True)
        return

    # --- Plotting ---
    try:
        print("| Plotting combined results...", flush=True)
        # Aggregate/smooth data for plotting (e.g., average results within step ranges)
        plot_data = []
        unique_steps = np.unique(combined_results[:, 0])
        # Define step bins for aggregation (e.g., 20 bins)
        num_bins = min(50, len(unique_steps))  # Limit number of points on plot
        if num_bins <= 1:
            step_bins = unique_steps
        else:
            step_bins = np.linspace(
                unique_steps.min(), unique_steps.max(), num_bins + 1
            )

        for i in range(len(step_bins) - 1):
            low_step, high_step = step_bins[i], step_bins[i + 1]
            # Include points exactly at high_step in the current bin, except for the last bin
            mask = (combined_results[:, 0] >= low_step) & (
                combined_results[:, 0] < high_step
            )
            if i == len(step_bins) - 2:  # Include max step in last bin
                mask |= combined_results[:, 0] == high_step

            if not np.any(mask):
                continue

            bin_data = combined_results[mask]
            avg_step = bin_data[:, 0].mean()
            avg_reward = bin_data[:, 1].mean()
            std_reward = bin_data[:, 1].std()
            avg_ep_steps = bin_data[:, 2].mean()  # Average episode length in bin

            plot_data.append((avg_step, avg_reward, std_reward, avg_ep_steps))

        if not plot_data:
            print("| No data points after aggregation for plotting.", flush=True)
            return

        plot_data = np.array(plot_data)
        plot_steps = plot_data[:, 0]
        plot_rewards = plot_data[:, 1]
        plot_rewards_std = plot_data[:, 2]
        plot_ep_steps = plot_data[:, 3]

        # Create Plot
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, figsize=(12, 6))
        color_reward = "royalblue"
        color_steps = "lightcoral"
        plot_title = f"{env_name}_{agent_class.__name__}_Combined_Learning_Curve"

        # Plot Avg Reward + Std Dev Band
        ax.plot(
            plot_steps,
            plot_rewards,
            label="Avg Episode Return (Smoothed)",
            color=color_reward,
            marker="o",
            markersize=4,
            linestyle="-",
        )
        ax.fill_between(
            plot_steps,
            plot_rewards - plot_rewards_std,
            plot_rewards + plot_rewards_std,
            facecolor=color_reward,
            alpha=0.2,
            label="Std Dev Band",
        )
        ax.set_ylabel("Episode Return", color=color_reward)
        ax.tick_params(axis="y", labelcolor=color_reward)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlabel("Total Training Steps")
        ax.legend(loc="upper left")

        # Plot Avg Episode Steps on Twin Axis
        ax_twin = ax.twinx()
        ax_twin.plot(
            plot_steps,
            plot_ep_steps,
            label="Avg Episode Steps (Smoothed)",
            color=color_steps,
            marker="x",
            markersize=4,
            linestyle=":",
        )
        ax_twin.set_ylabel("Episode Steps", color=color_steps)
        ax_twin.tick_params(axis="y", labelcolor=color_steps)
        ax_twin.legend(loc="upper right")
        # ax_twin.set_ylim(bottom=0) # Optional: force step axis start at 0

        plt.title(plot_title)
        plot_save_path = f"{env_name}_{agent_class.__name__}_combined_curve.png"
        print(f"| Saving combined plot to: {plot_save_path}", flush=True)
        plt.savefig(plot_save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    except ImportError:
        print(
            "| run: Matplotlib not found. Skipping plotting combined results.",
            flush=True,
        )
    except Exception as e:
        import traceback

        print(f"| run: Error during combined results plotting: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        if "plt" in locals() and plt.get_fignums():
            plt.close("all")


if __name__ == "__main__":
    # Select which function to run when the script is executed
    # demo_evaluator_actor_pth()
    # demo_load_pendulum_and_render()
    run()  # Runs the combined evaluation and plotting
