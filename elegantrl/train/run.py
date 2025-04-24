import os
import time
import numpy as np
import torch as th
import multiprocessing as mp
from copy import deepcopy
from typing import List, Optional
from multiprocessing import Process, Pipe

from .config import Config
from .config import build_env
from .replay_buffer import ReplayBuffer
from .evaluator import Evaluator
from .evaluator import get_rewards_and_steps

if os.name == 'nt':  # if is WindowOS (Windows NT)
    """Fix bug about Anaconda in WindowOS
    OMP: Error #15: Initializing libIOmp5md.dll, but found libIOmp5md.dll already initialized.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''train'''


def train_agent(args: Config, if_single_process: bool = False):
    if if_single_process:
        print(f"| train_agent_single_process() with GPU_ID {args.gpu_id}", flush=True)
        train_agent_single_process(args)
    elif len(args.learner_gpu_ids) == 0:
        print(f"| train_agent_multiprocessing() with GPU_ID {args.gpu_id}", flush=True)
        train_agent_multiprocessing(args)
    elif len(args.learner_gpu_ids) != 0:
        print(f"| train_agent_multiprocessing_multi_gpu() with GPU_ID {args.learner_gpu_ids}", flush=True)
        train_agent_multiprocessing_multi_gpu(args)
    else:
        ValueError(f"| run.py train_agent: args.learner_gpu_ids = {args.learner_gpu_ids}")


def train_agent_single_process(args: Config):
    args.init_before_training()
    th.set_grad_enabled(False)

    '''init environment'''
    env = build_env(args.env_class, args.env_args, args.gpu_id)
    # <<< Ensure args.state_dim matches env.state_dim >>>
    if hasattr(env, 'state_dim') and args.state_dim != env.state_dim:
        print(f"[SingleProcess] Correcting args.state_dim from {args.state_dim} to env.state_dim: {env.state_dim}", flush=True)
        args.state_dim = env.state_dim

    '''init agent'''
    # Use the potentially corrected args.state_dim
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    if args.continue_train:
        agent.save_or_load_agent(args.cwd, if_save=False)

    '''init agent.last_state'''
    state, info_dict = env.reset()
    if args.num_envs == 1:
        assert state.shape == (env.state_dim,)
        assert isinstance(state, np.ndarray)
        state = th.tensor(state, dtype=th.float32, device=agent.device).unsqueeze(0)
    else:
        state = state.to(agent.device)
    assert state.shape == (args.num_envs, env.state_dim)
    assert isinstance(state, th.Tensor)
    agent.last_state = state.detach()

    '''init buffer'''
    if args.if_off_policy:
        buffer = ReplayBuffer(
            gpu_id=args.gpu_id,
            num_seqs=args.num_envs,
            max_size=args.buffer_size,
            state_dim=args.state_dim, # Use corrected args.state_dim
            action_dim=1 if args.if_discrete else args.action_dim,
            if_use_per=args.if_use_per,
            if_discrete=args.if_discrete,
            args=args,
        )
    else:
        buffer = []

    '''init evaluator'''
    eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
    eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
    eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
    evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args, if_tensorboard=False)

    '''train loop'''
    cwd = args.cwd
    break_step = args.break_step
    horizon_len = args.horizon_len
    if_off_policy = args.if_off_policy
    if_save_buffer = args.if_save_buffer

    if_discrete = env.if_discrete
    # Correct calculation if num_workers doesn't exist in single process args
    num_workers_or_1 = getattr(args, 'num_workers', 1)
    show_weight = 1000 / horizon_len / args.num_envs / num_workers_or_1

    def action_to_str(_action_ary):
        _show_dict = dict(zip(*np.unique(_action_ary, return_counts=True)))
        _show_str = np.array([int(_show_dict.get(action_key, 0) * show_weight)
                              for action_key in range(env.action_dim)])
        return _show_str

    # Keep args for Evaluator, env closing etc.
    # del args # Keep args

    if_train = True
    while if_train:
        buffer_items = agent.explore_env(env, horizon_len)
        if if_off_policy:
            buffer.update(buffer_items)
        else:
            buffer[:] = buffer_items

        if if_discrete:
            action_data = buffer_items[1] if isinstance(buffer_items, tuple) and len(buffer_items) > 1 else None
            if action_data is not None:
                 show_str = action_to_str(_action_ary=action_data.data.cpu())
            else:
                 show_str = '' # Handle case where action data might be missing
        else:
            show_str = ''
        exp_r_index = 3 if not if_off_policy else 2 # Check reward index
        exp_r = buffer_items[exp_r_index].mean().item() if isinstance(buffer_items, tuple) and len(buffer_items) > exp_r_index else 0.0

        th.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        # Ensure explore_rate exists, provide default if not
        explore_rate_val = getattr(agent, 'explore_rate', 0.0)
        logging_tuple = (*logging_tuple, explore_rate_val, show_str)
        th.set_grad_enabled(False)

        evaluator.evaluate_and_save(actor=agent.act, steps=horizon_len, exp_r=exp_r, logging_tuple=logging_tuple)
        if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))

    print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}', flush=True)

    env.close() if hasattr(env, 'close') else None
    evaluator.save_training_curve_jpg()
    agent.save_or_load_agent(cwd, if_save=True)
    if if_save_buffer and hasattr(buffer, 'save_or_load_history'):
        buffer.save_or_load_history(cwd, if_save=True)


def train_agent_multiprocessing(args: Config):
    args.init_before_training()

    """Don't set method='fork' when send tensor in GPU"""
    method = 'spawn' if os.name == 'nt' else 'forkserver'  # os.name == 'nt' means Windows NT operating system (WinOS)
    mp.set_start_method(method=method, force=True)

    '''build the Pipe'''
    worker_pipes = [Pipe(duplex=False) for _ in range(args.num_workers)]  # receive, send
    learner_pipe = Pipe(duplex=False)
    evaluator_pipe = Pipe(duplex=True)

    '''build Process'''
    learner = Learner(learner_pipe=learner_pipe, worker_pipes=worker_pipes, evaluator_pipe=evaluator_pipe, args=args)
    workers = [Worker(worker_pipe=worker_pipe, learner_pipe=learner_pipe, worker_id=worker_id, args=args)
               for worker_id, worker_pipe in enumerate(worker_pipes)]
    evaluator = EvaluatorProc(evaluator_pipe=evaluator_pipe, args=args)

    '''start Process with single GPU'''
    process_list = [learner, *workers, evaluator]
    [process.start() for process in process_list]
    [process.join() for process in process_list]


def train_agent_multiprocessing_multi_gpu(args: Config):
    args.init_before_training()

    """Don't set method='fork' when send tensor in GPU"""
    method = 'spawn' if os.name == 'nt' else 'forkserver'  # os.name == 'nt' means Windows NT operating system (WinOS)
    mp.set_start_method(method=method, force=True)

    learners_pipe = [Pipe(duplex=True) for _ in args.learner_gpu_ids]
    process_list_list = []
    for gpu_id in args.learner_gpu_ids:
        args = deepcopy(args)
        args.gpu_id = gpu_id

        '''Pipe build'''
        worker_pipes = [Pipe(duplex=False) for _ in range(args.num_workers)]  # receive, send
        learner_pipe = Pipe(duplex=False)
        evaluator_pipe = Pipe(duplex=True)

        '''Process build'''
        learner = Learner(learner_pipe=learner_pipe,
                          worker_pipes=worker_pipes,
                          evaluator_pipe=evaluator_pipe,
                          learners_pipe=learners_pipe,
                          args=args)
        workers = [Worker(worker_pipe=worker_pipe, learner_pipe=learner_pipe, worker_id=worker_id, args=args)
                   for worker_id, worker_pipe in enumerate(worker_pipes)]
        evaluator = EvaluatorProc(evaluator_pipe=evaluator_pipe, args=args)

        '''Process append'''
        process_list = [learner, *workers, evaluator]
        process_list_list.append(process_list)

    '''Process start'''
    for process_list in process_list_list:
        [process.start() for process in process_list]
    '''Process join'''
    for process_list in process_list_list:
        [process.join() for process in process_list]


class Learner(Process):
    def __init__(
            self,
            learner_pipe: Pipe,
            worker_pipes: List[Pipe],
            evaluator_pipe: Pipe,
            learners_pipe: Optional[List[Pipe]] = None,
            args: Config = Config(),
    ):
        super().__init__()
        self.recv_pipe = learner_pipe[0]
        self.send_pipes = [worker_pipe[1] for worker_pipe in worker_pipes]
        self.eval_pipe = evaluator_pipe[1]
        self.learners_pipe = learners_pipe
        self.args = args

    def run(self):
        args = self.args
        th.set_grad_enabled(False)

        '''COMMUNICATE between Learners: init'''
        learner_id = args.learner_gpu_ids.index(args.gpu_id) if len(args.learner_gpu_ids) > 0 else 0
        num_learners = max(1, len(args.learner_gpu_ids))
        num_communications = num_learners - 1
        if len(args.learner_gpu_ids) >= 2:
            assert isinstance(self.learners_pipe, list)
        elif len(args.learner_gpu_ids) == 0:
            assert self.learners_pipe is None
        elif len(args.learner_gpu_ids) == 1:
            ValueError("| Learner: suggest to set `args.learner_gpu_ids=()` in default")

        # <<< START MODIFICATION FOR LEARNER >>>
        # Correct args.state_dim using a temp env instance BEFORE agent init
        try:
            print(f"[Learner] Determining correct state_dim. Initial args.state_dim: {args.state_dim}", flush=True)
            temp_env_args = deepcopy(args.env_args)
            temp_env_class_name = getattr(args, 'env_class_name', None)
            # Build env on the learner's designated GPU or CPU
            temp_env = build_env(args.env_class, temp_env_args, args.gpu_id, temp_env_class_name)
            actual_state_dim = getattr(temp_env, 'state_dim', getattr(temp_env, 'state_space', None))
            if actual_state_dim is not None and args.state_dim != actual_state_dim:
                print(f"[Learner] Correcting args.state_dim from {args.state_dim} to {actual_state_dim}", flush=True)
                args.state_dim = actual_state_dim
            elif actual_state_dim is None:
                print(f"[Learner] Warning: Could not determine actual state dim from temp env. Using {args.state_dim}", flush=True)
            del temp_env
        except Exception as e:
            print(f"[Learner] Error during temp env creation for state_dim check: {e}. Using initial args.state_dim {args.state_dim}", flush=True)
        # <<< END MODIFICATION FOR LEARNER >>>


        '''Learner init agent'''
        # Agent initialized with potentially corrected args.state_dim
        agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
        if args.continue_train:
            # Pass if_save=False to load
            agent.save_or_load_agent(cwd=args.cwd, if_save=False)


        '''Learner init buffer'''
        if args.if_off_policy:
            buffer = ReplayBuffer(
                gpu_id=args.gpu_id,
                num_seqs=args.num_envs * args.num_workers * num_learners,
                max_size=args.buffer_size,
                state_dim=args.state_dim, # Use potentially corrected state_dim
                action_dim=1 if args.if_discrete else args.action_dim,
                if_use_per=args.if_use_per,
                args=args,
            )
        else:
            buffer = []

        '''loop'''
        if_off_policy = args.if_off_policy
        if_discrete = args.if_discrete
        if_save_buffer = args.if_save_buffer

        num_workers = args.num_workers
        num_envs = args.num_envs
        num_steps = args.horizon_len * args.num_workers
        num_seqs = args.num_envs * args.num_workers * num_learners

        state_dim = args.state_dim # Use potentially corrected state_dim
        action_dim = args.action_dim
        horizon_len = args.horizon_len
        cwd = args.cwd
        # Keep args for learner communication section

        agent.last_state = th.empty((num_seqs, state_dim), dtype=th.float32, device=agent.device)

        # Initialize buffer tensors with the corrected state_dim
        states = th.zeros((horizon_len, num_seqs, state_dim), dtype=th.float32, device=agent.device)
        actions = th.zeros((horizon_len, num_seqs, action_dim), dtype=th.float32, device=agent.device) \
            if not if_discrete else th.zeros((horizon_len, num_seqs), dtype=th.int32).to(agent.device) # Correct discrete shape
        rewards = th.zeros((horizon_len, num_seqs), dtype=th.float32, device=agent.device)
        undones = th.zeros((horizon_len, num_seqs), dtype=th.bool, device=agent.device)
        unmasks = th.zeros((horizon_len, num_seqs), dtype=th.bool, device=agent.device)

        if if_off_policy:
            buffer_items_tensor = (states, actions, rewards, undones, unmasks)
        else:
            logprobs = th.zeros((horizon_len, num_seqs), dtype=th.float32, device=agent.device) # Correct logprobs shape
            buffer_items_tensor = (states, actions, logprobs, rewards, undones, unmasks)

        if_train = True
        while if_train:
            actor = agent.act
            actor = deepcopy(actor).cpu() if os.name == 'nt' else actor  # WindowsNT_OS can only send cpu_tensor

            '''Learner send actor to Workers'''
            for send_pipe in self.send_pipes:
                send_pipe.send(actor)
            '''Learner receive (buffer_items, last_state) from Workers'''
            for _ in range(num_workers):
                worker_id, buffer_items, last_state = self.recv_pipe.recv()

                buf_i = num_envs * worker_id
                buf_j = num_envs * (worker_id + 1)
                for i, (buffer_item, buffer_tensor) in enumerate(zip(buffer_items, buffer_items_tensor)):
                    # Check shapes carefully before assignment
                    target_shape = buffer_tensor[:, buf_i:buf_j].shape
                    source_shape = buffer_item.shape
                    if source_shape == target_shape:
                         buffer_tensor[:, buf_i:buf_j] = buffer_item.to(agent.device)
                    # Handle potential trailing dimension mismatch (e.g., state dim)
                    elif len(source_shape) == len(target_shape) and source_shape[:-1] == target_shape[:-1]:
                         print(f"Learner WARNING (Worker Recv): Mismatched last dim at index {i}! Source {source_shape}, Target {target_shape}. Truncating source.")
                         min_dim = min(source_shape[-1], target_shape[-1])
                         buffer_tensor[:, buf_i:buf_j, ..., :min_dim] = buffer_item[..., :min_dim].to(agent.device)
                    else:
                         print(f"Learner ERROR (Worker Recv): Incompatible shapes at index {i}! Source {source_shape}, Target {target_shape}. Skipping update.")

                # Check last_state shape
                target_last_state_shape = agent.last_state[buf_i:buf_j].shape
                source_last_state_shape = last_state.shape
                if source_last_state_shape == target_last_state_shape:
                     agent.last_state[buf_i:buf_j] = last_state.to(agent.device)
                elif len(source_last_state_shape) == len(target_last_state_shape) and source_last_state_shape[:-1] == target_last_state_shape[:-1]:
                     print(f"Learner WARNING (Worker Recv): Mismatched last_state last dim! Source {source_last_state_shape}, Target {target_last_state_shape}. Truncating source.")
                     min_dim = min(source_last_state_shape[-1], target_last_state_shape[-1])
                     agent.last_state[buf_i:buf_j, ..., :min_dim] = last_state[..., :min_dim].to(agent.device)
                else:
                     print(f"Learner ERROR (Worker Recv): Incompatible last_state shapes! Source {source_last_state_shape}, Target {target_last_state_shape}. Skipping update.")


            del buffer_items, last_state

            '''COMMUNICATE between Learners: Learner send actor to other Learners'''
            _buffer_len = num_envs * num_workers
            # Ensure tensors being sent match the expected buffer dimensions
            _buffer_items_tensor_send = [t[:, :_buffer_len].cpu().detach() for t in buffer_items_tensor]
            for shift_id in range(num_communications):
                _learner_pipe = self.learners_pipe[learner_id][0]
                _learner_pipe.send(_buffer_items_tensor_send)
            '''COMMUNICATE between Learners: Learner receive (buffer_items, last_state) from other Learners'''
            for shift_id in range(num_communications):
                _learner_id = (learner_id + shift_id + 1) % num_learners  # other_learner_id
                _learner_pipe = self.learners_pipe[_learner_id][1]
                _buffer_items_recv = _learner_pipe.recv()

                _buf_i = num_envs * num_workers * (shift_id + 1)
                _buf_j = num_envs * num_workers * (shift_id + 2)
                for i, (buffer_item, buffer_tensor) in enumerate(zip(_buffer_items_recv, buffer_items_tensor)):
                    # Check shapes carefully before assignment
                    target_shape = buffer_tensor[:, _buf_i:_buf_j].shape
                    source_shape = buffer_item.shape
                    if source_shape == target_shape:
                         buffer_tensor[:, _buf_i:_buf_j] = buffer_item.to(agent.device)
                    elif len(source_shape) == len(target_shape) and source_shape[:-1] == target_shape[:-1]:
                         print(f"Learner WARNING (Comm Recv): Mismatched last dim at index {i}! Source {source_shape}, Target {target_shape}. Truncating source.")
                         min_dim = min(source_shape[-1], target_shape[-1])
                         buffer_tensor[:, _buf_i:_buf_j, ..., :min_dim] = buffer_item[..., :min_dim].to(agent.device)
                    else:
                         print(f"Learner ERROR (Comm Recv): Incompatible shapes at index {i}! Source {source_shape}, Target {target_shape}. Skipping update.")

            '''Learner update training data to (buffer, agent)'''
            if if_off_policy:
                buffer.update(buffer_items_tensor)
            else:
                buffer[:] = buffer_items_tensor

            '''Learner update network using training data'''
            th.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            th.set_grad_enabled(False)

            '''Learner receive training signal from Evaluator'''
            if self.eval_pipe.poll():  # whether there is any data available to be read of this pipe0
                if_train = self.eval_pipe.recv()  # True means evaluator in idle moments.
            else:
                actor = None # Set actor to None if no signal received

            '''Learner send actor and training log to Evaluator'''
            if if_train:
                if actor is None: # Re-fetch actor if it was None
                     actor = agent.act
                     actor = deepcopy(actor).cpu() if os.name == 'nt' else actor
                # Determine reward index based on policy type
                reward_index = 3 if not agent.if_off_policy else 2
                # Safely access rewards tensor
                if isinstance(buffer_items_tensor, tuple) and len(buffer_items_tensor) > reward_index:
                    exp_r = buffer_items_tensor[reward_index].mean().item()
                else:
                    exp_r = 0.0 # Default if buffer structure is unexpected
                    print("Learner Warning: Could not determine exp_r from buffer_items_tensor.")

                self.eval_pipe.send((actor, num_steps, exp_r, logging_tuple))
            # Keep args for potential future communication needs
            # del args

        '''Learner send the terminal signal to workers after break the loop'''
        print("| Learner Close Worker", flush=True)
        for send_pipe in self.send_pipes:
            send_pipe.send(None)
            time.sleep(0.1)

        '''save'''
        agent.save_or_load_agent(cwd=cwd, if_save=True)
        if if_save_buffer and hasattr(buffer, 'save_or_load_history'):
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}", flush=True)
            buffer.save_or_load_history(cwd, if_save=True)
            print(f"| LearnerPipe.run: ReplayBuffer saved  in {cwd}", flush=True)
        print("| Learner Closed", flush=True)


class Worker(Process):
    def __init__(self, worker_pipe: Pipe, learner_pipe: Pipe, worker_id: int, args: Config):
        super().__init__()
        self.recv_pipe = worker_pipe[0]
        self.send_pipe = learner_pipe[1]
        self.worker_id = worker_id
        self.args = args # Store the config object

    def run(self):
        args = self.args # Use the stored config object
        worker_id = self.worker_id
        th.set_grad_enabled(False)

        '''init environment'''
        env_class_name_str = getattr(args, 'env_class_name', None)
        env = build_env(
            env_class=args.env_class,
            env_args=args.env_args,
            gpu_id=args.gpu_id,
            env_class_name=env_class_name_str
        )

        '''Determine Correct Dimension from Env'''
        actual_env_state_dim = None
        if hasattr(env, 'state_dim'):
            actual_env_state_dim = env.state_dim
        elif hasattr(env, 'state_space'): # Fallback
             actual_env_state_dim = env.state_space
        else:
             print(f"[Worker {worker_id}] FATAL ERROR: env object missing 'state_dim' or 'state_space'. Cannot determine dimension.", flush=True)
             actual_env_state_dim = args.state_dim # Last resort

        if args.state_dim != actual_env_state_dim:
             print(f"[Worker {worker_id}] INFO: Mismatch detected. args.state_dim={args.state_dim}, actual_env_state_dim={actual_env_state_dim}. Using actual.", flush=True)
             # It's safer to use the actual dim for agent init below, rather than modifying args here

        '''init agent'''
        # <<< MODIFIED: Use actual_env_state_dim determined above for agent init >>>
        agent = args.agent_class(args.net_dims, actual_env_state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
        if args.continue_train:
             # Pass if_save=False to load
             agent.save_or_load_agent(cwd=args.cwd, if_save=False)


        '''init agent.last_state'''
        state, info_dict = env.reset()
        if args.num_envs == 1:
            # Use actual_env_state_dim for assertion
            assert state.shape == (actual_env_state_dim,)
            assert isinstance(state, np.ndarray)
            state = th.tensor(state, dtype=th.float32, device=agent.device).unsqueeze(0)
        else:
            # Use actual_env_state_dim for assertion
            assert state.shape == (args.num_envs, actual_env_state_dim)
            assert isinstance(state, th.Tensor)
            state = state.to(agent.device)
        agent.last_state = state.detach()

        '''init buffer'''
        horizon_len = args.horizon_len

        '''loop'''
        # Keep args for the loop

        while True:
            '''Worker receive actor from Learner'''
            actor = self.recv_pipe.recv()
            if actor is None:
                break
            agent.act = actor.to(agent.device) if os.name == 'nt' else actor

            '''Worker send the training data to Learner'''
            buffer_items = agent.explore_env(env, horizon_len)
            last_state = agent.last_state
            if os.name == 'nt':
                buffer_items = [t.cpu() for t in buffer_items]
                last_state = deepcopy(last_state).cpu()
            self.send_pipe.send((worker_id, buffer_items, last_state))

        env.close() if hasattr(env, 'close') else None
        print(f"| Worker-{self.worker_id} Closed", flush=True)


class EvaluatorProc(Process):
    def __init__(self, evaluator_pipe: Pipe, args: Config):
        super().__init__()
        self.pipe0 = evaluator_pipe[0]
        self.pipe1 = evaluator_pipe[1]
        self.args = args

    def run(self):
        args = self.args
        th.set_grad_enabled(False)

        '''init evaluator'''
        eval_env_class_obj = args.eval_env_class
        eval_env_args = args.eval_env_args
        eval_env_class_name_str = getattr(args, 'eval_env_class_name', None)

        if eval_env_class_obj is None and eval_env_class_name_str is None:
             print("[EvaluatorProc] Warning: Both eval_env_class and eval_env_class_name are None. Falling back to main env config.")
             eval_env_class_obj = args.env_class
             eval_env_class_name_str = getattr(args, 'env_class_name', None)

        if eval_env_args is None:
             print("[EvaluatorProc] Warning: eval_env_args is None. Falling back to main env_args.")
             eval_env_args = args.env_args

        eval_env = build_env(
            env_class=eval_env_class_obj,
            env_args=eval_env_args,
            gpu_id=args.gpu_id,
            env_class_name=eval_env_class_name_str
        )
        evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args, if_tensorboard=False)

        '''loop'''
        cwd = args.cwd
        break_step = args.break_step
        device = th.device(f"cuda:{args.gpu_id}" if (th.cuda.is_available() and (args.gpu_id >= 0)) else "cpu")
        del args # Keep evaluator

        if_train = True
        while if_train:
            '''Evaluator receive training log from Learner'''
            actor, steps, exp_r, logging_tuple = self.pipe0.recv()

            '''Evaluator evaluate the actor and save the training log'''
            if actor is None:
                evaluator.total_step += steps
            else:
                actor = actor.to(device) if os.name == 'nt' else actor
                evaluator.evaluate_and_save(actor, steps, exp_r, logging_tuple)

            '''Evaluator send the training signal to Learner'''
            if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))
            self.pipe0.send(if_train)

        '''Evaluator save the training log and draw the learning curve'''
        evaluator.save_training_curve_jpg()
        print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}', flush=True)

        print("| Evaluator Closing", flush=True)
        while self.pipe1.poll():
            while self.pipe0.poll():
                try:
                    self.pipe0.recv()
                except RuntimeError:
                    print("| Evaluator Ignore RuntimeError in self.pipe0.recv()", flush=True)
                time.sleep(1)
            time.sleep(1)

        eval_env.close() if hasattr(eval_env, 'close') else None
        print("| Evaluator Closed", flush=True)


'''render'''


def valid_agent(env_class, env_args: dict, net_dims: List[int], agent_class, actor_path: str, render_times: int = 8):
    env = build_env(env_class, env_args)

    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    agent = agent_class(net_dims, state_dim, action_dim, gpu_id=-1)
    actor = agent.act

    print(f"| render and load actor from: {actor_path}", flush=True)
    actor.load_state_dict(th.load(actor_path, map_location=lambda storage, loc: storage))
    for i in range(render_times):
        cumulative_reward, episode_step = get_rewards_and_steps(env, actor, if_render=True)
        print(f"|{i:4}  cumulative_reward {cumulative_reward:9.3f}  episode_step {episode_step:5.0f}", flush=True)


def render_agent(env_class, env_args: dict, net_dims: [int], agent_class, actor_path: str, render_times: int = 8):
    env = build_env(env_class, env_args)

    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    agent = agent_class(net_dims, state_dim, action_dim, gpu_id=-1)
    actor = agent.act
    del agent

    print(f"| render and load actor from: {actor_path}", flush=True)
    actor.load_state_dict(th.load(actor_path, map_location=lambda storage, loc: storage))
    for i in range(render_times):
        cumulative_reward, episode_step = get_rewards_and_steps(env, actor, if_render=True)
        print(f"|{i:4}  cumulative_reward {cumulative_reward:9.3f}  episode_step {episode_step:5.0f}", flush=True)
