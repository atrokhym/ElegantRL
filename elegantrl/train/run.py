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
    evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args, if_tensorboard=True) # MODIFIED

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
        evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args, if_tensorboard=True) # MODIFIED

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
