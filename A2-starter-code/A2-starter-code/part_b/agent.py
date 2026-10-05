import random
import time

from env import HighwayEnv, ACTION_DECREASE_LANE, ACTION_DECREASE_SPEED, ACTION_INCREASE_LANE


class Agent:
    """
    Tabular Q-learning agent on a compact, ego-centric state abstraction.

    State: (speed, own-lane distance, previous own-lane distance, left distance,
    right distance), where a side lane outside the road is encoded as a wall.
    The previous own-lane bin tells the agent how far into the coarse distance
    bin the car ahead is. The highway is left/right symmetric, so states are
    canonicalised with left <= right and the lane-change actions are swapped
    when a state is mirrored.

    Learning: Q-learning with per-(state, action) learning rates, a decaying
    epsilon-greedy policy, uniform experience replay after every real step and
    a backward replay of each finished episode. Collisions are penalised more
    heavily during learning than the raw -5 so the policy trades a little
    speed for a much lower crash rate.
    """

    # Exploration: epsilon decays linearly from EPS_START to EPS_END over the
    # first EPS_DECAY_FRAC of the training budget.
    EPS_START = 0.5
    EPS_END = 0.02
    EPS_DECAY_FRAC = 0.8

    # Learning rate per (s, a): max(ALPHA_MIN, n(s, a) ** -ALPHA_POW).
    ALPHA_MIN = 0.01
    ALPHA_POW = 0.6

    # Replayed updates per environment step, and replay buffer capacity.
    REPLAY_K = 8
    REPLAY_CAPACITY = 200000

    # Multiplier applied to the collision reward for learning only.
    COLLISION_PENALTY_MULT = 20.0

    def __init__(self, env: HighwayEnv, discount_factor=0.99):
        self.gamma = float(discount_factor)
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"Invalid discount factor: {self.gamma}")

        self.num_lanes = int(env.num_lanes)
        self.num_actions = int(env.num_actions)
        self.num_dist = int(env.num_dist_states)
        self.wall = self.num_dist
        self.side_states = self.num_dist + 1

        num_states = (int(env.num_speed_states) * self.num_dist * self.num_dist
                      * self.side_states * self.side_states)
        self.q = [[0.0] * self.num_actions for _ in range(num_states)]
        self.visits = [[0] * self.num_actions for _ in range(num_states)]

        # Action remap for mirrored states: swap the two lane changes
        mirror = list(range(self.num_actions))
        mirror[ACTION_INCREASE_LANE] = ACTION_DECREASE_LANE
        mirror[ACTION_DECREASE_LANE] = ACTION_INCREASE_LANE
        self.mirror_action = mirror

        # Previous observation's distances, used for the previous own-lane bin
        self.prev_dist = None

    def _encode(self, speed, lane, min_dist):
        """
        Maps an observation to (state index, mirrored). Lanes are 0-indexed,
        matching env.get_obs(). Updates the stored previous observation.
        """
        prev = self.prev_dist
        self.prev_dist = min_dist
        own = min_dist[lane]
        prev_own = prev[lane] if prev is not None else own

        left = min_dist[lane - 1] if lane > 0 else self.wall
        right = min_dist[lane + 1] if lane < self.num_lanes - 1 else self.wall
        mirrored = left > right
        if mirrored:
            left, right = right, left

        nd, ns = self.num_dist, self.side_states
        index = (((speed * nd + own) * nd + prev_own) * ns + left) * ns + right
        return index, mirrored

    def learn_policy(self, time_limit):
        clock = time.monotonic
        start_time = clock()
        time_limit = max(0.0, float(time_limit))
        safety_margin = min(0.5, max(0.05, 0.02 * time_limit))
        budget = time_limit - safety_margin
        if budget <= 0.0:
            return
        deadline = start_time + budget

        # Private RNG: the environment reseeds NumPy's global RNG on every reset
        rand = random.Random().random

        # Local bindings for the hot loop
        q, visits, gamma = self.q, self.visits, self.gamma
        encode, mirror_action = self._encode, self.mirror_action
        num_actions = self.num_actions
        penalty_mult = self.COLLISION_PENALTY_MULT
        alpha_min = self.ALPHA_MIN
        alpha_cap = 100000
        alpha_table = [1.0] + [max(alpha_min, k ** -self.ALPHA_POW) for k in range(1, alpha_cap + 1)]
        replay_k = self.REPLAY_K
        capacity = self.REPLAY_CAPACITY
        buf_s = [0] * capacity
        buf_a = [0] * capacity
        buf_r = [0.0] * capacity
        buf_s2 = [0] * capacity
        buf_term = [False] * capacity
        buf_size = buf_pos = 0

        while clock() < deadline:
            progress = (clock() - start_time) / budget
            epsilon = max(self.EPS_END, self.EPS_START * (1.0 - progress / self.EPS_DECAY_FRAC))

            # Fresh environment each episode: reset() on a used env spawns
            # traffic relative to the previous episode's car position
            env = HighwayEnv()
            env_step = env.step
            self.prev_dist = None
            state, mirrored = encode(*env.get_state())
            episode = []
            done = False
            steps = 0

            while not done:
                state_q = q[state]
                if rand() < epsilon:
                    action = int(rand() * num_actions)
                else:
                    action = state_q.index(max(state_q))

                obs, reward, done = env_step(mirror_action[action] if mirrored else action)
                steps += 1
                next_state, next_mirrored = encode(*obs)

                # Collisions are the only negative reward and the only true
                # terminal; the 1000-step time limit bootstraps instead
                terminal = reward < 0
                if terminal:
                    reward *= penalty_mult
                    target = reward
                else:
                    target = reward + gamma * max(q[next_state])

                state_visits = visits[state]
                state_visits[action] += 1
                n = state_visits[action]
                state_q[action] += (alpha_table[n] if n <= alpha_cap else alpha_min) * (target - state_q[action])

                episode.append((state, action, reward, next_state, terminal))
                buf_s[buf_pos] = state
                buf_a[buf_pos] = action
                buf_r[buf_pos] = reward
                buf_s2[buf_pos] = next_state
                buf_term[buf_pos] = terminal
                buf_pos += 1
                if buf_pos == capacity:
                    buf_pos = 0
                if buf_size < capacity:
                    buf_size += 1

                # Experience replay: reuse stored transitions, since an
                # environment step costs several times more than an update
                for _ in range(replay_k):
                    j = int(rand() * buf_size)
                    rs, ra = buf_s[j], buf_a[j]
                    rq = q[rs]
                    rt = buf_r[j] if buf_term[j] else buf_r[j] + gamma * max(q[buf_s2[j]])
                    n = visits[rs][ra]
                    rq[ra] += (alpha_table[n] if n <= alpha_cap else alpha_min) * (rt - rq[ra])

                state, mirrored = next_state, next_mirrored
                if (steps & 127) == 0 and clock() >= deadline:
                    break

            # Backward replay propagates the episode's outcome along its path
            for es, ea, er, es2, et in reversed(episode):
                eq = q[es]
                et_target = er if et else er + gamma * max(q[es2])
                n = visits[es][ea]
                eq[ea] += (alpha_table[n] if n <= alpha_cap else alpha_min) * (et_target - eq[ea])

        self.prev_dist = None

    def get_action(self, speed: int, lane: int, min_dist: list) -> int:
        state, mirrored = self._encode(speed, lane, min_dist)
        state_visits = self.visits[state]
        if not any(state_visits):
            # Never seen this state: slowing down is the safest default
            return ACTION_DECREASE_SPEED
        state_q = self.q[state]
        best = max(state_q[a] for a in range(self.num_actions) if state_visits[a])
        action = next(a for a in range(self.num_actions) if state_visits[a] and state_q[a] == best)
        return self.mirror_action[action] if mirrored else action
