import random
import time

from env import TreasureHunt


class Agent:

    def __init__(self, layout_file, prob_file):
        self.layout_file = layout_file
        self.prob_file = prob_file

        self.env = TreasureHunt(layout_file, prob_file)

        self.gamma = self.env.df
        self.rewards = self.env.rewards
        self.ship_prob = self.env.ship_prob
        self.pirate_prob = self.env.pirate_prob

        self.n = self.env.N
        self.actions = range(4)
        self.action_delta = (
            (1, 0),
            (-1, 0),
            (0, -1),
            (0, 1)
        )

        self.treasure_locations = tuple(
            self.env.locations["treasure"]
        )

        self.pirate_areas = [
            tuple(self.env.pirate_areas[0]),
            tuple(self.env.pirate_areas[1])
        ]

        self.pirate_area_sets = [
            set(self.pirate_areas[0]),
            set(self.pirate_areas[1])
        ]

        self.fort_locations = set(self.env.locations["fort"])
        self.land_locations = set(self.env.locations["land"])

        self.ship_locations = []

        for row in range(self.n):
            for col in range(self.n):
                location = (row, col)

                if location not in self.land_locations:
                    self.ship_locations.append(location)

        self.states = []
        self.state_to_index = {}

        self._build_states()

        self.num_states = len(self.states)
        self.num_actions = 4

        self.transitions = [None] * self.num_states
        self.predecessors = [set() for _ in range(self.num_states)]

        self.ship_distributions = {}
        self.pirate_distributions = [{}, {}]

        self._build_transition_model()

        self.values = [0.0] * self.num_states

        # Start with a random policy.
        self.policy = [
            random.randrange(self.num_actions)
            for _ in range(self.num_states)
        ]

        self.policy_ready = False

    def _build_states(self):
        """
        Each state is represented as:

            (ship_location, pirate_1_location,
             pirate_2_location, treasure_mask)

        A treasure mask indicates which treasures are still present.
        """

        for ship_location in self.ship_locations:
            for pirate_1_location in self.pirate_areas[0]:
                for pirate_2_location in self.pirate_areas[1]:
                    for treasure_mask in range(
                        1 << len(self.treasure_locations)
                    ):
                        state = (
                            ship_location,
                            pirate_1_location,
                            pirate_2_location,
                            treasure_mask
                        )

                        state_index = len(self.states)

                        self.states.append(state)
                        self.state_to_index[state] = state_index

    def _is_terminal(self, ship_location, pirate_1_location,
                     pirate_2_location):
        if ship_location in self.fort_locations:
            return True

        if ship_location == pirate_1_location:
            return True

        if ship_location == pirate_2_location:
            return True

        return False

    def _move(self, location, action):
        """
        This follows the same coordinate system as env.py.

        Actions:
            0 -> UP
            1 -> DOWN
            2 -> LEFT
            3 -> RIGHT
        """

        row_delta, col_delta = self.action_delta[action]

        return (
            location[0] + row_delta,
            location[1] + col_delta
        )

    def _valid_ship_location(self, location):
        row, col = location

        if row < 0 or row >= self.n:
            return False

        if col < 0 or col >= self.n:
            return False

        if location in self.land_locations:
            return False

        return True

    def _pirate_distribution(self, location, pirate_number):
        """
        Return a list containing:

            (new_location, probability)

        Invalid pirate moves result in the pirate staying still.
        """

        cached = self.pirate_distributions[pirate_number].get(location)
        if cached is not None:
            return cached

        probabilities = self.pirate_prob[pirate_number]

        if pirate_number == 0:
            valid_area = self.pirate_area_sets[0]
        else:
            valid_area = self.pirate_area_sets[1]

        distribution = {}

        for action in self.actions:
            probability = probabilities[action]

            candidate = self._move(location, action)

            if candidate not in valid_area:
                candidate = location

            distribution[candidate] = (
                distribution.get(candidate, 0.0) + probability
            )

        result = tuple(distribution.items())
        self.pirate_distributions[pirate_number][location] = result
        return result

    def _ship_distribution(self, location, action):
        """
        The ship follows the selected action with probability
        ship_prob[0].

        With probability ship_prob[1], it chooses uniformly from
        the other three actions.
        """

        cached = self.ship_distributions.get(location)
        if cached is None:
            cached = []
            self.ship_distributions[location] = cached
        else:
            result = cached[action]
            if result is not None:
                return result

        if not cached:
            cached.extend([None] * self.num_actions)

        distribution = {}

        intended_probability = self.ship_prob[0]
        random_probability = self.ship_prob[1] / 3.0

        for possible_action in self.actions:
            if possible_action == action:
                probability = intended_probability
            else:
                probability = random_probability

            candidate = self._move(location, possible_action)

            if not self._valid_ship_location(candidate):
                candidate = location

            distribution[candidate] = (
                distribution.get(candidate, 0.0) + probability
            )

        result = tuple(distribution.items())
        cached[action] = result
        return result

    def _next_treasure_mask(self, ship_location, treasure_mask):
        """
        Remove a treasure when the ship reaches it.

        The environment checks pirate/fort collisions before treasure
        collection. This method is called only for non-terminal states.
        """

        new_mask = treasure_mask

        for treasure_index, treasure_location in enumerate(
            self.treasure_locations
        ):
            treasure_is_present = (
                (treasure_mask & (1 << treasure_index)) != 0
            )

            if (
                treasure_is_present
                and ship_location == treasure_location
            ):
                new_mask = new_mask & ~(1 << treasure_index)

        return new_mask

    def _transition_outcomes(self, state, action):
        """
        Construct all possible outcomes of taking action in state.

        Each outcome is:

            (next_state_index, probability, reward, terminal)
        """

        (
            ship_location,
            pirate_1_location,
            pirate_2_location,
            treasure_mask
        ) = state

        if self._is_terminal(
            ship_location,
            pirate_1_location,
            pirate_2_location
        ):
            return ()

        outcomes = {}

        pirate_1_distribution = self._pirate_distribution(
            pirate_1_location,
            0
        )

        pirate_2_distribution = self._pirate_distribution(
            pirate_2_location,
            1
        )

        for next_pirate_1, pirate_1_probability in pirate_1_distribution:
            for next_pirate_2, pirate_2_probability in pirate_2_distribution:
                pirate_probability = (
                    pirate_1_probability * pirate_2_probability
                )

                ship_distribution = self._ship_distribution(
                    ship_location,
                    action
                )

                for next_ship, ship_probability in ship_distribution:
                    probability = (
                        pirate_probability * ship_probability
                    )

                    reward = self.rewards["step"]

                    is_pirate_collision = (
                        next_ship == next_pirate_1
                        or next_ship == next_pirate_2
                    )

                    is_fort_collision = (
                        next_ship in self.fort_locations
                    )

                    terminal = (
                        is_pirate_collision
                        or is_fort_collision
                    )

                    next_treasure_mask = treasure_mask

                    if is_pirate_collision:
                        reward += self.rewards["pirate"]

                    elif is_fort_collision:
                        reward += self.rewards["fort"]

                    else:
                        next_treasure_mask = self._next_treasure_mask(
                            next_ship,
                            treasure_mask
                        )

                        if next_treasure_mask != treasure_mask:
                            reward += self.rewards["treasure"]

                    next_state = (
                        next_ship,
                        next_pirate_1,
                        next_pirate_2,
                        next_treasure_mask
                    )

                    next_state_index = self.state_to_index[next_state]

                    outcome_key = (
                        next_state_index,
                        reward,
                        terminal
                    )

                    outcomes[outcome_key] = (
                        outcomes.get(outcome_key, 0.0)
                        + probability
                    )

        return [
            (
                next_state_index,
                probability,
                reward,
                terminal
            )
            for (
                next_state_index,
                reward,
                terminal
            ), probability in outcomes.items()
        ]

    def _build_transition_model(self):
        """
        Build transitions and predecessor lists for every state/action.
        """

        for state_index, state in enumerate(self.states):
            self.transitions[state_index] = []

            for action in self.actions:
                outcomes = self._transition_outcomes(state, action)

                self.transitions[state_index].append(outcomes)

                for (
                    next_state_index,
                    probability,
                    reward,
                    terminal
                ) in outcomes:
                    if probability > 0 and not terminal:
                        self.predecessors[next_state_index].add(
                            state_index
                        )

    def _action_value(self, state_index, action):
        """
        Calculate Q(s, a) using the current value function.
        """

        value = 0.0

        for (
            next_state_index,
            probability,
            reward,
            terminal
        ) in self.transitions[state_index][action]:

            if terminal:
                future_value = 0.0
            else:
                future_value = self.values[next_state_index]

            value += probability * (
                reward + self.gamma * future_value
            )

        return value

    def _policy_backup(self, state_index):
        """
        Calculate T^pi V(s), using the current policy.
        """

        current_action = self.policy[state_index]

        return self._action_value(
            state_index,
            current_action
        )

    def _evaluate_policy(self, deadline):
        """
        Evaluate the current policy with in-place Bellman sweeps.

        A complete sweep is much cheaper than repeatedly propagating
        duplicate predecessor updates through this highly connected model.
        """

        tolerance = 1e-8
        max_sweeps = 250

        for sweep in range(max_sweeps):
            if time.monotonic() >= deadline:
                return

            maximum_change = 0.0

            for state_index in range(self.num_states):
                backed_up_value = self._policy_backup(state_index)
                change = abs(
                    backed_up_value - self.values[state_index]
                )

                if change > maximum_change:
                    maximum_change = change

                self.values[state_index] = backed_up_value

            if maximum_change <= tolerance:
                return

    def _improve_policy(self, deadline):
        """
        Perform policy improvement.

        Returns True if the policy did not change.
        """

        policy_stable = True

        for state_index in range(self.num_states):
            if time.monotonic() >= deadline:
                return False

            old_action = self.policy[state_index]

            best_action = old_action
            best_value = self._action_value(
                state_index,
                old_action
            )

            for action in self.actions:
                if time.monotonic() >= deadline:
                    return False

                candidate_value = self._action_value(
                    state_index,
                    action
                )

                # Keep the first action in a tie. This gives
                # deterministic tie-breaking.
                if candidate_value > best_value + 1e-12:
                    best_value = candidate_value
                    best_action = action

            self.policy[state_index] = best_action

            if best_action != old_action:
                policy_stable = False

        return policy_stable

    def get_action(
        self,
        ship_location,
        pirate_locations,
        treasure_locations
    ) -> int:
        """
        Convert the observed environment state into a state index and
        return the learned policy action.
        """

        pirate_1_location = pirate_locations[0]
        pirate_2_location = pirate_locations[1]

        treasure_mask = 0

        for treasure_index, treasure_location in enumerate(
            self.treasure_locations
        ):
            if treasure_location in treasure_locations:
                treasure_mask |= 1 << treasure_index

        state = (
            ship_location,
            pirate_1_location,
            pirate_2_location,
            treasure_mask
        )

        state_index = self.state_to_index[state]

        return self.policy[state_index]

    def learn_policy(self, time_limit):
        """
        Run policy iteration until convergence or until the time limit
        is reached.
        """

        start_time = time.monotonic()
        deadline = start_time + time_limit

        # Leave time for returning cleanly before run.py's alarm fires.
        deadline -= 0.5

        self.policy_ready = False

        max_policy_iterations = 25

        for _ in range(max_policy_iterations):
            if time.monotonic() >= deadline:
                break

            old_policy = self.policy.copy()

            self._evaluate_policy(deadline)

            policy_stable = self._improve_policy(deadline)

            if policy_stable:
                break

            if self.policy == old_policy:
                break

        self.policy_ready = True