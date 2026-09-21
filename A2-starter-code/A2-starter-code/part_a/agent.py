class Agent:

    def __init__(self, layout_file, prob_file):
        """
        Initialize the agent.

        Args:
            layout_file: Path to the grid layout file.
            prob_file: Path to the file containing environment probabilities.

        You may use this function to:
            - Read and store the grid layout.
            - Read and store transition probabilities.
            - Identify important locations such as the fort.
            - Construct the state space and transition model.
            - Initialize any data structures required for learning.
        """

        # TODO


    def get_action(self, ship_location, pirate_locations, treasure_locations) -> int:
        """
        Choose an action for the current state.

        Args:
            ship_location: Tuple (x, y) representing the current
                location of the ship.

            pirate_locations: List containing the locations of all
                pirates. There can be at most 2 pirates.

            treasure_locations: List containing the locations of all
                treasures. There can be at most 2 treasures.

        Returns:
            An integer representing the selected action:

                0 -> UP
                1 -> DOWN
                2 -> LEFT
                3 -> RIGHT

        Note:
            This function may be called multiple times after
            `learn_policy()` has been executed.
        """

        # TODO
        return 0


    def learn_policy(self, time):
        """
        Learn a policy for navigating the Treasure Hunt environment.

        Args:
            time: Maximum time (in seconds) allowed for learning.

        The learned policy should be stored internally by the agent
        and subsequently used by `get_action()`.

        Returns:
            None
        """

        # TODO