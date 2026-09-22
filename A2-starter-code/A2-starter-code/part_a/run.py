from agent import Agent
from env import TreasureHunt
import argparse
import os 
import imageio 
import signal

parser = argparse.ArgumentParser(
    description="Run the experiment."
)

parser.add_argument(
    "--layout_file",
    type=str,
    required=True,
    help="Path to the layout file"
)

parser.add_argument(
    "--prob_file",
    type=str,
    required=True,
    help="Path to the probability file"
)

parser.add_argument(
    "--num_runs",
    type=int,
    required=True,
    help="Number of runs"
)

parser.add_argument(
    "--T",
    type=int,
    required=True,
    help="Value of T"
)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory for output files"
)


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def run(args, visualize = True):

    layout_file = args.layout_file
    prob_file = args.prob_file
    num_runs = args.num_runs
    T = args.T
    output_dir = args.output_dir

    if visualize:
        os.makedirs(output_dir, exist_ok=True)

    #get the agent and the environemnt
    agent = Agent(layout_file, prob_file)
    

    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(T)

    try:
        agent.learn_policy(T)
    except TimeoutException:
        print(f"learn_policy exceeded {T} seconds. Stopping...")
        return None
    finally:
        signal.alarm(0)  # Cancel alarm

    rewards_all = []
    for i in range(num_runs):
        env = TreasureHunt(layout_file, prob_file)

        frames = []
        rewards = 0
        steps = 0

        #the intial state
        state = env.get_state()
        while (not env.done) and (steps < 2*(env.N**2)):

            #get the actions
            action = agent.get_action(*state)
            state, reward, done = env.step(action)

            #get the frames and rewards
            rewards = reward + env.df*rewards
            if(visualize):
                frames.append(env.render())
            steps+=1
        
        rewards_all.append(rewards)

        if(visualize):
            imageio.mimsave(os.path.join(output_dir, f"{i}.gif"), frames, duration=0.5, loop=0)

    return sum(rewards_all) /num_runs

if __name__ == "__main__":

    args = parser.parse_args()
    score = run(args, visualize = True)
    print("score: ", score)
