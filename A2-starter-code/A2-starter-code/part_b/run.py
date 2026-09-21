from ta_agents import Agent
from env import HighwayEnv
import signal
import argparse
import os 
import imageio 
parser = argparse.ArgumentParser(
    description="Run the experiment."
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

parser.add_argument(
    "--df", 
    type=float,
    required=True,
    help="Discount Factor for the environment"
)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def run(args, visualize = True):
    
    num_runs = args.num_runs
    T = args.T
    output_dir = args.output_dir
    df = args.df
    #get the agent and the environemnt
    env = HighwayEnv()
    agent = Agent(env, discount_factor= df)
    

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
        env = HighwayEnv()

        frames = []
        rewards = 0
        steps = 0

        #the intial state
        state = env.get_state()
        while (not env.done):

            #get the actions
            action = agent.get_action(*state)
            state, reward, done = env.step(action)

            #get the frames and rewards
            rewards = reward + df*rewards
            if(visualize):
                frames.append(env.render())
            steps+=1
        
        rewards_all.append(rewards)
        if(visualize):
            imageio.mimsave(os.path.join(output_dir, f"{i}.gif"), frames, duration=0.7, loop=0)

    return (sum(rewards_all) /num_runs)





if __name__ == "__main__":

    args = parser.parse_args()
    score = run(args, visualize= True)
    print("score: ", score)


