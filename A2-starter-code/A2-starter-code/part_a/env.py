
from copy import deepcopy
import random
from turtle import left
import numpy as np
import sys 
import numpy as np
from PIL import Image 
import cv2
import pdb
import matplotlib.pyplot as plt 
class Grid:

    def __init__(self, 
        locations = dict(),
        N=10):
        
        self.B = 3 
        self.color = [0.2588, 0.4039, 0.6980]
        self.brightness = 1.8
        self.grid = []
        self.P = 3
        self.W = 25 
        self.M = N 
        self.N = N
        self.ew = self.W + self.P//2

        self.locations = locations
        self.icons = dict()
        for i in self.locations:
            if(i != 'pirate_area'):
                self.icons[i] = self.load_image(f"images/{i}.png")

        #generate the grid
        self.grid = self.generate_grid()
        
    def load_image(self, path):

        image = Image.open(path)
        image = cv2.resize(np.array(image), (self.W,self.W))
        image = image.astype('float') / 255
        return image
        
    def get_real_coordinates(self, i, j):

        x = j
        y = self.N - 1 - i
        return (y,x)
    
    def put_icon(self, grid, icon):

        gg = icon[:,:,3:]*icon[:,:,:3]
        grid[self.P//2:-1*self.P//2, self.P//2:-1*self.P//2] = gg + (1-icon[:,:,3:])*grid[self.P//2:-1*self.P//2, self.P//2:-1*self.P//2]
        return grid

    def generate_grid(self):
        grid = []

        #iterate over each grid cell
        for i in range(self.M):
            
            #the row
            row_i = []

            #for each row
            for j in range(self.N):
                
                #the i and j th grid cell
                grid_ij = np.ones((self.W+self.P, self.W+self.P, 3))

                #paint with colors  
                for c in range(3):
                    grid_ij[:,:,c] = self.color[c]*(0.7 + (self.brightness - 0.9)/60*(i+30-j))

                #see if any icon should be here
                real_coordinates = self.get_real_coordinates(i,j)
                
                for k in self.locations:
                    if(real_coordinates in self.locations[k]):
                        if(k == 'pirate_area'):
                            grid_ij = 0.7*grid_ij
                        else:
                            grid_ij = self.put_icon(grid_ij, self.icons[k])
                 
                #make white
                grid_ij[:self.P//2,:,:] = 0
                grid_ij[-self.P//2:,:,:] = 0

                grid_ij[:,:self.P//2,:] = 0
                grid_ij[:,-self.P//2:,:] = 0
                
                #append to row_i
                row_i.append(grid_ij)
        
            #append to grid
            grid.append(row_i)
        return grid

    def get_grid(self, x, y):
        grid = self.grid[self.M - y - 1][x]
        return grid 

    def draw_one_step_grid(self, sp, dp, grid, color):
        #print(sp, dp)
        if(sp[0] == dp[0]):

            grid[sp[0]-self.P//2:dp[0] + self.P//2, sp[1]:dp[1],:] *= 0
            for c in range(3):
                grid[sp[0]-self.P//2:dp[0] + self.P//2, sp[1]:dp[1],c] = color[c]
            #print(grid[sp[0]-self.P//2:dp[0] + self.P//2, sp[1]:dp[1],0].shape)
        else:
            grid[sp[0]:dp[0], sp[1]-self.P//2:dp[1] + self.P//2,:] *= 0
            for c in range(3):
                grid[sp[0]:dp[0], sp[1]-self.P//2:dp[1] + self.P//2,c] = color[c]
            #print(grid[sp[0]:dp[0], sp[1]-self.P//2:dp[1] + self.P//2,0].shape)
        
    def draw_one_step(self, sp, dp, color):

        delta_x = dp[0] - sp[0] 
        delta_y = dp[1] - sp[1]
        
        sgrid = self.get_grid(sp[0]-1, sp[1]-1) 
        dgrid = self.get_grid(dp[0]-1, dp[1]-1)
        if(delta_x == delta_y):
            return
        if(delta_x == -1):
            self.draw_one_step_grid((self.ew//2 + 1,0),(self.ew//2 + 1, self.ew//2+1), sgrid, color)
            self.draw_one_step_grid((self.ew//2 + 1, self.ew//2 + 1),(self.ew//2+1,self.W + self.P), dgrid, color)
            
        elif(delta_x == 1):
            self.draw_one_step_grid((self.ew//2+1, 0),(self.ew//2 + 1, self.ew//2 + 1), dgrid, color)
            self.draw_one_step_grid((self.ew//2 + 1, self.ew//2 + 1),(self.ew//2+1, self.W + self.P), sgrid, color)
        elif(delta_y == -1):
            self.draw_one_step_grid((0 ,self.ew//2 + 1),( self.ew//2 + 1, self.ew//2+1), dgrid, color)
            self.draw_one_step_grid((self.ew//2 + 1, self.ew//2 + 1),(self.W + self.P, self.ew//2+1), sgrid, color)
            
        elif(delta_y == 1):
            self.draw_one_step_grid((0, self.ew//2+1),(self.ew//2 + 1, self.ew//2 + 1), sgrid, color)
            self.draw_one_step_grid((self.ew//2 + 1, self.ew//2 + 1),(self.W + self.P,self.ew//2+1), dgrid, color)
            
        return

    def draw_path(self, sequence, color = [0,0,1]):

        start_x, start_y = sequence[0]
        start_grid = self.get_grid(start_x-1,start_y-1)
        start_grid[self.P//2:-1*self.P//2,self.P//2:-1*self.P//2] *= 0
        start_grid[self.P//2:-1*self.P//2,self.P//2:-1*self.P//2,0] += 1
        

        end_x, end_y = sequence[-1]
        end_grid = self.get_grid(end_x-1,end_y-1)
        end_grid[self.P//2:-1*self.P//2,self.P//2:-1*self.P//2] *= 0
        end_grid[self.P//2:-1*self.P//2,self.P//2:-1*self.P//2,1] += 1
        
        for i in range(1,len(sequence)):
            self.draw_one_step(sequence[i-1],sequence[i], color)
    
    def show(self):

        grid = [np.concatenate(row_i, axis = 1) for row_i in self.grid]
        grid = np.concatenate(grid, axis = 0)
        grid = np.clip(grid, 0, 1)
        grid_big = np.zeros((grid.shape[0] + 2*self.B, grid.shape[1] + 2*self.B, 3))
        grid_big[:,:,0] = 1
        grid_big[:,:,1] = 1
        grid_big[self.B:-1*self.B,self.B:-1*self.B] = grid

        grid = (grid_big*255).astype('uint8')
        return grid
        
    def clear(self):
        self.grid = self.generate_grid()


UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
DIRECTIONS  = set([UP, DOWN, LEFT, RIGHT]) 
LAND_CHAR = "L"
PIRATE1_CHAR = "1"
PIRATE2_CHAR = "2"
PIRATE_AREA_CHAR = "!"
WATER_CHAR = "W"
TREASURE_CHAR = "T"
FORT_CHAR = 'F'
SHIP_CHAR = 'S'
POSSIBLE_CHARS = set([LAND_CHAR, PIRATE1_CHAR, PIRATE2_CHAR, WATER_CHAR, TREASURE_CHAR, FORT_CHAR, SHIP_CHAR, PIRATE_AREA_CHAR])

class TreasureHunt():

    def __init__(self, layout_file = None, prob_file = None):
        
        assert layout_file is not None 
        assert prob_file is not None
        self.locations = self.read_layout_file(layout_file)
        self.original_locations = deepcopy(self.locations)
        self.ship_prob, self.pirate_prob, self.rewards, self.df = self.read_prob_file(prob_file)
     
        #the grid size
        self._action_delta = [[1,0],[-1,0],[0,-1],[0,1]]
        self.action_name = ['up','down','left','right']

        #check pirate area
        self.pirate_areas = self.get_pirate_areas(self.locations['pirate_area'])
        assert len(self.pirate_areas) == 2
        assert len(self.pirate_areas[0]) >= 1 and len(self.pirate_areas[1]) >= 1

        #get the number of treasures
        self.num_treasures = len(self.locations['treasure'])
        assert self.num_treasures == 2

        #count the states
        num_cells = self.N**2
        self.num_states = num_cells*len(self.pirate_areas[0])*len(self.pirate_areas[1])*4
        self.num_actions = 4

        #the treasure indicator
        self.done = False

        #first state
        self.state = self.get_state()
    
    def get_state(self):

        #get the locations and treasures
        return self.locations['ship'][0], self.locations['pirate'], self.locations['treasure']
    
    def _dfs(self, stack, visited):

        area = []
        while len(stack) > 0:
            loc = stack.pop(-1)
            area.append(loc)
            #print(loc)
            for i in DIRECTIONS:
                nloc = self.move(loc, i)
                if(nloc in visited and not visited[nloc]):
                    visited[nloc] = True
                    stack.append(nloc)
        return area 
    
    def get_pirate_areas(self, locations):
        visited = {loc: False for loc in locations}
        stack = [locations[0]]
        visited[locations[0]] = True
        p1_area = self._dfs(stack, visited)
        assert len(p1_area) != len(visited)
        stack = []
        for loc in visited:
            if(not visited[loc]):
                visited[loc] = True
                stack = [loc]
                break 
        p2_area = self._dfs(stack, visited)
        #print(len(p1_area), len(p2_area), len(visited))
        assert len(p1_area) + len(p2_area) == len(visited)
        
        #see which one to go for
        if(self.locations['pirate'][0] in p2_area):
            p1_area, p2_area = p2_area, p1_area

        return [p1_area, p2_area]

    def read_prob_file(self, prob_file):

        with open(prob_file, "r") as f:

            probs = f.readlines()
            assert len(probs) == 5

            #ship prob
            ship_prob = float(probs[0].strip())
            ship_prob = [ship_prob, 1-ship_prob]
            
            #the pirate prob 
            p1_prob = probs[1].strip().split(' ')
            p1_prob = [float(i) for i in p1_prob]
            
            #the pirate prob 
            p2_prob = probs[2].strip().split(' ')
            p2_prob = [float(i) for i in p2_prob]
            
            #the reward 
            rewards = probs[3].strip().split(' ')
            rewards = [float(i) for i in rewards]
            rewards = {'step': rewards[0],
                       'treasure': rewards[1],
                       'fort': rewards[2],
                       'pirate': rewards[3]}

            #the discount factor
            df = float(probs[4].strip())
            return ship_prob, [p1_prob, p2_prob], rewards, df

    
    def read_layout_file(self, layout_file):

        #read lines 
        with open(layout_file, "r") as f:
            
            rows = f.readlines()
            self.N = len(rows)
            pirate1_loc = None 
            pirate2_loc = None
            locations = {
                'ship': [],
                'land': [],
                'fort': [],
                'pirate_area': [],
                'pirate': [],
                'treasure': []
            }
            for i,row in enumerate(rows):
                row = row.strip()
                for j,c in enumerate(row):
                    if(c not in POSSIBLE_CHARS):
                        raise ValueError(f"Incorrect character in layout file {c}")
                    if(c == PIRATE1_CHAR):
                        pirate1_loc = (i,j)
                        locations['pirate_area'].append((i,j))
                    if(c == PIRATE2_CHAR):
                        pirate2_loc = (i,j)
                        locations['pirate_area'].append((i,j))
                    if(c == LAND_CHAR):
                        locations['land'].append((i,j))
                    if(c == TREASURE_CHAR):
                        locations['treasure'].append((i,j))
                    if(c == PIRATE_AREA_CHAR):
                        locations['pirate_area'].append((i,j))
                    if(c == SHIP_CHAR):
                        locations['ship'].append((i,j))
                    if(c == FORT_CHAR):
                        locations['fort'].append((i,j))
                    locations['pirate'] = [pirate1_loc, pirate2_loc]
            #asser the validity of pirate area

            return locations

    def reset(self):
        self.locations = deepcopy(self.original_locations)
        self.state = self.get_state()
        self.done = False
        return self.state
    
    def sample_action(self, prob):

        #sample the action
        action =  int(np.random.multinomial(1, prob).nonzero()[0][0])
        return action

    def sample_random_actions(self, excluded_actions):
        valid_actions = DIRECTIONS - excluded_actions
        return np.random.choice(list(valid_actions))

    def move(self, location, action):
        direction = self._action_delta[action]
        return (location[0] + direction[0], location[1] + direction[1])
    
    def ship_loc_validity(self, ship_loc, check_for_land = True):

        for l in ship_loc:
            if(l >= self.N or l < 0):
                return False 
        
        if(check_for_land and ship_loc in self.locations['land']):
            return False
        return True
    
    def transition(self, action):
        
        #transition the pirate 1 
        for i in range(2):
            p_action = self.sample_action(self.pirate_prob[i])
            p_location = self.move(self.locations['pirate'][i], p_action)
            if(p_location not in self.locations['pirate_area']):
                p_location = self.locations['pirate'][i]
            # excluded_actions = set([p_action])
            # while p_location not in self.locations['pirate_area']:
            #     p_action = self.sample_random_actions(excluded_actions)
            #     p_location = self.move(self.locations['pirate'][i], p_action)
            #     excluded_actions = set([p_action])
            self.locations['pirate'][i] = p_location
   
        #transition the ship
        if(random.random() > self.ship_prob[0]):
            action = self.sample_random_actions(set([action]))
        ship_loc = self.move(self.locations['ship'][0], action)
        if(not self.ship_loc_validity(ship_loc)):
            ship_loc = self.locations['ship'][0]
        # #excluded_actions = set([action])
        # while not self.ship_loc_validity(ship_loc):
        #     print(ship_loc)
        #     action = self.sample_random_actions(excluded_actions)
        #     ship_loc = self.move(self.locations['ship'][0], action)
        #     excluded_actions.add(action)  
        self.locations['ship'][0] = ship_loc  

    def step(self, action):

        if(self.done):
            return self.get_state(), 0, self.done
        
        #transition the env
        self.transition(action)

        #check for done and reward 
       
        ship_loc = self.locations['ship'][0]
        # print()
        # print(ship_loc)
        # print(self.locations['pirate'])
        reward = self.rewards['step']
        if(ship_loc in self.locations['pirate']):
            self.done = True 
            reward += self.rewards['pirate'] 
        elif(ship_loc in self.locations['fort']):
            self.done = True 
            reward += self.rewards['fort']
        elif(ship_loc in self.locations['treasure']):
            reward +=  self.rewards['treasure']
            self.locations['treasure'].remove(ship_loc)
            
        return self.get_state(), reward, self.done

    
    def render(self):
        grid = Grid(self.locations, N = self.N)
        image = grid.show()
        return image

    

if __name__ == '__main__':

    layout_file = sys.argv[1]
    prob_file = sys.argv[2]
    env = TreasureHunt(layout_file, prob_file)

    img = env.render()
    plt.imshow(img)
    plt.show()
    env.step(RIGHT)
    
    img = env.render()
    plt.imshow(img)
    plt.show()
      