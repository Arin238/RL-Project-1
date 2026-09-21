
import numpy as np
import pdb 
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw, ImageFont

DELTA_T = 0.3
NUM_LANES = 4
FRAME_LENGTH = 10
EPISODE_STEPS = 1000

#collision hyper parameters
COLLISION_THRESH_OTHER_CAR = 4
COLLISION_THRESH_CONTROL_CAR = 0.5

#Rewards
REWARD_COLLISION = -5
REWARD_DISTANCE = 0.1 
REWARD_OVERTRAKE = 1
REWARD_TYPE = 'dist'  #REWARD_TYPE = 'overtakes'

#the sppeds
OTHER_CAR_SAFE_DISTANCE = COLLISION_THRESH_OTHER_CAR
OTHER_CAR_MAX_SPEED = 2
OTHER_CAR_MIN_SPEED = 1
OTHER_CAR_OUT_OFF_CONTEXT_DIST = 5

#the car generation hyperparameters
OTHER_CAR_GENERATION_PROB = 0.25
OTHER_CAR_GENERATION_DIST_FROM_CONTROL_CAR = 4
OTHER_CAR_GENERATION_DIST_FROM_OTHER_CAR = FRAME_LENGTH
OTHER_CAR_GENERATION_TRIES = 4
OTHER_CAR_GENERATION_MAX_LENGTH = 2*FRAME_LENGTH

#the change lane hyper parameters
LANE_CHANGE_ALLOWED = True
LANE_CHANGE_THRESHOLD = 2
LANE_CHANGE_PROB = 0.05
LANE_CHANGE_CONTROL_CAR_THRESHOLD = 3

#the action semantic
ACTION_INCREASE_SPEED = 0
ACTION_DECREASE_SPEED = 1
ACTION_INCREASE_LANE = 2
ACTION_DECREASE_LANE = 3
ACTION_SUCESS_RATE = 0.8
ACTION_NO_OP = 4

#the control car hyper parameters
CONTROL_CAR_DELTA_SPEED = 1
CONTROL_CAR_MAX_SPEED = 4
CONTROL_CAR_MIN_SPEED = 1

#the observation hyper parameters
OBS_TYPE = 'discrete'
OBS_DIST_STATES = 5
OBS_SPEED_STATES = round(CONTROL_CAR_MAX_SPEED - CONTROL_CAR_MIN_SPEED) + 1

#the visulizer hyperparameters
VISUALIZER_LANE_WIDTH = FRAME_LENGTH*15
VISUALIZER_LANE_HEIGHT = 10
VISUALIZER_CAR_HEIGHT = 4
VISUALIZER_CAR_WIDTH = 9
VISUALIZER_LANE_MARKING_HEIGHT = 1
VISUALIZER_LANE_COLOR = np.array([128,128,128])
VISUALIZER_LANE_MARKING_WIDTH = 30
VISUALIZER_LANE_MARKING_COLOR = np.array([255,255,0])
VISUALIZER_CONTROL_CAR_COLOR = np.array([0,255,0])
VISUALIZER_OTHER_CAR_COLOR = np.array([255,0,0])
VISUALIZER_CONTROL_CAR_OFFSET = VISUALIZER_CAR_WIDTH // 2 + 2

class Visualizer:

    def __init__(self):
        self.width = VISUALIZER_LANE_WIDTH
        self.extra_width = VISUALIZER_LANE_MARKING_WIDTH*2
        self.height = NUM_LANES*VISUALIZER_LANE_HEIGHT + (NUM_LANES+1)*VISUALIZER_LANE_MARKING_HEIGHT
        self.pixels_per_unit = VISUALIZER_LANE_WIDTH / FRAME_LENGTH
        self.t = VISUALIZER_LANE_HEIGHT + VISUALIZER_LANE_MARKING_HEIGHT
        self.road = self.road_image()

    def road_image(self):
        height = NUM_LANES*VISUALIZER_LANE_HEIGHT + (NUM_LANES+1)*VISUALIZER_LANE_MARKING_HEIGHT
        width = VISUALIZER_LANE_WIDTH + self.extra_width # VISUALIZER_LANE_MARKING_WIDTH*2
        image = np.zeros((height, width, 3)) + VISUALIZER_LANE_COLOR
        image[:VISUALIZER_LANE_MARKING_HEIGHT] = VISUALIZER_LANE_MARKING_COLOR
        image[-1*VISUALIZER_LANE_MARKING_HEIGHT:] = VISUALIZER_LANE_MARKING_COLOR
        t = VISUALIZER_LANE_HEIGHT + VISUALIZER_LANE_MARKING_HEIGHT 
        for i in range(NUM_LANES - 1):
            for j in range(0, width, 2*VISUALIZER_LANE_MARKING_WIDTH):
                image[t:t+VISUALIZER_LANE_MARKING_HEIGHT,j:j+VISUALIZER_LANE_MARKING_WIDTH] = VISUALIZER_LANE_MARKING_COLOR
            t += VISUALIZER_LANE_MARKING_HEIGHT + VISUALIZER_LANE_HEIGHT
        return image.astype(np.uint8)

    def visualize_car(self, image, position, lane, color, off = 0):
        x = round(position / FRAME_LENGTH * VISUALIZER_LANE_WIDTH) + off
        y = lane*self.t + VISUALIZER_LANE_MARKING_HEIGHT + VISUALIZER_LANE_HEIGHT // 2 - 1
        image[y-VISUALIZER_CAR_HEIGHT//2:y+VISUALIZER_CAR_HEIGHT, x-VISUALIZER_CAR_WIDTH//2:x+VISUALIZER_CAR_WIDTH//2] = color
        return image 
    
    def render(self, env, ignore_control_car = False):
        x = env.control_car.pos 
        image = self.road.copy()
        px = round(x / FRAME_LENGTH * VISUALIZER_LANE_WIDTH)
        off = (px % self.extra_width)
        image = image[:,off:off+self.width]
        if(not ignore_control_car):
            lane = env.control_car.lane_id
            image = self.visualize_car(image, 0, lane, VISUALIZER_CONTROL_CAR_COLOR, off = VISUALIZER_CONTROL_CAR_OFFSET)
        for lane in env.lanes:
            for car in lane.cars:
                if(car.pos < x):
                    continue
                if(car.pos - x < FRAME_LENGTH):
                    #print(car)
                    image = self.visualize_car(image, car.pos - x, car.lane_id, VISUALIZER_OTHER_CAR_COLOR, off = VISUALIZER_CONTROL_CAR_OFFSET)
        return image

def sample(min_x, max_x):
    return np.random.randint(min_x, max_x+1)

def sample_cont(min_x, max_x):
    return np.random.random()*(max_x - min_x) + min_x

class Car:

    def __init__(self, lane_id, pos, speed) -> None:
        self.lane_id = lane_id
        self.pos = pos 
        self.speed = speed
    
    def __eq__(self, value: object) -> bool:
        return self.lane_id == value.lane_id and self.pos == value.pos
    
    def __hash__(self) -> int:
        return hash((self.lane_id, self.pos))
    
    def __lt__(self, other: object) -> bool:
        return self.pos < other.pos

    def __le__(self, other: object) -> bool:
        return self.pos <= other.pos

    def __str__(self) -> str:
        return f'pox: {self.pos}, speed: {self.speed}'
    
    def step(self) -> None:
        self.pos = self.pos + self.speed*DELTA_T
        return self.speed*DELTA_T
    
class Lane:

    def __init__(self, lane_id) -> None:
        self.lane_id = lane_id
        self.cars = []

    def add_car(self, car):
        self.cars.append(car)
        self.cars.sort()

    def ahead(self, car):
        prev_car = None 
        for car_i in reversed(self.cars):
            if(car_i.pos < car.pos):
                break 
            prev_car = car_i
        return prev_car
    
    def reset(self):
        self.cars =  []

    def behind(self, car):
        prev_car = None 
        for car_i in self.cars:
            if(car_i.pos > car.pos):
                break 
            prev_car = car_i
        return prev_car

    def modulate_speeds(self):
        prev_car = None
        for car in self.cars:
            if(prev_car is not None):
                if(prev_car.pos + OTHER_CAR_SAFE_DISTANCE > car.pos):
                    prev_car.speed = min(prev_car.speed, car.speed)
            prev_car = car 
    
    def remove_cars(self, x):
        if(len(self.cars) == 0):
            return 0
        index = 0
        for car in self.cars:
            if(car.pos + COLLISION_THRESH_CONTROL_CAR < x):
                index += 1 
            else:
                break 
        self.cars = self.cars[index:]
        return index
        
    def remove_car(self, car):
        for index, car_i in enumerate(self.cars):
            if(car_i == car):
                del self.cars[index]
                return
    
    def generate_cars(self, x, first = False):
        if(len(self.cars) == 0):
            if(first):
                start_pos = x + OTHER_CAR_GENERATION_DIST_FROM_CONTROL_CAR
            else:
                start_pos = x + FRAME_LENGTH
        else:
            start_pos = self.cars[-1].pos + OTHER_CAR_GENERATION_DIST_FROM_OTHER_CAR
        if(start_pos > x + OTHER_CAR_GENERATION_MAX_LENGTH):
            return 

        for i in range(OTHER_CAR_GENERATION_TRIES):
            if(np.random.random() < OTHER_CAR_GENERATION_PROB):
                speed = sample_cont(OTHER_CAR_MIN_SPEED, OTHER_CAR_MAX_SPEED)
                car = Car(self.lane_id, start_pos+i,speed)
                self.add_car(car)
                break
    
    def min_dis(self, x):
        
        if(len(self.cars) == 0):
            return 0
        if(OBS_TYPE !=  'discrete'):
            return min(max(self.cars[0].pos - x, 0),10) / 10 # / 2
        if(self.cars[0].pos < x):
            return 1
        if(OBS_TYPE == 'discrete'):
            state = (self.cars[0].pos - x)*(OBS_DIST_STATES-2) / FRAME_LENGTH + 1
            state = int(min(round(state), OBS_DIST_STATES-1))
            return state
        
    
    def step(self):
        for c in self.cars:
            c.step()
        
    def __str__(self) -> str:
        s = f'Lane: {self.lane_id}\n'
        for c in self.cars:
            s += str(c) + '\n'
        return s 

class HighwayEnv:

    def __init__(self):
        self.num_lanes = NUM_LANES
        self.num_speed_states = OBS_SPEED_STATES
        self.num_dist_states = OBS_DIST_STATES

        self.lanes = []
        for i in range(self.num_lanes):
            self.lanes.append(Lane(i))
        self.control_car = Car(lane_id=self.num_lanes//2, pos = 2, speed=1)
        self.reset()
        self.visualizer = Visualizer()
        if(OBS_TYPE == 'discrete'):
            self.num_states = OBS_DIST_STATES**self.num_lanes*OBS_SPEED_STATES*NUM_LANES
        else:
            self.num_states = None 
        self.num_actions = 5
    
    def act(self, action):
        if(np.random.rand() < ACTION_SUCESS_RATE):
            if(action == ACTION_DECREASE_LANE):
                self.control_car.lane_id = max(self.control_car.lane_id-1, 0)
            elif(action == ACTION_INCREASE_LANE):
                self.control_car.lane_id = min(self.control_car.lane_id+1, self.num_lanes-1)
            elif(action == ACTION_INCREASE_SPEED):
                self.control_car.speed = self.control_car.speed + CONTROL_CAR_DELTA_SPEED
                self.control_car.speed = min(self.control_car.speed, CONTROL_CAR_MAX_SPEED)
            elif(action == ACTION_DECREASE_SPEED):
                self.control_car.speed = self.control_car.speed - CONTROL_CAR_DELTA_SPEED
                self.control_car.speed = max(self.control_car.speed, CONTROL_CAR_MIN_SPEED)
            else:
                if(action != ACTION_NO_OP):
                    raise NotImplementedError(f"Action {action} not defined!!")
        reward = self.control_car.step()
        return reward*REWARD_DISTANCE
    
    def check_collision(self, car, lane, thresh):
        car_ahead = self.lanes[lane].ahead(car)
        car_behind = self.lanes[lane].behind(car)
        collided = False 
        if(car_behind is not None):
            collided = abs(car_behind.pos - car.pos) < thresh
        if(not collided and car_ahead is not None):
            collided = abs(car_ahead.pos - car.pos) < thresh
        return collided
    
    def step_other_cars(self):
        for lane in self.lanes:
            lane.step()
    
    def change_lanes(self):
        x = self.control_car.pos
        for lane in self.lanes:
            allowed_lanes = [min(lane.lane_id + 1, self.num_lanes-1), max(lane.lane_id-1,0)]
            for car in lane.cars:
                if(car.pos < x + LANE_CHANGE_THRESHOLD):
                    continue
                if(np.random.random() < LANE_CHANGE_PROB):
                    new_lane = np.random.choice(allowed_lanes)
                    if(not self.check_collision(car, new_lane, COLLISION_THRESH_OTHER_CAR)):
                        lane.remove_car(car)
                        car.lane_id = new_lane
                        self.lanes[new_lane].add_car(car)
                        break

    def modulate_speeds(self):
        for lane in self.lanes:
            lane.modulate_speeds()
    
    def remove_cars(self):
        x = self.control_car.pos 
        cars_removed = 0
        for lane in self.lanes:
            cars_removed += lane.remove_cars(x)
        return cars_removed*REWARD_OVERTRAKE 
    
    def add_cars(self):
        x = self.control_car.pos
        lanes = np.random.choice(self.lanes, NUM_LANES//2)
        for lane in lanes:
            lane.generate_cars(x, self.steps == 0)

    def get_obs(self):
        x = self.control_car.pos 
        speed = round(self.control_car.speed)
        speed = speed - CONTROL_CAR_MIN_SPEED
        #speed = min(speed, OBS_SPEED_STATES-1)
        min_dis = []
        for lane in self.lanes:
            min_dis.append(lane.min_dis(x))
        return (speed, self.control_car.lane_id, min_dis)

    def get_state(self):
        return self.get_obs()

    def step(self, action = None, ignore_control_car = False):
        self.steps += 1
        if(not ignore_control_car):
            if(self.check_collision(self.control_car, self.control_car.lane_id, COLLISION_THRESH_CONTROL_CAR)):
                reward = REWARD_COLLISION
                self.done = True 
                obs = self.get_obs()
                return obs, reward, self.done
        if(not ignore_control_car):
            reward_1 = self.act(action)
        if(ignore_control_car):
            self.control_car.pos += 1
            reward_1 = 0
        self.step_other_cars()
        if(LANE_CHANGE_ALLOWED):
            self.change_lanes()
        self.add_cars()
        reward_2 = self.remove_cars()
        self.modulate_speeds()
        obs = self.get_obs()
        reward = reward_2 if REWARD_TYPE =='overtakes' else reward_1
        if(self.steps == EPISODE_STEPS):
            self.done = True
        return obs, reward, self.done

    def reset(self, seed = None):
        if(seed is not None):
            np.random.seed(seed)
        else:
            np.random.seed(None) 
        self.steps = 0
        for lane in self.lanes:
            lane.reset()
        self.add_cars()
        self.done = False
        self.control_car = Car(self.num_lanes//2, 0, CONTROL_CAR_MIN_SPEED + 1)
        return self.get_obs()

    def get_all_lane_states(self):
        obs = []
        for i in range(NUM_LANES):
            self.control_car.lane_id = i 
            for j in range(CONTROL_CAR_MIN_SPEED, CONTROL_CAR_MAX_SPEED+1):
                self.control_car.speed = j
                obs.append(self.get_obs())
        return obs  

    def get_all_speed_states(self):
        obs = []
        for j in range(CONTROL_CAR_MIN_SPEED, CONTROL_CAR_MAX_SPEED+1):
            self.control_car.speed = j
            obs.append(self.get_obs())
        return obs  

    def render_lane_state_values(self, scores)-> np.ndarray:
        image = self.visualizer.render(self, True)
        scores = np.array(scores).reshape(NUM_LANES, CONTROL_CAR_MAX_SPEED - CONTROL_CAR_MIN_SPEED + 1)
        scores = scores.sum(axis = -1, keepdims= True)
        max_score = np.max(scores) + 1e-5
        patches = []
        for i, si in enumerate(scores):
            patches.append([])
            for j, s in enumerate(si):
                pij = np.zeros((10,10,3))
                pij += int(s / max_score*255) 
                pij = np.pad(pij, ((1,0),(1,1),(0,0)), constant_values = 0)
                patches[-1].append(pij)
        patches = np.array(patches)
        patches = np.moveaxis(patches, [0, 1, 2, 3, 4], [0, 2, 1, 3, 4])
        patches = patches.reshape(11*NUM_LANES, 12*scores.shape[1], 3)
        patches = np.pad(patches, ((1,0),(0,0),(0,0)), constant_values = 0).astype('uint8')
        image = np.concatenate([patches, image], axis = 1)
        return np.array(image) 

    def render_speed_state_values(self, scores)-> np.ndarray:
        image = self.visualizer.render(self, False)
        scores = np.array(scores).reshape(CONTROL_CAR_MAX_SPEED - CONTROL_CAR_MIN_SPEED + 1)
        max_score = np.max(scores) + 1e-5
        patches = []
        for i, s in enumerate(scores):
            pij = np.zeros((15,30,3))
            pij += int(s / max_score*255) 
            pij = np.pad(pij, ((2,2),(3,3),(0,0)), constant_values = 0)
            patches.append(pij)
        patches = np.array(patches)
        patches = np.moveaxis(patches, [0, 1, 2, 3], [1, 0, 2, 3])
        patches = patches.reshape(19, 36*NUM_LANES, 3)
        patches = np.pad(patches, ((0,0),(3,3),(0,0)), constant_values = 0).astype('uint8')
        image = np.concatenate([patches, image], axis = 0)
        return np.array(image) 
    
    def render(self) -> np.ndarray:
        image = self.visualizer.render(self, False)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        width, _ = image.size
        text = str(round(self.control_car.pos)) + "," + str(round(self.control_car.speed))
        bbox = draw.textbbox((0, 0), text, font=font)  
        text_width = bbox[2] - bbox[0]
        position = (width - text_width - 10, 0)  
        draw.text(position, text, fill="white", font=font)   
        return np.array(image) 

    def __str__(self):
        s = f"Control car: lane: {self.control_car.lane_id}, {str(self.control_car)}\n"
        for lane in self.lanes:
            s += str(lane) + '\n'
        return s

def get_highway_env(dist_obs_states = 5, reward_type = 'dist', obs_type = 'discrete') -> HighwayEnv:
    global OBS_DIST_STATES, REWARD_TYPE, OBS_TYPE
    OBS_DIST_STATES = dist_obs_states
    REWARD_TYPE = reward_type
    OBS_TYPE = obs_type
    env = HighwayEnv()
    return env
