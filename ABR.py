# import tensorflow as tf
#NN_MODEL = "./submit/results/nn_model_ep_18200.ckpt" # model path settings, if using ML-based method
M_IN_K = 1000.0
FUTURE_P = 5
import numpy as np
from comyco_pitree import predict
from statsmodels.stats.weightstats import DescrStatsW
from Network import ActorNetwork
from torch.distributions import Categorical
import torch

S_INFO = 6
S_LEN = 8
A_DIM = 6
VIDEO_BIT_RATE = [500.0, 850, 1200.0,1850.0, 1850.0, 1850.0]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0

class Algorithm:
    def __init__(self):
    # fill your self params
       self.some_param = 0
       self.day = 3
       self.history_bit_rate = []
       self.model_path = "actor.pt"

       self.net = ActorNetwork([S_INFO,S_LEN],A_DIM)
       if torch.cuda.is_available():
           self.net.load_state_dict(torch.load(self.model_path))
       else: 
           self.net.load_state_dict(torch.load(self.model_path, map_location='cpu'))
       self.state=torch.zeros((S_INFO,S_LEN))
       self.last_bit_rate = 1
   
    # Initail
    def Initial(self):
    # Initail your session or something
       self.some_param = 0

    def updateHistoryBitRate(self, bitRate) :
        self.history_bit_rate.append(bitRate)
    
    def adjustBitRate(self) :
        while len(self.history_bit_rate) > self.day :
            self.history_bit_rate.pop(0)
        
        return sum(self.history_bit_rate) / len(self.history_bit_rate)    

    def mean_var(self, throu, delay):
        thr_, delay_ = np.array(throu), np.array(delay)
        weighted_stats = DescrStatsW(thr_, weights=delay_)
        return weighted_stats.mean, weighted_stats.std

    # Define your algo
    def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,S_end_delay, S_decision_flag, S_buffer_flag,S_cdn_flag,S_skip_time, end_of_video, cdn_newest_id,download_id,cdn_has_frame,IntialVars):

        # If you choose the marchine learning
        '''state = []

        state[0] = ...
        state[1] = ...
        state[2] = ...
        state[3] = ...
        state[4] = ...

        decision = actor.predict(state).argmax()
        bit_rate, target_buffer = decison//4, decison % 4 .....
        return bit_rate, target_buffer'''


        state = torch.roll(self.state,-1,dims=-1)

        state[0, -1] = VIDEO_BIT_RATE[self.last_bit_rate] / (float(np.max(VIDEO_BIT_RATE))+0.000000001)# last quality
        state[1, -1] = S_buffer_size[-1] / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(S_chunk_len[-1]) / (float(S_end_delay[-1]) + 0.000000001) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(S_end_delay[-1]) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = torch.tensor(S_chunk_len[-1]) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = min(S_chunk_len[-1], CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        with torch.no_grad():
            probability=self.net.forward(state.unsqueeze(0))
            m=Categorical(probability)
            bit_rate=m.sample().item()

        self.state = state
        self.last_bit_rate = bit_rate

        if bit_rate > 3 :
            bit_rate = 3


        """
        # If you choose BBA
        RESEVOIR = 0.5
        CUSHION =  1.5
        
        if S_buffer_size[-1] < RESEVOIR:
            bit_rate = 0    
        elif S_buffer_size[-1] >= RESEVOIR + CUSHION and S_buffer_size[-1] < CUSHION +CUSHION:
            bit_rate = 2
        elif S_buffer_size[-1] >= CUSHION + CUSHION:
            bit_rate = 3
        else:
            bit_rate = 1
        """


        # self.updateHistoryBitRate(bit_rate)
        # rba_bit_rate = self.adjustBitRate()

        # bit_min = 0
        # bit_max = 3
        # time = time + 0.0000000001
        # fuck = rba_bit_rate * (bit_rate + time) / time
        # bit_rate = min(max(fuck, bit_min), bit_max)
        """
        throu_array, delay_array = [], []


        throughput = S_chunk_len[-1] / (S_end_delay[-1] + 0.000000001) / M_IN_K
        throu_array.append(throughput)
        delay_array.append(S_end_delay[-1])
        if len(throu_array) >= FUTURE_P:
            throu_array.pop(0)
            delay_array.pop(0)
        mean, var = self.mean_var(throu_array, delay_array)
        action_prob = predict(66.2045315334, S_buffer_size[-1], mean, var)
        bit_rate = np.argmax(action_prob)


        """



        #pitree : VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
        # our :       bit_rate: {500, 850, 1200, 1850} kbps
        """



        if bit_rate >= 4 :
            bit_rate = 3
        """

        target_buffer = 1
        latency_limit = 2

        # bit_rate = int(np.random.uniform(0, 4, 1)[0])

        return bit_rate, target_buffer, latency_limit

         # If you choose other
         #......



    def get_params(self):
    # get your params
       your_params = []
       return your_params
