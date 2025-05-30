from sglang.srt.predictor.model.models import LightGBMModel
from sglang.srt.predictor.base_predictor import BinaryPredictor
from sglang.srt.predictor.base_predictor import ReuseDistancePredictor

import lightgbm as lgb
import numpy as np
import collections
import json
import os

from concurrent.futures import ThreadPoolExecutor

class LRBReuseDistancePredictor(ReuseDistancePredictor):
    def __init__(self, memory_window=1000000):
        super().__init__()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_config = None
        with open(os.path.join(current_dir, 'model/checkpoints/lightgbm/model_config.json'), "r") as f:
            model_config = json.load(f)
            deltanums = model_config['delta_nums']
            edcnums = model_config['edc_nums']

        this_dir = os.path.join(current_dir, 'model/checkpoints/lightgbm')
        if not os.path.exists(this_dir):
            raise ValueError(f'Benchmark: {this_dir} not found checkpoints')
        this_ckpt_path = os.path.join(this_dir, f'{deltanums}_{edcnums}.txt')
        if not os.path.exists(this_ckpt_path):
            raise ValueError(f'Benchmark: {this_ckpt_path} not found checkpoints')

        self._model = LightGBMModel.from_config(deltanums, edcnums, this_ckpt_path)
        self.delta_nums = self._model.deltanums
        self.edc_nums = self._model.edcnums
        self.memory_window = memory_window

        # online training
        self.training_config = model_config['training']
        self.training_interval = 10000
        self.f = open(os.path.join(current_dir, "model/access_history.txt"), "a") 
        self.existing_online_training = 0
        
        self.deltas = [{} for _ in range(self.delta_nums)]
        self.edcs = [{} for _ in range(self.edc_nums)]
        self.access_time_dict = {}
        self.access_ts = 0
        
        self.belady_value = collections.defaultdict(float)  

    # def _training_task(self):
    #     bst = lgb.train(self.training_config, train_data, valid_sets=[valid_data], callbacks=[
    #         lgb.early_stopping(stopping_rounds=50),
    #     ])

    # def _online_training(self):
    #     if self.existing_online_training == 1:
    #         return
        
    #     executor = ThreadPoolExecutor(max_workers=1)
    #     future = executor.submit(train_lgb)
        

    def access(self, address):
        self.f.write(f"1, {address}")
        self.f.flush()

        if address not in self.access_time_dict:
            self.access_time_dict[address] = collections.deque()
        
        this_access_list = self.access_time_dict[address]
        if len(this_access_list) == self.delta_nums + 1:
            this_access_list.popleft()
            this_access_list.append(self.access_ts)
        else:
            this_access_list.append(self.access_ts)

        for i in range(1, self.delta_nums + 1):
            this_delta = self.deltas[i-1]
            if len(this_access_list) > i:
                delta_i = this_access_list[-i] - this_access_list[-i-1]
                this_delta[address] = delta_i
            else:
                this_delta[address] = np.inf

        delta1 = self.deltas[0][address]
        for i in range(1, self.edc_nums + 1):
            this_edc = self.edcs[i-1]
            if address not in this_edc:
                this_edc[address] = 0
            this_edc[address] = 1 + this_edc[address] * 2 ** (-delta1 / (2 ** (9 + i)))

        self.access_ts += 1

        #if self.access_ts % self.training_interval == 0:
        #    self._online_training()

    def predict(self, address):
        #if address not in self.access_time_dict:
        #    return -1
        pred = self._model((1, address, *[self.deltas[i][address] for i in range(self.delta_nums)], *[self.edcs[i][address] for i in range(self.edc_nums)]))
        
        # if pred == 0: 
        #     self.belady_value[address] += 1.0
        # else:  
        #     self.belady_value[address] = max(0, self.belady_value[address] - 0.1)
            
        # if self.access_ts % 1000 == 0:
        #     to_delete = []
        #     for key in self.belady_value:
        #         if key not in self.access_time_dict or (self.access_ts - self.access_time_dict[key][-1]) > self.memory_window:
        #             to_delete.append(key)
        #     for key in to_delete:
        #         del self.belady_value[key]
        #         if key in self.access_time_dict:
        #             del self.access_time_dict[key]
        #         for i in range(self.delta_nums):
        #             if key in self.deltas[i]:
        #                 del self.deltas[i][key]
        #         for i in range(self.edc_nums):
        #             if key in self.edcs[i]:
        #                 del self.edcs[i][key]
        return pred


class LRBBinaryPredictor(BinaryPredictor):
    def __init__(self, shared_model, threshold, memory_window=1000000):
        super().__init__()
        self._model = shared_model
        self.delta_nums = self._model.deltanums
        self.edc_nums = self._model.edcnums
        self.memory_window = memory_window
        self.threshold = threshold
        
        self.deltas = [{} for _ in range(self.delta_nums)]
        self.edcs = [{} for _ in range(self.edc_nums)]
        self.access_time_dict = {}
        self.access_ts = 0
        
        self.belady_value = collections.defaultdict(float)  
    
    def predict_score(self, ts, pc, address, cache_state):
        if address not in self.access_time_dict:
            self.access_time_dict[address] = collections.deque()
        
        this_access_list = self.access_time_dict[address]
        if len(this_access_list) == self.delta_nums + 1:
            this_access_list.popleft()
            this_access_list.append(self.access_ts)
        else:
            this_access_list.append(self.access_ts)
        
        for i in range(1, self.delta_nums + 1):
            this_delta = self.deltas[i-1]
            if len(this_access_list) > i:
                delta_i = this_access_list[-i] - this_access_list[-i-1]
                this_delta[address] = delta_i
            else:
                this_delta[address] = np.inf

        delta1 = self.deltas[0][address]
        for i in range(1, self.edc_nums + 1):
            this_edc = self.edcs[i-1]
            if address not in this_edc:
                this_edc[address] = 0
            this_edc[address] = 1 + this_edc[address] * 2 ** (-delta1 / (2 ** (9 + i)))

        self.access_ts += 1
        
        ypred = self._model((pc, address, *[self.deltas[i][address] for i in range(self.delta_nums)], *[self.edcs[i][address] for i in range(self.edc_nums)]))
        pred = 0
        if ypred > self.threshold:
            pred = 1

        if pred == 0: 
            self.belady_value[address] += 1.0
        else:  
            self.belady_value[address] = max(0, self.belady_value[address] - 0.1)
            
        if self.access_ts % 1000 == 0: 
            to_delete = []
            for key in self.belady_value:
                if key not in self.access_time_dict or (self.access_ts - self.access_time_dict[key][-1]) > self.memory_window:
                    to_delete.append(key)
            for key in to_delete:
                del self.belady_value[key]
                if key in self.access_time_dict:
                    del self.access_time_dict[key]
                for i in range(self.delta_nums):
                    if key in self.deltas[i]:
                        del self.deltas[i][key]
                for i in range(self.edc_nums):
                    if key in self.edcs[i]:
                        del self.edcs[i][key]
        
        return pred