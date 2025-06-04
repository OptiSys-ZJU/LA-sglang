from sglang.srt.predictor.model.models import LightGBMModel
from sglang.srt.predictor.base_predictor import BinaryPredictor
from sglang.srt.predictor.base_predictor import ReuseDistancePredictor

import lightgbm as lgb
import numpy as np
import collections
import asyncio
import logging
import copy
import time
import json
import os

logger = logging.getLogger(__name__)

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
        self._model_save_path = this_ckpt_path
        self.delta_nums = self._model.deltanums
        self.edc_nums = self._model.edcnums
        self.memory_window = memory_window

        # online training
        self.training_config = model_config['training']
        self.training_interval = 500
        self.training_accumu_num = 0
        self.training_window = 10000
        self.training_data = []
        self.existing_online_training = 0
        self.feature_history = {}
        self.features = []
        self.trained = 0
        
        self.deltas = [{} for _ in range(self.delta_nums)]
        self.edcs = [{} for _ in range(self.edc_nums)]
        self.access_time_dict = {}
        self.access_ts = 0
        
        self.belady_value = collections.defaultdict(float)  

    def _training_task(self):
        train_data = [t[:-1] for t in self.features]
        labels = [t[-1] for t in self.features]
        X = np.array(train_data)
        y = np.array(labels)
        train_dataset = lgb.Dataset(X, label=y)
        return lgb.train(self.training_config, train_dataset)

    async def _online_training(self):
        if self.existing_online_training == 1:
            logger.info(f"current training, conflict !!!")
            return
        
        self.existing_online_training = 1
        start = time.time()
        model = await asyncio.to_thread(self._training_task)
        model.save_model(self._model_save_path)
        end = time.time()
        self.features = collections.deque()
        logger.info(f"current_access_ts = {self.access_ts}, training time cost = {end - start}, training interval = {self.training_interval}")
        
        self._model = LightGBMModel.from_config(self.delta_nums, self.edc_nums, self._model_save_path)
        self.existing_online_training = 0
        self.trained = 1

    def access(self, address):
        self.access_ts += 1

        if address not in self.access_time_dict:
            self.access_time_dict[address] = collections.deque()
        elif address in self.feature_history:
            self.features.append((*self.feature_history[address], self.access_ts))
            if len(self.features) > self.training_window:
                self.features.popleft()
            self.training_accumu_num += 1
            #logger.info(f"address: {address}, features num: {len(self.features)}")
            if self.training_accumu_num >= self.training_interval:
                self.training_accumu_num = 0
                #logger.info(f"current features num: {len(self.features)}, time: {time.time()}")
                asyncio.run(self._online_training())

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

        # collect training data
        self.feature_history[address] = [1, address, *[self.deltas[i][address] for i in range(self.delta_nums)], *[self.edcs[i][address] for i in range(self.edc_nums)]]

    def predict(self, address):
        if address not in self.access_time_dict:
            return 2**62
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