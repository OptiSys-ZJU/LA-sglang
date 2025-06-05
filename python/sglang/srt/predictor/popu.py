from sglang.srt.predictor.base_predictor import ReuseDistancePredictor
import time

class POPUPredictor(ReuseDistancePredictor):
    def __init__(self):
        super().__init__()
        self.counts = {}
        self.base_time = time.monotonic()

    def split_access(self, original_address, child_addr, parent_addr):
        self.access(original_address)
        # copy features from node with key = original_address
        self.counts[child_addr] = self.counts[original_address]
        self.counts[parent_addr] = self.counts[original_address]

    def spawn_access(self, address, new_address):
        # copy features from parent node
        self.counts[new_address] = self.counts[address]
        #self.access(new_address)

    def access(self, address):
        if address not in self.counts:
            self.counts[address] = 0
        self.counts[address] += 1
    
    def predict(self, address):
        pred = (time.monotonic() - self.base_time) / self.counts[address]
        return pred