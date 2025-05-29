from sglang.srt.predictor.base_predictor import ReuseDistancePredictor

class POPUPredictor(ReuseDistancePredictor):
    def __init__(self):
        super().__init__()
        self.counts = {}
        self.timestamp = 1

    def access(self, address):
        self.timestamp += 1
    
    def predict(self, address):
        if address not in self.counts:
            self.counts[address] = 0
        self.counts[address] += 1

        pred = self.timestamp + self.timestamp / self.counts[address]
        return pred