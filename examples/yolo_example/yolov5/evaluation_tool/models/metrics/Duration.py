import time
import torch


class Duration:
    last_time = 0
    last_duration = 0
    startevent = torch.cuda.Event(enable_timing=True)
    endevent = torch.cuda.Event(enable_timing=True)

    def start(self):
        # self.last_time = time.time(
        self.startevent.record()

    def end(self):
        # self.last_duration = time.time() - self.last_time
        # self.last_time = time.time()
        #
        # return self.last_duration
        self.endevent.record()
        torch.cuda.synchronize()
        self.last_duration = self.startevent.elapsed_time(self.endevent)
        return self.last_duration

    def get_last_duration(self):
        return self.last_duration

    def set_last_duration(self, dur):
        self.last_duration = dur

    def get_last_time(self):
        return self.last_time
