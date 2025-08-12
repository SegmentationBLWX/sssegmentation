'''
Function:
    Implementation of TrainTimer
Author:
    Zhenchao Jin
'''
import time


'''TrainTimer'''
class TrainTimer:
    def __init__(self, max_iters: int, ema_beta: float = 0.9):
        self.max_iters = int(max_iters)
        self.ema_beta = float(ema_beta)
        self.t0 = time.perf_counter()
        self.last = None
        self.iter_ema = None
        self.max_iters_in_epoch = None
    '''formatsecs'''
    @staticmethod
    def formatsecs(secs: float) -> str:
        secs = max(0, int(secs))
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    '''initbeforeepochstart'''
    def initbeforeepochstart(self, max_iters_in_epoch: int | None = None, reset_iter_ema: bool = False):
        self.max_iters_in_epoch = int(max_iters_in_epoch) if max_iters_in_epoch is not None else None
        self.last = None
        if reset_iter_ema: self.iter_ema = None
    '''updateperiter'''
    def updateperiter(self, cur_iter: int, cur_iter_in_epoch: int | None = None) -> dict:
        now = time.perf_counter()
        # per-iter time & EMA
        if self.last is None:
            iter_time = 0.0
            if self.iter_ema is None: self.iter_ema = 0.0
        else:
            iter_time = now - self.last
            self.iter_ema = (
                iter_time if (self.iter_ema is None or self.iter_ema == 0.0) else self.ema_beta * self.iter_ema + (1 - self.ema_beta) * iter_time
            )
        self.last = now
        # time elapsed
        elapsed = now - self.t0
        # average time per time calculated by EMA mechanism
        avg_iter = self.iter_ema if self.iter_ema and self.iter_ema > 0 else (iter_time or 0.0)
        remain_iters = max(self.max_iters - int(cur_iter), 0)
        eta = remain_iters * (avg_iter if avg_iter > 0 else 0.0)
        # construct outputs
        out = {
            "time_elapsed": self.formatsecs(elapsed), "eta": self.formatsecs(eta),
            "iter_time_s": round(iter_time, 3), "iter_time_ema_s": round(self.iter_ema or 0.0, 3),
        }
        # optional: epoch ETA if provided
        if self.max_iters_in_epoch is not None and cur_iter_in_epoch is not None:
            remain_in_epoch = max(self.max_iters_in_epoch - int(cur_iter_in_epoch), 0)
            eta_epoch = remain_in_epoch * (avg_iter if avg_iter > 0 else 0.0)
            out["eta_epoch"] = self.formatsecs(eta_epoch)
        # return outputs
        return out