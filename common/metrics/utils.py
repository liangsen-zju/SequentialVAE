class AverageMeter:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class AverageMeterDict:
    def __init__(self):
        self.count = 0

        self.sum_dict = {}
        self.avg_dict = {}
        self.val_dict = {}

    def reset(self):
        self.count = 0
        self.sum_dict = {}
        self.avg_dict = {}
        self.val_dict = {}

    def update(self, adict, n=1):
        
        self.count += n

        # first init 
        if len(self.val_dict) == 0 :
            self.val_dict = adict.copy()
            self.sum_dict = adict.copy()
            self.avg_dict = adict.copy()
        
        else:
            for key, item in adict.items():
                self.val_dict[key] = item
                self.sum_dict[key] += item * n
                self.avg_dict[key] = (self.sum_dict[key] / self.count)


    def print_msg(self, debug="avg", split=""):
        msg = ""
        
        if debug == "sum":
            tmp_dict = self.avg_dict
        elif debug == "val":
            tmp_dict = self.val_dict
        else:
            tmp_dict = self.avg_dict

        for (key, item) in tmp_dict.items():
            msg += f"{key}={item:0.6f} {split}"

        return msg
        


    