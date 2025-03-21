import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
from PIL import Image
import pdb
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from sklearn.datasets import load_iris
import seaborn as sns
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.6f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, args, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    if args.log:
                        print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB), flush=True)
                else:
                    if args.log:
                        print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time)), flush=True)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if args.log:
            print('{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)), flush=True)
        


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
def add_multiline_text(image, text, position, font, font_scale, font_color, thickness, max_width):
    """在图像上添加多行文本"""
    # 分割文本为多行
    lines = []
    words = text.split(' ')
    current_line = ''
    
    for word in words:
        # 检查当前行加上下一个单词是否超出最大宽度
        test_line = current_line + word + ' '
        text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
        
        if text_size[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word + ' '
    lines.append(current_line)  # 添加最后一行

    # 在图像上绘制每一行文本
    y_offset = position[1]
    for line in lines:
        image = cv2.putText(image, line.strip(), (position[0], y_offset), font, font_scale, font_color, thickness)
        y_offset += int(text_size[1] * 1.5)  # 行间距
    return image
def visualize_res(inputs, preds):
    data_len = len(inputs['img'])
    for index in range(data_len):
        img = inputs['img'][index]
        text = inputs['text'][index]
        label = inputs['label'][index]
        gt_box = inputs['gt_box'][index].detach().cpu().numpy()
        gt_text = inputs['gt_text'][index].detach().cpu().numpy()

        binary = preds['binary'][index]
        multi = preds['multi'][index].detach().cpu()
        pred_box = preds['box'][index].detach().cpu().numpy()
        pred_text = preds['text'][index].detach().cpu()

        tag = torch.max(binary, dim=-1)[1]
        if tag==0:
            pred_tag='Real'
        else:
            pred_tag='Fake-'
            multi[multi>=0]=1
            multi[multi<0]=0
            if torch.equal(multi, torch.tensor([1.0,0.0,0.0,0.0])):
                pred_tag+='FS'
            elif torch.equal(multi, torch.tensor([0.0,1.0,0.0,0.0])):
                pred_tag+='FA'
            elif torch.equal(multi, torch.tensor([0.0,0.0,1.0,0.0])):
                pred_tag+='TS'
            elif torch.equal(multi, torch.tensor([0.0,0.0,0.0,1.0])):
                pred_tag+='TA'
            elif torch.equal(multi, torch.tensor([1.0,0.0,1.0,0.0])):
                pred_tag+='FS&TS'
            elif torch.equal(multi, torch.tensor([1.0,0.0,0.0,1.0])):
                pred_tag+='FS&TA'
            elif torch.equal(multi, torch.tensor([0.0,1.0,1.0,0.0])):
                pred_tag+='FA&TS'
            elif torch.equal(multi, torch.tensor([0.0,1.0,0.0,1.0])):
                pred_tag+='FA&TA'
            else:
                pred_tag='Real'
            
        if label=='orig':
            gt_tag='Real'
        else:
            gt_tag='Fake-'
            gt_tag += label.replace("face_attribute", "FA").replace("face_swap", "FS").replace("text_attribute", "TA").replace("text_swap", "TS")
        if gt_tag=='Real': #真实的时候不可视化
            continue

        image = Image.open(img)
        W, H = image.size
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        gt_box = [gt_box[0]*W, gt_box[1]*H, gt_box[2]*W, gt_box[3]*H]
        gt_x1 = int(gt_box[0] - gt_box[2] / 2)
        gt_y1 = int(gt_box[1] - gt_box[3] / 2)
        gt_x2 = int(gt_box[0] + gt_box[2] / 2)
        gt_y2 = int(gt_box[1] + gt_box[3] / 2)
        image = cv2.rectangle(image, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), 2)  # 绿色框，线宽为2
        pred_box = [pred_box[0]*W, pred_box[1]*H, pred_box[2]*W, pred_box[3]*H]
        pred_x1 = int(pred_box[0] - pred_box[2] / 2)
        pred_y1 = int(pred_box[1] - pred_box[3] / 2)
        pred_x2 = int(pred_box[0] + pred_box[2] / 2)
        pred_y2 = int(pred_box[1] + pred_box[3] / 2)
        image = cv2.rectangle(image, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 0, 255), 2)  # 红色框，线宽为2

        image = cv2.resize(image, (1024, 1024))
        # 创建一个空白区域（可以根据需要调整高度）
        blank_area_height = 400
        blank_area = np.ones((blank_area_height, 1024, 3), dtype=np.uint8) * 255  # 白色空白区域
        blank_top = np.ones((80, 1024, 3), dtype=np.uint8) * 255  # 白色空白区域
        # 将图像和空白区域合并
        image = np.vstack((blank_top, image, blank_area))

        top_text = "GT: "+gt_tag+', '+"Pred: "+pred_tag
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 1.5
        font_color = (0, 0, 0)  # 黑色
        thickness = 1
        text_size = cv2.getTextSize(top_text, font, font_scale, thickness)[0]
        ttext_x = (image.shape[1] - text_size[0]) // 2  # 文字居中
        ttext_y = 15
        image = cv2.putText(image, top_text, (ttext_x, 50), font, font_scale, font_color, thickness)

        # 添加多行文字
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 1.5
        font_color = (0, 0, 0)  # 黑色
        thickness = 1
        max_width = 1024 - 10  # 最大宽度，留出边距
        x_start = 20
        y_position = 1024 + 150
        image_width = 1024
        # 在空白区域的中心添加多行文字
        # image = add_multiline_text(image, text, (10, 1024 + 150), font, font_scale, font_color, thickness, max_width)
        words = text.split(' ')

        gt_indice = np.nonzero(gt_text == 1)[0].tolist()

        pred_text_label = torch.max(pred_text,dim=-1)[1]

        for i, word in enumerate(words):
            # 计算文本大小
            (w, h), baseline = cv2.getTextSize(word, font, font_scale, thickness)

            if x_start + w > image_width - 10:  # 如果超出宽度，换行
                x_start = 20
                y_position += 50

            # 如果当前词在高亮索引中，绘制绿色背景
            if i in gt_indice:
                cv2.rectangle(image, (x_start, y_position - h - 5), (x_start + w, y_position + baseline), (180, 238, 180), cv2.FILLED)

            # 绘制文本
            if pred_text_label[i] == 1:
                cv2.putText(image, word, (x_start, y_position), font, font_scale, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, word, (x_start, y_position), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            # 更新 x 坐标，添加一个空格的宽度
            x_start += w + 10  # 10 pixels space between words

            # 换行处理
            if x_start > image_width - 10:  # 如果超出宽度，换行
                x_start = 20
                y_position += 50
        cv2.imwrite('pic_ours/'+str(index)+'.png', image)