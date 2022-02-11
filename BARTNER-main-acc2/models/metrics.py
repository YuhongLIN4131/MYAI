
from fastNLP import MetricBase
from fastNLP.core.metrics import _compute_f_pre_rec
import numpy as np
import json


class Seq2SeqSpanMetric(MetricBase):
    def __init__(self):
        super(Seq2SeqSpanMetric, self).__init__()
        self.task = ["Onto5","conll03","ace2004","ace2005","genia","cadce","share2013","share2014"]


    def evaluate(self, target_span, pred,sample_id=None):
        all_fn = 0
        all_fp = 0
        all_tp = 0
        for i, (ts, ps) in enumerate(zip(target_span, pred)):
            if sample_id is not None:
                tp, fn, fp = _compute_tp_fn_fp(ps, ts,sample_id[i])
            else:
                tp, fn, fp = _compute_tp_fn_fp(ps, ts)
            all_fn += fn
            all_tp += tp
            all_fp += fp
        return all_fn,all_tp,all_fp


def _compute_tp_fn_fp(ps, ts,sample_id=None):
    ps = ps.copy()
    tp = 0
    fp = 0
    fn = 0
    if isinstance(ts, (set, list, np.ndarray)):
        ts = {tuple(key):1 for key in list(ts)}
    if isinstance(ps, (set, list, np.ndarray)):
        ps = {tuple(key):1 for key in list(ps)}
    for key in ts.keys():
        t_num = ts[key]
        if key not in ps:
            p_num = 0
        else:
            p_num = ps[key]
        tp += min(p_num, t_num)
        fp += max(p_num - t_num, 0)
        fn += max(t_num - p_num, 0)
        if key in ps:
            ps.pop(key)
    fp += sum(ps.values())
    '''计算一下错误的'''
    if sample_id is not None:
        error_id_span=json.load(open("/home/wyq/BARTNER-main/data/ace2005/sample_error_entity", encoding="utf-8"))#记录所有错误的预测实体
        for key,val in ps.items():
            if key not in error_id_span[sample_id]:
                error_id_span[sample_id].append(key)
        json.dump(error_id_span, open("/home/wyq/BARTNER-main/data/ace2005/sample_error_entity", "w", encoding="utf-8"))
    return tp, fn, fp
