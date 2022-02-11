r"""undocumented"""

import torch
import copy
from torch import nn
from fastNLP.models.seq2seq_model import Seq2SeqModel
from fastNLP.modules.decoder.seq2seq_decoder import Seq2SeqDecoder, State
import torch.nn.functional as F
from itertools import chain
from fastNLP.core.utils import _get_model_device
from functools import partial


class SequenceGeneratorModel(nn.Module):
    """
    用于封装Seq2SeqModel使其可以做生成任务

    """

    def __init__(self, seq2seq_model: Seq2SeqModel, bos_token_id, eos_token_id=None, max_length=30, max_len_a=0.0,
                 num_beams=1, do_sample=True,
                 repetition_penalty=1, length_penalty=1.0, pad_token_id=0,
                 restricter=None,target_shift=3,real_target_shift=3):
        """

        :param Seq2SeqModel seq2seq_model: 序列到序列模型. 会使用seq2seq_model的decoder进行生成
        :param int,None bos_token_id: 句子开头的token id
        :param int,None eos_token_id: 句子结束的token id
        :param int max_length: 生成句子的最大长度, 每句话的decode长度为max_length + max_len_a*src_len
        :param float max_len_a: 每句话的decode长度为max_length + max_len_a*src_len。 如果不为0，需要保证State中包含encoder_mask
        :param int num_beams: beam search的大小
        :param bool do_sample: 是否通过采样的方式生成
        :param float temperature: 只有在do_sample为True才有意义
        :param int top_k: 只从top_k中采样
        :param float top_p: 只从top_p的token中采样，nucles sample
        :param float repetition_penalty: 多大程度上惩罚重复的token
        :param float length_penalty: 对长度的惩罚，小于1鼓励长句，大于1鼓励短剧
        :param int pad_token_id: 当某句话生成结束之后，之后生成的内容用pad_token_id补充
        """
        super().__init__()
        self.seq2seq_model = seq2seq_model
        self.restricter = restricter
        self.target_shift = target_shift
        self.real_target_shift = real_target_shift
        self.num_beams=num_beams#放在字典，加快查询
        self.eos_token_id=eos_token_id#放在字典，加快查询


        self.generator = SequenceGenerator(seq2seq_model.decoder, max_length=max_length, max_len_a=max_len_a,
                                           num_beams=num_beams,
                                           do_sample=do_sample,
                                           bos_token_id=bos_token_id,
                                           eos_token_id=eos_token_id,
                                           repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                                           pad_token_id=pad_token_id,
                                           restricter=restricter
                                           )

    def forward(self, src_tokens, tgt_tokens, src_seq_len=None, tgt_seq_len=None,mask_query=None,OOV_con=None,position_ids=None,attention=None,error_flag=None,decoder_embedding_index=None):
        """
        透传调用seq2seq_model的forward

        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor tgt_tokens: bsz x max_len'
        :param torch.LongTensor src_seq_len: bsz
        :param torch.LongTensor tgt_seq_len: bsz
        :return:
        """
        return self.seq2seq_model(src_tokens, tgt_tokens, src_seq_len, tgt_seq_len,mask_query,OOV_con,position_ids,attention,error_flag,decoder_embedding_index)

    def predict(self, src_tokens, src_seq_len=None,mask_query=None,OOV_con=None,position_ids=None,attention=None,tgt_tokens=None,decoder_embedding_index=None,Generate_negative_samples=False,):
        """
        给定source的内容，输出generate的内容
        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor src_seq_len: bsz
        :return:
        """
        state = self.seq2seq_model.prepare_state(src_tokens, src_seq_len,mask_query=mask_query,position_ids=position_ids,OOV_con=OOV_con,attention=attention,decoder_embedding_index=decoder_embedding_index)

        batch_size, seq_len, target_length = tgt_tokens.size()
        all_word_entity_length = ((tgt_tokens != 1).sum(dim=-1))  # ，batch*ser_len
        indices = torch.arange(batch_size, dtype=torch.long).to(src_tokens.device)
        indices = indices.repeat_interleave(seq_len)
        all_word_entity_length = all_word_entity_length.view(-1)
        # 1\抽取长度为2的一起训练,至少有起点和一个起点符号的才不是填充
        noo_entity_pos = torch.nonzero(all_word_entity_length >= 2).squeeze(-1)
        pos_all = indices[noo_entity_pos]
        tgt_tokens = tgt_tokens.view(batch_size * seq_len, target_length)
        tokens = (tgt_tokens[noo_entity_pos])[:, 0:2]  # 只需要前两个
        '''然后再抽取非空的实体'''
        state.reorder_state(pos_all)
        # batch_size = len(src_tokens)
        # tokens = torch.full([batch_size, 1], fill_value=0, dtype=torch.long).to(mask_query.device)
        '''然后再抽取非空的实体'''
        result_old = self.generator.generate(state,tokens,Generate_negative_samples=Generate_negative_samples)
        if Generate_negative_samples and self.num_beams>1:#只是生成负样本，不是预测
            result = []
            for i in range(batch_size):  # 把每个batch收录的结果全部放入
                i_pos = torch.nonzero(pos_all == i).squeeze(-1)
                pairs = []
                cur_pair = []
                now_result_old = [result_old[itemss] for itemss in range(len(result_old)) if itemss in i_pos]
                for temp_ps in now_result_old:
                    for ps in temp_ps:
                        if len(ps) == 0:
                            continue
                        ps=ps[1:]#不要头token
                        if ps[1] != self.eos_token_id:  # 有实体生成
                            for j in ps:
                                if j < self.target_shift:  # target_shift==3
                                    if len(cur_pair) > 0 and j < self.real_target_shift and j != self.eos_token_id:
                                        if all([cur_pair[i] < cur_pair[i + 1] for i in range(len(cur_pair) - 1)]):
                                            pairs.append(cur_pair + [j])
                                    cur_pair = []
                                else:
                                    cur_pair.append(j)
                result.append(copy.deepcopy(pairs))
            return result
        else:
            result = []
            for i in range(batch_size):  # 把每个batch收录的结果全部放入
                i_pos = torch.nonzero(pos_all == i).squeeze(-1)
                pairs = []
                cur_pair = []
                aim_pred=[result_old[ki] for ki in i_pos]
                aim_pred = list(chain(*aim_pred))
                for ps in aim_pred:
                    ps = ps[1:].tolist()  # 不要头token
                    for j in ps:
                        if j < self.target_shift:  # target_shift==3
                            if len(cur_pair) > 0 and j < self.real_target_shift and j != self.eos_token_id:
                                if all([cur_pair[i] < cur_pair[i + 1] for i in range(len(cur_pair) - 1)]):
                                        pairs.append(cur_pair + [j])
                            cur_pair = []
                        else:
                            cur_pair.append(j)
                result.append(copy.deepcopy(pairs))
            return result
r"""

"""

__all__ = [
    'SequenceGenerator'
]


class SequenceGenerator:
    """
    给定一个Seq2SeqDecoder，decode出句子

    """
    def __init__(self, decoder: Seq2SeqDecoder, max_length=20, max_len_a=0.0, num_beams=1,
                 do_sample=False, bos_token_id=None, eos_token_id=None,
                 repetition_penalty=1, length_penalty=1.0, pad_token_id=0, restricter=None):
        """

        :param Seq2SeqDecoder decoder: Decoder对象
        :param int max_length: 生成句子的最大长度, 每句话的decode长度为max_length + max_len_a*src_len
        :param float max_len_a: 每句话的decode长度为max_length + max_len_a*src_len。 如果不为0，需要保证State中包含encoder_mask
        :param int num_beams: beam search的大小
        :param bool do_sample: 是否通过采样的方式生成
        :param float temperature: 只有在do_sample为True才有意义
        :param int top_k: 只从top_k中采样
        :param float top_p: 只从top_p的token中采样，nucles sample
        :param int,None bos_token_id: 句子开头的token id
        :param int,None eos_token_id: 句子结束的token id
        :param float repetition_penalty: 多大程度上惩罚重复的token
        :param float length_penalty: 对长度的惩罚，小于1鼓励长句，大于1鼓励短剧
        :param int pad_token_id: 当某句话生成结束之后，之后生成的内容用pad_token_id补充
        """
        self.generate_func = partial(greedy_generate, decoder=decoder, max_length=max_length, max_len_a=max_len_a,
                                     num_beams=num_beams,
                                     bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                                     repetition_penalty=repetition_penalty,
                                     length_penalty=length_penalty, pad_token_id=pad_token_id,
                                     restricter=restricter)
        self.do_sample = do_sample
        self.max_length = max_length
        self.num_beams = num_beams
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.decoder = decoder
        self.pad_token_id = pad_token_id
        self.restricter = restricter
        self.max_len_a = max_len_a

    def set_new_generator(self, max_length=-1, max_len_a=-1, num_beams=-1,
                          repetition_penalty=-1, length_penalty=-1, restricter=-1):
        if max_length == -1:
            max_length = self.max_length
        if max_len_a == -1:
            max_len_a = self.max_len_a
        if num_beams == -1:
            num_beams = self.num_beams
        if repetition_penalty == -1:
            repetition_penalty = self.repetition_penalty
        if length_penalty == -1:
            length_penalty = self.length_penalty
        if restricter == -1:
            restricter = self.restricter
        self.generate_func = partial(greedy_generate, decoder=self.decoder, max_length=max_length, max_len_a=max_len_a,
                                     num_beams=num_beams,
                                     bos_token_id=self.bos_token_id, eos_token_id=self.eos_token_id,
                                     repetition_penalty=repetition_penalty,
                                     length_penalty=length_penalty, pad_token_id=self.pad_token_id,
                                     restricter=restricter)

    @torch.no_grad()
    def generate(self, state, tokens=None,Generate_negative_samples=False):
        """

        :param State state: encoder结果的State, 是与Decoder配套是用的
        :param torch.LongTensor,None tokens: batch_size x length, 开始的token
        :return: bsz x max_length' 生成的token序列。如果eos_token_id不为None, 每个sequence的结尾一定是eos_token_id
        """
        return self.generate_func(tokens=tokens, state=state,Generate_negative_samples=Generate_negative_samples)


@torch.no_grad()
def greedy_generate(decoder, tokens=None, state=None, max_length=20, max_len_a=0.0, num_beams=1,
                    bos_token_id=None, eos_token_id=None, pad_token_id=0,
                    repetition_penalty=1, length_penalty=1.0, restricter=None,Generate_negative_samples=False):
    """
    贪婪地搜索句子

    :param Decoder decoder: Decoder对象
    :param torch.LongTensor tokens: batch_size x len, decode的输入值，如果为None，则自动从bos_token_id开始生成
    :param State state: 应该包含encoder的一些输出。
    :param int max_length: 生成句子的最大长度, 每句话的decode长度为max_length + max_len_a*src_len
    :param float max_len_a: 每句话的decode长度为max_length + max_len_a*src_len。 如果不为0，需要保证State中包含encoder_mask
    :param int num_beams: 使用多大的beam进行解码。
    :param int bos_token_id: 如果tokens传入为None，则使用bos_token_id开始往后解码。
    :param int eos_token_id: 结束的token，如果为None，则一定会解码到max_length这么长。
    :param int pad_token_id: pad的token id
    :param float repetition_penalty: 对重复出现的token多大的惩罚。
    :param float length_penalty: 对每个token（除了eos）按照长度进行一定的惩罚。
    :return:
    """
    num_beams=num_beams if Generate_negative_samples else 1
    if num_beams == 1:
        token_ids = _no_beam_search_generate(decoder, tokens=tokens, state=state, max_length=max_length, max_len_a=max_len_a,
                                             bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                                             repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                                             pad_token_id=pad_token_id, restricter=restricter)
    else:
        token_ids = _beam_search_generate(decoder, tokens=tokens, state=state, max_length=max_length, max_len_a=max_len_a,
                                          num_beams=num_beams,
                                          bos_token_id=bos_token_id, eos_token_id=eos_token_id, do_sample=False,
                                          repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                                          pad_token_id=pad_token_id, restricter=restricter)

    return token_ids


def _no_beam_search_generate(decoder: Seq2SeqDecoder, state, tokens=None, max_length=20, max_len_a=0.0, bos_token_id=None,
                             eos_token_id=None,
                             repetition_penalty=1.0, length_penalty=1.0, pad_token_id=0,
                             restricter=None):
    device = _get_model_device(decoder)
    if tokens is None:
        if bos_token_id is None:
            raise RuntimeError("You have to specify either `tokens` or `bos_token_id`.")
        batch_size = state.num_samples
        if batch_size is None:
            raise RuntimeError("Cannot infer the number of samples from `state`.")
        tokens = torch.full([batch_size, 1], fill_value=bos_token_id, dtype=torch.long).to(device)
    batch_size = tokens.size(0)
    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id
    '''给出长度限制'''
    if max_len_a!=0:
        # (bsz x num_beams, )
        if state.encoder_mask is not None:
            max_lengths = (state.encoder_mask.sum(dim=1).float()*max_len_a).long() + max_length
        else:
            max_lengths = tokens.new_full((tokens.size(0), ), fill_value=max_length, dtype=torch.long)
        real_max_length = min(50,max_lengths.max().item())
    else:
        real_max_length = max_length
        if state.encoder_mask is not None:
            max_lengths = state.encoder_mask.new_ones(state.encoder_mask.size(0)).long()*max_length
        else:
            max_lengths = tokens.new_full((tokens.size(0),), fill_value=max_length, dtype=torch.long)
    token_ids = tokens
    cur_len = token_ids.size(1)
    new_batch_size = tokens.size(0)
    dones = token_ids.new_zeros(new_batch_size).eq(1)
    all_pos = torch.LongTensor(list(range(len(token_ids)))).to(token_ids.device)
    org_pos = copy.deepcopy(all_pos)
    '''如果是所有词生成,那么这里还需要再一步去删除那些直接被预测为空的词'''
    result_all=[]
    for i in all_pos:
        result_all.append([])
    seq_scores=torch.zeros_like(max_lengths)#记录序列的分数
    while cur_len < real_max_length:
        scores = decoder.decode(tokens=token_ids, state=state)  # new_batch_size x vocab_size
        '''凡是大于0的都可以'''
        new_batch_size,vocab_size=scores.size()
        '''记录每个开头词对应的实体的最大个数 每个实体最多对应5个序列'''
        '''错误样本采样，每次直接前四个都继续采样'''
        # judge_scores = torch.clamp(torch.min(torch.topk(scores,4,1,largest=True)[0],dim=-1)[0],-10,-3)
        judge_scores=torch.clamp(torch.min(torch.topk(scores,5,1,largest=True)[0],dim=-1)[0],0,100)
        scores_flag = (scores > judge_scores.unsqueeze(1))
        '''常规采样'''
        # scores_flag = scores > 0
        ''''''
        scores_flag = scores_flag.view(-1)
        '''对于嵌套实体和flat实体，每次只能得到标签或者下一个词作为下一个点 《9或者 相减=1
        对于不连续的实体，每次也只能选择后15，或者标签作为下一个点'''
        next_pos = torch.LongTensor(list(range(vocab_size))).unsqueeze(0).expand(new_batch_size,-1).to(token_ids.device)
        pre_flag = torch.logical_or(((next_pos-token_ids[:,-1:])==1).view(-1),(next_pos<9).view(-1))
        scores_flag = scores_flag & pre_flag
        '''不连续的话就是<20'''
        scores = scores.view(-1)
        new_pos = torch.nonzero(scores_flag).squeeze(-1)#大于0都可
        scores=scores[new_pos]
        '''每个唯一下标的序列只有前五能够继续训练，防止爆炸'''
        seq_scores = seq_scores.repeat_interleave(vocab_size, dim=0)
        seq_scores = scores + seq_scores[new_pos]#
        all_pos = all_pos.repeat_interleave(vocab_size, dim=0)
        all_pos = all_pos[new_pos]#每个可能下标的位置
        '''如何让每种下标最多为5'''
        next_tokens_pos=[]
        for ijx in org_pos:
            i_pos = torch.nonzero(all_pos == ijx).squeeze(-1)#错误的
            if len(i_pos)>16:#只能最多5
                i_pos= i_pos[(torch.sort(seq_scores[i_pos], descending=True)[1])[0:16]]
            if len(i_pos)>0:
                next_tokens_pos = next_tokens_pos + new_pos[i_pos].tolist()
        next_tokens_pos = torch.LongTensor(next_tokens_pos).to(token_ids.device)
        ''''''
        # next_tokens_pos = new_pos #大于0都可


        next_tokens = next_tokens_pos % vocab_size
        '''得到max_lengths'''
        max_lengths = max_lengths.repeat_interleave(vocab_size, dim=0)
        max_lengths = max_lengths[next_tokens_pos]
        ''''''
        '''得到done'''
        dones = dones.repeat_interleave(vocab_size, dim=0)
        dones = dones[next_tokens_pos]

        ''''''
        if _eos_token_id!=-1:
            next_tokens = next_tokens.masked_fill(max_lengths.eq(cur_len+1), _eos_token_id)
        next_tokens = next_tokens.masked_fill(dones, pad_token_id)  # 对已经搜索完成的sample做padding
        '''去掉已经完成的了，然后得到下个位置'''
        end_mask = next_tokens.eq(_eos_token_id)
        token_ids = token_ids.repeat_interleave(vocab_size, dim=0)
        token_ids = torch.cat((token_ids[next_tokens_pos], next_tokens.unsqueeze(1)), dim=-1)
        dones = dones.__or__(end_mask)
        # 选择那些已经完成了的
        end_mask_pos = torch.nonzero(end_mask).squeeze(-1)#已经结束的就不放进去添乱了
        for k in end_mask_pos:#已经结束的
            result_all[all_pos[k]].append(token_ids[k])
        continue_pos = torch.nonzero(~end_mask).squeeze(-1)  # 已经结束的就不放进去添乱了
        indices = torch.arange(new_batch_size, dtype=torch.long).to(device)
        indices = indices.repeat_interleave(vocab_size)
        indices = indices[next_tokens_pos]
        '''给出需要继续执行的信息，然后就继续执行'''
        if len(continue_pos)>0:#还有没有正常结束的序列
            dones = dones[continue_pos]
            all_pos = all_pos[continue_pos]
            max_lengths = max_lengths[continue_pos]
            token_ids=token_ids[continue_pos]
            ''''''
            state.reorder_state(indices[continue_pos])
        else:#没有了
            token_ids=[]
            all_pos=[]
            break
        # 如果已经达到对应的sequence长度了，就直接填为eos了
        cur_len += 1
        if dones.min() == 1:
            break
    '''组装结果'''
    # if eos_token_id is not None:
    #     tokens.scatter(index=max_lengths[:, None], dim=1, value=eos_token_id)  # 将最大长度位置设置为eos
    # if cur_len == max_length:
    #     token_ids[:, -1].masked_fill_(~dones, eos_token_id)  # 若到最长长度仍未到EOS，则强制将最后一个词替换成eos
    for i,val in enumerate(all_pos):
        result_all[val].append(token_ids[i])

    return result_all


def _beam_search_generate(decoder: Seq2SeqDecoder, tokens=None, state=None, max_length=20, max_len_a=0.0, num_beams=4,
                          bos_token_id=None, eos_token_id=None, do_sample=True,
                          repetition_penalty=1.0, length_penalty=None, pad_token_id=0,
                          restricter=None) -> torch.LongTensor:
    assert do_sample is False
    # beam search是用来生成负样本的
    device = _get_model_device(decoder)
    if tokens is None:
        if bos_token_id is None:
            raise RuntimeError("You have to specify either `tokens` or `bos_token_id`.")
        batch_size = state.batch_size
        if batch_size is None:
            raise RuntimeError("Cannot infer the number of samples from `state`.")
        tokens = torch.full([batch_size, 1], fill_value=bos_token_id, dtype=torch.long).to(device)
    batch_size = tokens.size(0)

    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id
    scores = decoder.decode(tokens=tokens, state=state,test=True)  # 这里要传入的是整个句子的长度
    vocab_size = scores.size(1)
    assert vocab_size >= num_beams, "num_beams should be smaller than the number of vocabulary size."
    scores = F.log_softmax(scores, dim=-1)  # (batch_size, vocab_size)
    if restricter is not None:
        _next_scores, _next_tokens = restricter(state, tokens, scores, num_beams+1)
    else:
        # 是bsz x (num_beams+1)大小的东西
        _next_scores, _next_tokens = torch.topk(scores, num_beams+1, dim=1, largest=True, sorted=True)#下一个token的前n个值，排序

    # 根据index来做顺序的调转
    indices = torch.arange(batch_size, dtype=torch.long).to(device)
    indices = indices.repeat_interleave(num_beams)
    state.reorder_state(indices)

    tokens = tokens.index_select(dim=0, index=indices)  # batch_size * num_beams x length

    if max_len_a!=0:
        if state.encoder_mask is not None:
            max_lengths = (state.encoder_mask.sum(dim=1).float()*max_len_a).long() + max_length
        else:
            max_lengths = tokens.new_full((batch_size*num_beams, ), fill_value=max_length, dtype=torch.long)
        real_max_length = max_lengths.max().item()
    else:
        real_max_length = max_length
        if state.encoder_mask is not None:
            max_lengths = state.encoder_mask.new_ones(state.encoder_mask.size(0)).long()*max_length
        else:
            max_lengths = tokens.new_full((batch_size*num_beams,), fill_value=max_length, dtype=torch.long)
    '''生成路径'''
    hypos = [
        BeamHypotheses(num_beams, real_max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
    ]
    #
    not_eos_mask = _next_tokens.ne(_eos_token_id)  # 非结束点
    keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)  # 为1的地方需要保留
    keep_mask = not_eos_mask.__and__(keep_mask)  # 为1的地方是需要进行下一步search的

    next_tokens = _next_tokens.masked_select(keep_mask).view(batch_size, num_beams)  # 这是真的接下来要继续的
    next_scores = _next_scores.masked_select(keep_mask).view(batch_size, num_beams)

    rows, cols = not_eos_mask.eq(0)[:, :num_beams].nonzero(as_tuple=True)

    if len(rows)>0:  # 说明有的开头就结束了
        for row, col in zip(rows.tolist(), cols.tolist()):
            _token = torch.cat([tokens[row*num_beams], _next_tokens[row, col:col+1]], dim=0)
            hypos[row].add(_token.clone(), _next_scores[row, col].item())

    # 记录生成好的token (batch_size', cur_len)
    token_ids = torch.cat([tokens, next_tokens.view(-1, 1)], dim=-1)
    dones = [False] * batch_size

    beam_scores = next_scores.view(-1)  # batch_size * num_beams

    #  用来记录已经生成好的token的长度
    cur_len = token_ids.size(1)

    # 0, num_beams, 2*num_beams, ...
    batch_inds_with_numbeams_interval = (torch.arange(batch_size) * num_beams).view(-1, 1).to(token_ids)

    while cur_len < real_max_length:
        scores = decoder.decode(token_ids, state,test=True)  # (bsz x num_beams, vocab_size)

        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            scores.scatter_(dim=1, index=token_ids, src=token_scores)

        if _eos_token_id!=-1:
            max_len_eos_mask = max_lengths.eq(cur_len+1)
            eos_scores = scores[:, _eos_token_id]
            # 如果已经达到最大长度，就把eos的分数加大
            scores[:, _eos_token_id] = torch.where(max_len_eos_mask, eos_scores+1e32, eos_scores)

        scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
        _scores = scores + beam_scores[:, None]  # (batch_size * num_beams, vocab_size)
        _scores = _scores.view(batch_size, -1)  # (batch_size, num_beams*vocab_size)
        # TODO 把限制加到这个位置
        if restricter is not None:
            next_scores, ids = restricter(state, token_ids, _scores, 2 * num_beams)
        else:
            next_scores, ids = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)  # (bsz, 2*num_beams)
        from_which_beam = ids // vocab_size  # (batch_size, 2*num_beams)
        next_tokens = ids % vocab_size  # (batch_size, 2*num_beams)

        #  接下来需要组装下一个batch的结果。
        #  需要选定哪些留下来
        # next_scores, sorted_inds = next_scores.sort(dim=-1, descending=True)
        # next_tokens = next_tokens.gather(dim=1, index=sorted_inds)
        # from_which_beam = from_which_beam.gather(dim=1, index=sorted_inds)

        not_eos_mask = next_tokens.ne(_eos_token_id)  # 为1的地方不是eos
        keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)  # 为1的地方需要保留
        keep_mask = not_eos_mask.__and__(keep_mask)  # 为1的地方是需要进行下一步search的

        _next_tokens = next_tokens.masked_select(keep_mask).view(-1, 1)
        _from_which_beam = from_which_beam.masked_select(keep_mask).view(batch_size, num_beams)  # 上面的token是来自哪个beam
        _next_scores = next_scores.masked_select(keep_mask).view(batch_size, num_beams)
        beam_scores = _next_scores.view(-1)

        flag = True
        if cur_len+1 == real_max_length:
            eos_batch_idx = torch.arange(batch_size).to(next_tokens).repeat_interleave(repeats=num_beams, dim=0)
            eos_beam_ind = torch.arange(num_beams).to(token_ids).repeat(batch_size)  # 表示的是indice
            eos_beam_idx = from_which_beam[:, :num_beams].reshape(-1)  # 表示的是从哪个beam获取得到的
        else:
            # 将每个batch中在num_beam内的序列添加到结束中, 为1的地方需要结束了
            effective_eos_mask = next_tokens[:, :num_beams].eq(_eos_token_id)  # batch_size x num_beams
            if effective_eos_mask.sum().gt(0):
                eos_batch_idx, eos_beam_ind = effective_eos_mask.nonzero(as_tuple=True)
                # 是由于from_which_beam是 (batch_size, 2*num_beams)的，所以需要2*num_beams
                eos_beam_idx = eos_batch_idx * num_beams * 2 + eos_beam_ind
                eos_beam_idx = from_which_beam.view(-1)[eos_beam_idx]  # 获取真实的从哪个beam获取的eos
            else:
                flag = False

        if flag:
            _token_ids = torch.cat([token_ids, _next_tokens], dim=-1)
            for batch_idx, beam_ind, beam_idx in zip(eos_batch_idx.tolist(), eos_beam_ind.tolist(),
                                                     eos_beam_idx.tolist()):
                if not dones[batch_idx]:
                    score = next_scores[batch_idx, beam_ind].item()
                    # 之后需要在结尾新增一个eos
                    if _eos_token_id!=-1:
                        hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx, :cur_len].clone(), score)
                    else:
                        hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx].clone(), score)

        # 更改state状态, 重组token_ids
        reorder_inds = (batch_inds_with_numbeams_interval + _from_which_beam).view(-1)  # flatten成一维
        state.reorder_state(reorder_inds)
        # 重新组织token_ids的状态
        token_ids = torch.cat([token_ids.index_select(index=reorder_inds, dim=0), _next_tokens], dim=-1)

        for batch_idx in range(batch_size):
            dones[batch_idx] = dones[batch_idx] or hypos[batch_idx].is_done(next_scores[batch_idx, 0].item()) or \
                               max_lengths[batch_idx*num_beams]==cur_len+1

        cur_len += 1

        if all(dones):
            break

    # select the best hypotheses
    tgt_len = token_ids.new_zeros(batch_size)
    best = []

    for i, hypotheses in enumerate(hypos):
        temp = []
        for res in hypotheses.hyp:#每个字的抽取结果
            temp.append(res[1].cpu().tolist()+[1])
        best.append(temp)
    # for i, hypotheses in enumerate(hypos):
    #     best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
    #     # 把上面替换为非eos的词替换回eos
    #     if _eos_token_id!=-1:
    #         best_hyp = torch.cat([best_hyp, best_hyp.new_ones(1)*_eos_token_id])
    #     tgt_len[i] = len(best_hyp)
    #     best.append(best_hyp)
    #
    # # generate target batch
    # decoded = token_ids.new_zeros(batch_size, tgt_len.max().item()).fill_(pad_token_id)
    # for i, hypo in enumerate(best):
    #     decoded[i, :tgt_len[i]] = hypo

    # return decoded,torch.LongTensor(range(len(decoded))).to(decoded.device)
    return best,

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty


