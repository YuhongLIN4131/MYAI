import torch
from .modeing_bart import BartEncoder, BartDecoder, BartModel
from transformers import BartTokenizer
from fastNLP import seq_len_to_mask
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch.nn.functional as F
from fastNLP.models import Seq2SeqModel
from torch import nn
import math
import json
from itertools import chain

def tokenizess(tokenizer, word):
    bpes1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word, add_prefix_space=True))
    bpes2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
    bpes3 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word.lower(), add_prefix_space=True))
    bpes4 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word.lower()))
    if len(bpes1) == 1:
        return bpes1
    elif len(bpes2) == 1:
        return bpes2
    elif len(bpes3) == 1:
        return bpes3
    elif len(bpes4) == 1:
        return bpes4
    else:
        return bpes1

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        # features_output1 = F.relu(features_output1)
        features_output1 = F.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2
class wieght_layer(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(wieght_layer, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        # features_output1 = F.relu(features_output1)
        features_output1 = F.gelu(features_output1)
        # features_output1 = torch.tanh(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2
class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

    def forward(self, src_tokens, src_seq_len,attention,position_ids):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        batch_size, seq_len, seq_len = attention.size()
        ll = torch.eye(seq_len, seq_len).to(attention.device)
        attention = (attention + ll.unsqueeze(0).expand(batch_size, -1, -1)) > 0
        assert position_ids is not None
        dict = self.bart_encoder(input_ids=src_tokens, attention_mask=attention, return_dict=True,position_ids=position_ids,
                                 output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states

        return encoder_outputs, mask, hidden_states


class FBartDecoder(Seq2SeqDecoder):
    def __init__(self, decoder, pad_token_id,use_decoder=True,use_cat=True,OOV_Integrate=True,label_ids=None, label_in_context_tail=True):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)  # 返回矩阵上三角部分，其余部分定义为0
        # 如果diagonal为空，输入矩阵保留主对角线与主对角线以上的元素；
        # 如果diagonal为正数n，输入矩阵保留主对角线与主对角线以上除去n行的元素；（上三角不要对角线）
        # 如果diagonal为负数 - n，输入矩阵保留主对角线与主对角线以上与主对角线下方h行对角线的元素；
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        self.use_decoder = use_decoder
        self.use_cat = use_cat
        self.three_decoder = False
        self.label_in_context_tail = label_in_context_tail#标签是否放在句子的最后，吸取句子的信息
        self.OOV_Integrate = OOV_Integrate
        self.label_ids = label_ids
        mapping = torch.LongTensor([0,1]+label_ids)  #存储特殊符号的token,
        self.register_buffer('mapping', mapping)
        hidden_size = decoder.embed_tokens.weight.size(1)
        if self.use_cat:
            self.special_output = MultiNonLinearClassifier(hidden_size * 2, 1, 0.3)
        if self.OOV_Integrate:
            self.weight_layer = wieght_layer(hidden_size * 2, 1, 0.1)  # 根据上一状态和当前的输出状态确定
            self.max_or_attention = "Attention"


    def forward(self, tokens, state):
        # bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]  ##得到特殊符号的token映射

        src_tokens_index = tokens - self.src_start_index  # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:  #
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)
        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                ##一个上三角矩阵，每个字不去attend之后的字
                                return_dict=True)
            hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
            hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        # hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full(
            (hidden_state.size(0), hidden_state.size(1), self.src_start_index + src_tokens.size(-1)),
            fill_value=-1e24)

        # 首先计算的是特殊符号
        '''对于特殊符号的话，输出是固定的，可以用网络判断'''
        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[2:3])  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[
                                            self.label_start_id:self.label_end_id])  # bsz x max_len x num_class
        # bsz x max_word_len x hidden_size
        src_outputs = state.encoder_output

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)

        mask = mask.unsqueeze(1).__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits

    def decode(self, tokens, state,test=False):
        logits = self(tokens, state,test)


        return logits[:, -1]


class CaGFBartDecoder(FBartDecoder):
    def __init__(self, decoder, pad_token_id, use_decoder=True,use_cat=True,OOV_Integrate=True,label_ids=None,label_in_context_tail=True):
        super().__init__(decoder, pad_token_id,use_decoder=use_decoder,use_cat=use_cat,OOV_Integrate=OOV_Integrate,label_ids=label_ids,label_in_context_tail=label_in_context_tail)
        self.dropout_layer = nn.Dropout(0.3)

    def forward(self, tokens, state,test=False, Discriminator=False):
        bsz, max_len = tokens.size()
        src_encoder_outputs = state.encoder_output  # batch_size,seq,1024
        bsz, seq_len, hidden_size = src_encoder_outputs.size()
        src_encoder_pad_mask = state.encoder_mask
        '''使用分词头token编码'''
        encoder_outputs = state.org_embedding
        encoder_pad_mask = state.org_mask
        tgt_seq_len = state.tgt_seq_len
        decoder_embedding_index = state.decoder_embedding_index
        mask_query = state.mask_query  # 屏蔽所有的query词
        specific_tokens_mask =torch.ones([bsz,len(self.mapping)],dtype=torch.long).to(mask_query.device)
        specific_tokens_mask[:,0]=0##第一个点是不可能获得的
        mask_query = torch.cat((specific_tokens_mask,mask_query[:,:-(len(self.mapping)-1)]),dim=1)
        src_mask1 = mask_query  # 不等于填充的值
        '''应该让'''
        if self.label_in_context_tail:#标签在模型的最后
            all_embedding = encoder_outputs.gather(index=decoder_embedding_index.unsqueeze(2).expand(-1, -1, hidden_size), dim=1)
        else:#特殊标签单独与句子之外
            start_and_end_embedding = self.decoder.embed_tokens.weight[self.mapping].unsqueeze(0).expand(bsz,-1,-1)#直接使用最底层的embedding了
            all_embedding = torch.cat((start_and_end_embedding, encoder_outputs[:,:-(len(self.mapping)-1)]), dim=1)
        target_tokens_output = all_embedding.gather(
            index=tokens.unsqueeze(2).expand(-1, -1, hidden_size),
            dim=1)
        if self.training and test==False:
            decoder_pad_mask = (~seq_len_to_mask(tgt_seq_len, max_len=tokens.size(1)))
            decoder_pad_mask = decoder_pad_mask[:,:-1]
            tokens = tokens[:, :-1]
            target_tokens_embedding = target_tokens_output[:, :-1]
            now_causal_masks = self.causal_masks[:tokens.size(1), :tokens.size(1)]
            '''使用Taboo随机mask0.1的没有实体的样本，减少漏标损害'''
            dict = self.decoder(input_ids=tokens,
                                target_tokens_embedding=target_tokens_embedding,
                                encoder_hidden_states=src_encoder_outputs,
                                encoder_padding_mask=src_encoder_pad_mask,#掩码模型，屏蔽一部分的非实体位置
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=now_causal_masks,
                                return_dict=True)
            hidden_state = dict.last_hidden_state[:, 1:]  # 不需要预测第一个字符  bsz x max_len x hidden_size
            decoder_mask1 = (~decoder_pad_mask)[:, 1:]  # 表示那些decoder是非填充的
            target_tokens_embedding = target_tokens_embedding[:,1:]
        else:
            if max_len == 2:  # 起点信息需要加载
                target_tokens_embedding = target_tokens_output
                past_key_values = None
                dict = self.decoder(input_ids=tokens[:, 0:1],
                                    target_tokens_embedding=target_tokens_embedding[:, 0:1],
                                    encoder_hidden_states=src_encoder_outputs,
                                    encoder_padding_mask=src_encoder_pad_mask,  # 掩码模型
                                    decoder_padding_mask=tokens[:, 0:1] == self.pad_token_id,
                                    decoder_causal_mask=self.causal_masks[:1,
                                                        :1],
                                    past_key_values=past_key_values,
                                    use_prompt_cache=True,
                                    return_dict=True)
                past_key_values = dict.past_key_values
            else:
                target_tokens_embedding = target_tokens_output
                past_key_values = state.past_key_values
            '''等下，映射好像不对'''
            dict = self.decoder(input_ids=tokens,  # 这里的token就是前面预测出来的加上后面填充为0的
                                target_tokens_embedding=target_tokens_embedding,
                                encoder_hidden_states=src_encoder_outputs,
                                encoder_padding_mask=src_encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
            hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
            decoder_mask1 = torch.zeros([hidden_state.size(0), hidden_state.size(1)]).to(
                hidden_state.device) == 0  # 表示那些decoder是非填充的
            target_tokens_embedding=target_tokens_embedding[:,-1:]
        batch_size, target_len, hidden_len = hidden_state.size()
        if not self.training:
            state.past_key_values = dict.past_key_values
        logits = hidden_state.new_full(
            (hidden_state.size(0), hidden_state.size(1), all_embedding.size(1)),
            fill_value=-1e30)
        if self.use_decoder==True:#那么
            src_outputs = all_embedding
        else:#这个情况需要重算all_embedding,用encoder的输出编码来计算分数
            src_outputs = state.src_embed_outputs.gather(
                index=decoder_embedding_index.unsqueeze(2).expand(-1, -1, hidden_size), dim=1)
        batch_size, scr_seq_len, _ = src_outputs.size()
        if self.use_cat:
            word_scores_2 = torch.ones([hidden_state.size(0), hidden_state.size(1), src_outputs.size(1)],
                                       dtype=logits.dtype).to(logits.device) * -1e30
            word_scores_2 = word_scores_2.view(-1)
            embedding_mask = (src_mask1.unsqueeze(1).expand(-1, target_len, -1) == 1) & decoder_mask1.unsqueeze(2).expand(-1,-1,scr_seq_len)
            '''尽量节约内存，'''
            posit = torch.nonzero(embedding_mask.view(-1)).squeeze(-1)
            # word_scores_2[posit] = self.special_output(
            #     torch.cat((torch.masked_select(hidden_state.unsqueeze(2).expand(-1, -1, scr_seq_len, -1),embedding_mask.unsqueeze(3).
            #                                    expand(-1,-1,-1,hidden_len)).view(-1, hidden_len),
            #                torch.masked_select(src_outputs.unsqueeze(1).expand(-1, target_len, -1, -1),embedding_mask.unsqueeze(3).
            #                                    expand(-1,-1,-1,hidden_len)).view(-1, hidden_len),
            #                ),
            #               dim=-1)).squeeze(-1)  # batch_size,target_len,src_outputs,2048
            '''换成原编码之间cat'''
            word_scores_2[posit] = self.special_output(
                torch.cat((torch.masked_select(target_tokens_embedding.unsqueeze(2).expand(-1, -1, scr_seq_len, -1),embedding_mask.unsqueeze(3).
                                               expand(-1,-1,-1,hidden_len)).view(-1, hidden_len),
                           torch.masked_select(src_outputs.unsqueeze(1).expand(-1, target_len, -1, -1),embedding_mask.unsqueeze(3).
                                               expand(-1,-1,-1,hidden_len)).view(-1, hidden_len),
                           ),
                          dim=-1)).squeeze(-1)  # batch_size,target_len,src_outputs,2048

            word_scores_2 = word_scores_2.view(batch_size, target_len, scr_seq_len)
        '''计算点积的结果'''
        hidden_state = self.dropout_layer(hidden_state)
        src_outputs = self.dropout_layer(src_outputs)
        '''input_embed用的是decoder的token编码，而src_outputs是encoder隐藏层输出'''
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        mask = mask_query.unsqueeze(1) ==0  # batch_size,1,seq_len,1024
        if self.use_cat:
            word_scores = (word_scores + word_scores_2) / 2
        word_scores = word_scores.masked_fill(mask, -1e30)  ##填充值的分数补上负无穷
        '''计算一下对比损失：对比的两边为：编码的向量表示（句子、pad、终点、实体类别），mask表示那些点不要计算相似度'''
        logits = word_scores  # 生成的句子中为某个词的分数
        '''得到模长，也就是进行l2 norm  torch.einsum('blh,bnh->bln', hidden_state, src_outputs) '''
        return logits


class BartSeq2SeqModel(Seq2SeqModel):
    @classmethod
    def build_model(cls, bart_model, tokenizer, use_decoder=True, use_cat=True, OOV_Integrate=True, label_ids=None, label_in_context_tail=True):
        '''需要'''
        model = BartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens) + num_tokens)
        '''默认情况下，encoder和decoder共享相同的embedding'''
        encoder = model.encoder  # 并不会编码关于标签的特殊符号
        decoder = model.decoder
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':  # 特殊字符
                index = tokenizess(tokenizer, token)
                if len(index) > 1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index >= num_tokens, (index, num_tokens, token)
                indexes = []
                text = token[2:-2].split()
                for izs in text:
                    indexes.append(tokenizess(tokenizer, izs))
                indexes = list(chain(*indexes))
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                '''改变了decoder 关于标签类别的embedding'''
                model.decoder.embed_tokens.weight.data[index] = embed  ##用这几个词的平均结果初始化这符号词
                model.encoder.embed_tokens.weight.data[index] = embed  ##用这几个词的平均结果初始化这符号词
        encoder = FBartEncoder(encoder)
        decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id,
                                  use_decoder=use_decoder,use_cat=use_cat,OOV_Integrate=OOV_Integrate,label_ids=label_ids,label_in_context_tail=label_in_context_tail)
        return cls(encoder=encoder, decoder=decoder)

    def prepare_state(self, src_tokens, src_seq_len=None, tgt_seq_len=None, mask_query=None
                    ,OOV_con=None,position_ids=None,attention=None,decoder_embedding_index=None,flag=False
                      ):
        encoder_outputs,mask, hidden_states = self.encoder(src_tokens,src_seq_len,attention,position_ids)
        src_embed_outputs = hidden_states[0]
        '''判别式：判断是否有解，等下，这里平均好像忘记删除padding了'''
        '''重要:encoder_padding_mask是那些填充那些mask,
        '''
        hidden_size = encoder_outputs.size(2)
        org_mask=None#用于给decoder赋值的编码
        org_embedding = None#用于给decoder赋值的mask
        '''NOTE:对于那些不使用整合OOV词的情况,有两个decoder_output编码
        第一个decoder_output就是原来的encoder_outputs，他用来作为decoder的条件输入
        第二个decoder_output是只保留头token的情况，用来给decoder的输入赋值'''
        if self.decoder.use_decoder==True and self.decoder.OOV_Integrate==False:#
            '''使用encoder后的头分词,'''
            decoder_encoder=encoder_outputs#条件输入
            decoder_encoder_mask = mask#条件输入的mask
            OOV_con_now = (OOV_con < 0) * 1 + OOV_con
            OOV_flag = OOV_con >= 0
            org_mask = OOV_flag.sum(dim=-1) > 0
            OOV_con_now = OOV_con_now[:, :, 0]
            org_embedding = encoder_outputs.gather(
                index=OOV_con_now.unsqueeze(2).expand(-1, -1, hidden_size), dim=1)#转化给decoder的输入编码
        elif self.decoder.use_decoder==True and self.decoder.OOV_Integrate==True:#
            ''''使用decoder整合后的结果'''
            '''最大池化操作或者根据 头token计算'''
            OOV_flag = OOV_con >= 0
            OOV_con_now = (OOV_con < 0) * 1 + OOV_con
            decoder_encoder_mask = OOV_flag.sum(dim=-1) > 0
            batch_size, seq_OOV_lenn, max_oov_len = OOV_con_now.size()
            '''根据attention合并分词'''
            decoder_encoder = encoder_outputs.unsqueeze(1).expand(-1, seq_OOV_lenn, -1, -1).gather(
                index=OOV_con_now.unsqueeze(3).expand(-1, -1, -1, hidden_size), dim=2)
            if self.decoder.max_or_attention=='Attention':#加权池化
                Attention = self.decoder.weight_layer(torch.cat((encoder_outputs, encoder_outputs.gather(
                            index=(src_seq_len-len(self.decoder.label_ids)-1).unsqueeze(1).unsqueeze(2).expand(-1, encoder_outputs.size(1), hidden_size), dim=1)),
                                                                dim=-1)).squeeze(-1)
                attention_OOV = Attention.unsqueeze(1).expand(-1, seq_OOV_lenn, -1).gather(
                    index=OOV_con_now, dim=2)  ##现在已经取得了attention
                attention_OOV = attention_OOV.masked_fill((~OOV_flag), -1e30)
                attention_OOV = torch.softmax(attention_OOV, dim=-1)
                decoder_encoder = (decoder_encoder * attention_OOV.unsqueeze(3) * OOV_flag.unsqueeze(3)).sum(dim=-2)
            else:#只有加权池化或者最大池化
                min_data = torch.min(decoder_encoder)-1
                decoder_encoder = (decoder_encoder * OOV_flag.unsqueeze(3) + (~OOV_flag).unsqueeze(3)*min_data).max(dim=-2)[0]
            '''这个情况下条件输入和decoder的源输入一致'''
            org_embedding = decoder_encoder
            org_mask = decoder_encoder_mask
        else:#直接使用原编码
            decoder_encoder =encoder_outputs#条件
            decoder_encoder_mask =mask
            org_embedding = self.decoder.decoder.embed_tokens.weight[src_tokens]
            OOV_con_now = (OOV_con < 0) * 1 + OOV_con
            OOV_flag = OOV_con >= 0
            org_mask = OOV_flag.sum(dim=-1) > 0
            OOV_con_now = OOV_con_now[:, :, 0]
            org_embedding = org_embedding.gather(
                index=OOV_con_now.unsqueeze(2).expand(-1, -1, hidden_size), dim=1)
            src_embed_outputs = encoder_outputs.gather(
                index=OOV_con_now.unsqueeze(2).expand(-1, -1, hidden_size), dim=1)##这个是用来相乘的
        '''第一步：先根据'''
        '''对起点同样判断，只不过只判断起点的类型'''
        if flag:#训练式
            return decoder_encoder, decoder_encoder_mask, src_tokens, src_embed_outputs, mask_query, org_mask, org_embedding, decoder_embedding_index
        else:#预测式
            state = BartState(decoder_encoder, decoder_encoder_mask, src_tokens, src_embed_outputs, mask_query
                            ,org_mask,org_embedding,decoder_embedding_index,tgt_seq_len)
            return state


    def forward(self, src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, mask_query=None,OOV_con=None,position_ids=None,attention=None,error_flag=None,decoder_embedding_index=None
                ):
        """
        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        '''为每一个标签都带一个state'''
        decoder_encoder, decoder_encoder_mask, src_tokens, src_embed_outputs, mask_query, org_mask, org_embedding, decoder_embedding_index = self.prepare_state(src_tokens, src_seq_len, tgt_seq_len, mask_query, OOV_con, position_ids, attention,decoder_embedding_index,flag=True)
        batch_size, seq_len, target_length = tgt_tokens.size()
        indices = torch.arange(batch_size, dtype=torch.long).to(src_tokens.device)
        indices = indices.repeat_interleave(seq_len)
        all_word_entity_length = ((tgt_tokens != 1).sum(dim=-1))  # ，batch*ser_len
        all_word_entity_length = all_word_entity_length.view(-1)
        error_flag = error_flag.view(-1)  # 用于束搜索生成的负样本的
        all_train_pos = torch.nonzero(all_word_entity_length > 0).squeeze(-1)#训练标签的所有位置
        all_train_mask_query = mask_query[indices[all_train_pos]]
        decoder_output = decoder_encoder.new_full((len(all_train_pos), target_length-2, decoder_embedding_index.size(1)),fill_value=-1e30)#所有训练样本的分数
        specific_tokens_mask = torch.ones([all_train_mask_query.size(0), len(self.decoder.mapping)], dtype=torch.long).to(
            mask_query.device)
        specific_tokens_mask[:, 0] = 0  ##第一个点是不可能获得的
        target_mask = torch.cat((specific_tokens_mask, all_train_mask_query[:, :-(len(self.decoder.mapping) - 1)]), dim=1)
        '''将位置转化为需要训练的位置'''
        all_word_entity_length = all_word_entity_length[all_train_pos]
        indices = indices[all_train_pos]
        error_flag=error_flag[all_train_pos]
        tgt_tokens = tgt_tokens.view(-1, target_length)[all_train_pos]
        tgt_seq_len = tgt_seq_len.view(-1)[all_train_pos]
        '''分开处理，先处理判别式的实体，再处理生成式的实体'''
        '''划分句子，《=4的划分到一块，大于4的划分到一块的尝试'''
        if ((all_word_entity_length <=5)).sum() > 0:#可能也没有
            noo_entity_pos = torch.nonzero((all_word_entity_length <=5)).squeeze(-1)
            pos1 = indices[noo_entity_pos]
            tgt_tokens1= tgt_tokens[noo_entity_pos,0:6]#只需要前3个
            # tgt_seq_len1=tgt_seq_len[noo_entity_pos]
            state=BartState(decoder_encoder[pos1], decoder_encoder_mask[pos1], src_tokens[pos1], src_embed_outputs[pos1], mask_query[pos1]
                            ,org_mask[pos1],org_embedding[pos1],decoder_embedding_index[pos1],tgt_seq_len[noo_entity_pos])
            decoder_output[noo_entity_pos,0:4] = self.decoder(tgt_tokens1, state)#问题是有可能有些batch没有机会有起点

        '''第二步：3-6长度的'''
        if ((all_word_entity_length > 5) & (all_word_entity_length <= 15)).sum()>0:
            '''再调用两次一次划分为<=6，另一个是6之后的，用时间换空间（因为有些数据集中的实体数量太多了）'''
            noo_entity_pos2 = torch.nonzero((all_word_entity_length > 5) & (all_word_entity_length <= 15)).squeeze(-1)
            pos2 = indices[noo_entity_pos2]
            tgt_tokens2 = tgt_tokens[noo_entity_pos2, 0:16]  # 只需要前3个
            state = BartState(decoder_encoder[pos2], decoder_encoder_mask[pos2], src_tokens[pos2],
                              src_embed_outputs[pos2], mask_query[pos2]
                              , org_mask[pos2], org_embedding[pos2], decoder_embedding_index[pos2],tgt_seq_len[noo_entity_pos2])
            decoder_output[noo_entity_pos2, 0:14] = self.decoder(tgt_tokens2, state) # 问题是有可能有些batch没有机会有起点
        if ((all_word_entity_length > 15) & (all_word_entity_length <=30)).sum()>0:
            '''再调用两次一次划分为<=6，另一个是6之后的，用时间换空间（因为有些数据集中的实体数量太多了）'''
            noo_entity_pos3 = torch.nonzero((all_word_entity_length > 15) & (all_word_entity_length <=30)).squeeze(-1)
            pos3 = indices[noo_entity_pos3]
            tgt_tokens3 = tgt_tokens[noo_entity_pos3,0:31]  # 只需要前3个
            state = BartState(decoder_encoder[pos3], decoder_encoder_mask[pos3], src_tokens[pos3],
                              src_embed_outputs[pos3], mask_query[pos3]
                              , org_mask[pos3], org_embedding[pos3], decoder_embedding_index[pos3],tgt_seq_len[noo_entity_pos3])
            decoder_output[noo_entity_pos3,0:29] = self.decoder(tgt_tokens3, state)  # 问题是有可能有些batch没有机会有起点
        if ((all_word_entity_length > 30)).sum()>0:
            '''再调用两次一次划分为<=6，另一个是6之后的，用时间换空间（因为有些数据集中的实体数量太多了）'''
            noo_entity_pos4 = torch.nonzero((all_word_entity_length > 30)).squeeze(-1)
            pos4 = indices[noo_entity_pos4]
            tgt_tokens4 = tgt_tokens[noo_entity_pos4]
            state = BartState(decoder_encoder[pos4], decoder_encoder_mask[pos4], src_tokens[pos4],
                              src_embed_outputs[pos4], mask_query[pos4]
                              , org_mask[pos4], org_embedding[pos4], decoder_embedding_index[pos4],tgt_seq_len[noo_entity_pos4])
            decoder_output[noo_entity_pos4] = self.decoder(tgt_tokens4, state)  # 问题是有可能有些batch没有机会有起点
        tgt_tokens = tgt_tokens[:, 1:]  # 第一个不计算损失
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output,'aim_token':tgt_tokens,'target_mask':target_mask,'position':all_train_pos,'tgt_seq_len':tgt_seq_len,'error_flag':error_flag}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")


class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, src_embed_outputs, mask_query,org_mask,org_embedding,decoder_embedding_index,tgt_seq_len):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.src_embed_outputs = src_embed_outputs
        self.mask_query = mask_query
        self.org_mask = org_mask
        self.org_embedding = org_embedding
        self.tgt_seq_len = tgt_seq_len
        self.decoder_embedding_index = decoder_embedding_index

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        self.mask_query = self._reorder_state(self.mask_query, indices)
        self.decoder_embedding_index = self._reorder_state(self.decoder_embedding_index, indices)
        if self.tgt_seq_len is not None:
            self.tgt_seq_len = self._reorder_state(self.tgt_seq_len, indices)
        if self.org_mask is not None:
            self.org_mask = self._reorder_state(self.org_mask, indices)
            self.org_embedding = self._reorder_state(self.org_embedding, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new