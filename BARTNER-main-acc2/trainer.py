# encoding: utf-8
from itertools import permutations
import argparse
import os
from collections import namedtuple
from typing import Dict
import json
from itertools import chain
import pytorch_lightning as pl
import torch
import itertools
import torch.nn.functional as F
import random
import copy
import numpy as np
from fastNLP import seq_len_to_mask
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from tokenizers import BertWordPieceTokenizer
from torch import Tensor
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BartTokenizer
from torch.optim import SGD
from models.bart import BartSeq2SeqModel
from models.metrics import Seq2SeqSpanMetric
from transformers import AdamW
from models.generater import SequenceGeneratorModel
from datasets.mrc_ner_dataset import MultiTaskBatchSampler
from datasets.mrc_ner_dataset import MRCNERDataset
from datasets.mrc_ner_dataset import MultiTaskDataset
from datasets.truncate_dataset import TruncateDataset
from datasets.collate_functions import collate_to_max_length
from utils.get_parser import get_parser
from utils.radom_seed import set_random_seed
import logging
from models.losses import Seq2SeqLoss
from loss.focal_loss import FocalLoss
from loss.LabelSmoothingLoss import LabelSmoothingLoss


# seed_num = 1
# np.random.seed(seed_num)
# torch.manual_seed(seed_num)
# random.seed(seed_num)
# os.environ['PYTHONHASHSEED'] = str(seed_num)
# torch.cuda.manual_seed_all(seed_num)
# torch.cuda.manual_seed(seed_num)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False

TASK_ID2STRING={"Onto":0,"conll03":1,"ace2004":2,"ace2005":3,'genia':4,'cadec':5,'share2013':6,'share2014':7}
Dataset_label_number = [18, 4, 7, 7, 5, 1, 1, 1]
class BertLabeling(pl.LightningModule):
    """MLM Trainer"""

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
            self.args = args
        else:
            # eval mode
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        self.bert_dir = args.bert_config_dir
        self.data_dir = self.args.data_dir
        self.loss_type = self.args.loss_type
        self.optimizer = args.optimizer
        self.tokenizer = BartTokenizer.from_pretrained(self.bert_dir)
        '''????????????????????????'''
        self.use_decoder = args.use_decoder  # ????????????encoder??????????????????decoder??????
        self.OOV_Integrate = args.OOV_Integrate  # ????????????encoder??????OOV?????????????????????decoder????????????   ?????????use_decoder=True????????????
        self.use_cat = args.use_cat  # ???encoder??????????????????decoder????????????????????????????????????cat?????????????????????
        self.Negative_sampling = args.Negative_sampling  # ??????????????????
        self.Generate_Negative_example = args.Generate_Negative_example  # ????????????epoch?????????????????????????????????beam=4?????????????????????????????????????????????
        self.label_in_context_tail = args.label_in_context_tail  # ???????????????token??????encoder???????????????
        '''????????????????????????'''
        '''??????label?????????token,?????????????????????????????????????????????????????????token??????,????????????????????????????????????'''
        self.label_num = Dataset_label_number[args.task_id]  # ?????????????????????
        specific_token_file = self.args.data_dir+'/'+"specific_token.json"
        # specific_token_file = self.args.data_dir+'/'+"specific_token_three.json"
        specific_token = json.load(open(specific_token_file, encoding="utf-8"))
        mapping_uncased=[]
        for i in specific_token:
            mapping_uncased.append(i)#??????
        self.tokenizer.add_tokens(mapping_uncased)  # ??????????????????????????????
        self.tokenizer.unique_no_split_tokens = self.tokenizer.unique_no_split_tokens + mapping_uncased
        '''??????????????????????????????????????????token_id'''
        self.label2token = []
        for i in specific_token:
            self.label2token.append(self.tokenizer.convert_tokens_to_ids(i))
        '''?????????????????????'''
        self.target_shift = 2 + self.label_num  # ?????????????????????
        self.now_epoch=0
        self.metric = Seq2SeqSpanMetric()
        self.GetLoss=Seq2SeqLoss(loss_type=self.loss_type)
        length_penalty = 1
        self.model = BartSeq2SeqModel.build_model(self.bert_dir, self.tokenizer, use_decoder=self.use_decoder, use_cat=self.use_cat,
                                                  OOV_Integrate=self.OOV_Integrate,label_ids=self.label2token,label_in_context_tail=self.label_in_context_tail)
        print("??????????????????{}".format(self.model.decoder.decoder.embed_tokens.weight.data.size(0)))
        self.model = SequenceGeneratorModel(self.model, bos_token_id=args.bos_token_id,
                                       eos_token_id=args.eos_token_id,max_length=args.max_len,
                                       max_len_a=args.max_len_a, num_beams=args.num_beams, do_sample=False,
                                       repetition_penalty=1, length_penalty=length_penalty, pad_token_id=1,
                                       restricter=None,target_shift=self.target_shift)
        all_data_train = json.load(open(self.args.data_dir + "/entity.train", encoding="utf-8"))
        all_data_dev = json.load(open(self.args.data_dir + "/entity.dev", encoding="utf-8"))
        all_data_test = json.load(open(self.args.data_dir + "/entity.test", encoding="utf-8"))
        self.train_data_arguments = all_data_train
        self.dev_data = all_data_test
        self.test_data = all_data_test

        self.dev_num=0
        self.Generateor={}#????????????????????????????????????
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--mrc_dropout", type=float, default=0.3,
                            help="mrc dropout rate")
        parser.add_argument("--chinese", action="store_true",
                            help="is chinese dataset")
        parser.add_argument("--optimizer", choices=["adamw", "sgd", "torch.adam"], default="adamw",
                            help="loss type")
        parser.add_argument("--bos_token_id", type=int, default=0, help="???????????????token")
        parser.add_argument("--eos_token_id", type=int, default=1, help="???????????????token")
        parser.add_argument("--dice_smooth", type=float, default=1e-8, help="smooth value of dice loss")
        parser.add_argument("--num_beams", type=int, default=1, help="??????????????????")
        parser.add_argument("--loss_type", choices=["cross_entropy", "LabelSmoothingLoss", "focal"], default="adamw",
                            help="????????????")
        parser.add_argument("--target_type",choices=["word", "span", "bpe"], default="word", help="?????????????????????")
        parser.add_argument("--max_len", type=int, default=10, help="????????????????????????")
        parser.add_argument("--task_id", type=int, default=6, help="??????????????????????????????")
        parser.add_argument("--dataset_name", type=str, default="share2013", help="???????????????")
        parser.add_argument("--length_penalty", type=float, default=1.0, help="???????????????")
        parser.add_argument("--part_entity_Negative_sampling", type=float, default=0.5, help="??????????????????????????????")
        parser.add_argument("--max_len_a", type=float, default=1.0, help="???????????? = max_len+ src_len*max_len_a")
        parser.add_argument("--final_div_factor", type=float, default=1e4,
                            help="final div factor of linear decay scheduler")
        return parser
    def tokenizes(self, word):
        bpes1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True))
        bpes2 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))
        bpes3 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word.lower(), add_prefix_space=True))
        bpes4 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word.lower()))
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
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon,)
        elif self.optimizer == "torch.adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)
        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        warmup_steps = round(t_total * 0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.args.lr, pct_start=float(warmup_steps/t_total),
            final_div_factor=self.args.final_div_factor,
            total_steps=t_total, anneal_strategy='linear'
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


    def training_step(self, batch, batch_idx):
        """"""
        tf_board_logs = {
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        target, word_bpes,attention, position_ids,OOV_con,mask_query,error_flag,tgt_seq_len,decoder_embedding_index,all_word_entity_label,src_seq_len,target_span,sample_id,task_coding = batch
        '''???????????????????????????????????????????????????????????????'''
        batch_size,target_number,entity_len,dimss=all_word_entity_label.size()
        outputs = self.model(word_bpes, target, src_seq_len=src_seq_len, tgt_seq_len=tgt_seq_len,mask_query=mask_query,OOV_con=OOV_con,position_ids=position_ids,attention=attention,error_flag=error_flag,decoder_embedding_index=decoder_embedding_index)
        all_word_entity_label = all_word_entity_label.view(batch_size*target_number,entity_len,dimss)[outputs["position"]]
        loss = self.GetLoss.get_loss(outputs['aim_token'], outputs["pred"],target_mask=outputs["target_mask"],labels=all_word_entity_label)
        tf_board_logs[f"train_loss"] = loss
        return {'loss': loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        """"""

        output = {}
        target, word_bpes,attention, position_ids,OOV_con,mask_query,error_flag,tgt_seq_len,decoder_embedding_index,all_word_entity_label,src_seq_len,target_span,sample_id,task_coding = batch
        if task_coding[0]==0:#??????????????????
            outputs = self.model.predict(word_bpes, src_seq_len=src_seq_len,mask_query=mask_query,OOV_con=OOV_con,position_ids=position_ids,attention=attention,tgt_tokens=target,decoder_embedding_index=decoder_embedding_index)
            pred =outputs
            '''??????,'''
            fn, tp, fp = self.metric.evaluate(target_span, pred)
            """"""
            output['tp']= tp
            output['fp']= fp
            output['fn']= fn
        else:
            outputs = self.model.predict(word_bpes, src_seq_len=src_seq_len, mask_query=mask_query, OOV_con=OOV_con,
                                          attention=attention, tgt_tokens=target,decoder_embedding_index=decoder_embedding_index,Generate_negative_samples=True)
            for i,ids in enumerate(sample_id):#???????????????????????????????????????????????????????????????????????????????????????????????????????????????
                self.Generateor[ids]=outputs[i]
            """"""
            output['tp'] = 0
            output['fp'] = 0
            output['fn'] = 0
        return output

    def validation_epoch_end(self, outputs):
        """"""
        tensorboard_logs = {}
        span_tp = torch.LongTensor([x['tp'] for x in outputs]).sum()
        span_fp = torch.LongTensor([x['fp'] for x in outputs]).sum()
        span_fn = torch.LongTensor([x['fn'] for x in outputs]).sum()
        span_recall = span_tp / (span_tp + span_fn + 1e-10)
        span_precision = span_tp / (span_tp + span_fp + 1e-10)
        span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
        tensorboard_logs[f"span_precision"] = span_precision
        tensorboard_logs[f"span_recall"] = span_recall
        tensorboard_logs[f"span_f1"] = span_f1
        if ~self.Generate_Negative_example:
            print("??????????????????{}????????????{}???F1???{}".format(span_precision,span_recall,span_f1))
            print("?????????span_tp???{}???span_fp???{}???span_fn???{}".format(span_tp,span_fp,span_fn))
        return {'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        """"""
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(
        self,
        outputs
    ) -> Dict[str, Dict[str, Tensor]]:
        """"""
        return self.validation_epoch_end(outputs)

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def test_dataloader(self):
        return self.get_dataloader("test")

    def get_dataloader(self, prefix="train", limit: int = None) -> DataLoader:
        """get training dataloader"""
        """
        load_mmap_dataset
        """
        if prefix=="train":
            task0 = self.train_data_arguments
        elif prefix=="dev":
            '''??????epoch???????????????????????????????????????????????????'''
            self.dev_num +=1
            if self.Generate_Negative_example:#??????????????????????????????????????????
                if self.dev_num%2==1:
                    task0 = self.test_data
                else:
                    task0 =self.train_data_arguments
            else:
                task0 =self.test_data
        else:
            task0 = self.test_data

        dataset = MRCNERDataset(task_data=task0,
                                tokenizer=self.tokenizer,
                                max_length=self.args.max_length,
                                pad_to_maxlen=False,
                                prefix=prefix,
                                task_id=self.args.task_id,#?????????????????????
                                label2token =self.label2token,#?????????????????????token
                                label_in_context_tail =self.label_in_context_tail, #????????????????????????????????????
                                Negative_sampling = self.Negative_sampling,  # ??????????????????
                                Generateor=self.Generateor if (prefix=="train" and self.Generate_Negative_example) else None,
                                part_entity_Negative_sampling = self.args.part_entity_Negative_sampling
                                )
        if limit is not None:
            dataset = TruncateDataset(dataset, limit)
        if prefix == "train":
            now_batch = self.args.batch_size
        elif self.dev_num%2==1 or self.Generate_Negative_example==False:
            now_batch = 16
        else:#?????????
            now_batch = 4
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=now_batch,
            num_workers=self.args.workers,
            shuffle=True if prefix == "train" else False,
            collate_fn=collate_to_max_length
        )
        return dataloader

def main():
    """main"""

    parser = get_parser()
    parser = BertLabeling.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    ##????????????????????????????????????
    args.bert_config_dir="/home/wyq/BARTNER-main-acc/facebook/bart-large"
    args.default_root_dir = "mnt/data/mrc/train_logs/debug"
    args.max_length = 320
    args.batch_size = 8
    args.lr = 1e-5
    args.gpus = "1"
    #args.precision = 16
    args.progress_bar_refresh_rate=10#???10???????????????
    args.max_epochs=100#
    # args.loss_type="LabelSmoothingLoss" #"cross_entropy", "LabelSmoothingLoss", "focal"
    args.loss_type="focal" #"cross_entropy", "LabelSmoothingLoss", "focal"
    args.dataset_name = "conll03"  # ?????????"Onto":0,"conll03":1,"ace2004":3,"ace2005":4,'genia':5,'cadec':6,'share2013':7,'share2014':8
    args.data_dir = '/home/wyq/BARTNER-main/data/' + args.dataset_name
    args.task_id=TASK_ID2STRING[args.dataset_name]#??????id
    args.gradient_clip_val = 1.0#
    '''????????????????????????'''
    args.use_decoder = True  # ????????????encoder??????????????????decoder??????
    args.OOV_Integrate = True  # ????????????encoder??????OOV?????????????????????decoder????????????   ?????????use_decoder=True????????????
    args.use_cat = True  # ???encoder??????????????????decoder????????????????????????????????????cat?????????????????????
    args.Negative_sampling = 0.3  # ??????????????????????????????????????????0.5??????????????????
    args.part_entity_Negative_sampling = 0.3  # ??????????????????????????????????????????????????????????????????????????????????????????
    args.Generate_Negative_example = False  # ????????????epoch?????????????????????????????????beam=4?????????????????????????????????????????????
    args.label_in_context_tail=True #???????????????token??????encoder???????????????
    args.num_beams=4#????????????????????????
    # args.pretrained_checkpoint ="/home/wyq/BARTNER_lightning/mnt/data/mrc/train_logs/debug/epoch=0.ckpt"
    model = BertLabeling(args)
    if args.pretrained_checkpoint:##??????????????????
        model.load_state_dict(torch.load(args.pretrained_checkpoint,
                                         map_location=torch.device('cpu'))["state_dict"])

    checkpoint_callback = ModelCheckpoint(
        filepath=args.default_root_dir,
        save_top_k=3,
        verbose=True,
        monitor="span_f1",
        period=-1,
        mode="max",
    )

    trainer = Trainer.from_argparse_args(
        args,
        reload_dataloaders_every_epoch=True,#????????????????????????????????????
        check_val_every_n_epoch=1,
        accumulate_grad_batches=1,
        val_check_interval=0.5 if args.Generate_Negative_example else 1.0,
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
