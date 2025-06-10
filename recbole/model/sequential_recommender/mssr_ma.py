import torch
from torch import nn
import math
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeatureSeqEmbLayer, FeedForward, VanillaAttention
from recbole.model.loss import BPRLoss
import copy
import torch.nn.functional as F
import numpy as np


class ACGMultiHeadAttention(nn.Module):

    def __init__(self, n_heads, hidden_size, attribute_hidden_size, feat_num, hidden_dropout_prob, attn_dropout_prob,
                 layer_norm_eps, fusion_type, max_len, ada_fuse=0):
        super(ACGMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attribute_attention_head_size = [int(_ / n_heads) for _ in attribute_hidden_size]
        self.attribute_all_head_size = [self.num_attention_heads * _ for _ in self.attribute_attention_head_size]
        self.fusion_type = fusion_type
        self.max_len = max_len
        self.feat_num = feat_num

        # item Q K V projection
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # pos Q K V projection
        self.query_p = nn.Linear(hidden_size, self.all_head_size)
        self.key_p = nn.Linear(hidden_size, self.all_head_size)
        self.value_p = nn.Linear(hidden_size, self.all_head_size)

        # item-attr  attr-item W for inter-seq attention
        self.query_ia = nn.ModuleList([copy.deepcopy(nn.Linear(hidden_size, self.attribute_all_head_size[_])) for _ in
             range(self.feat_num)])
        self.query_ai = nn.ModuleList([copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.all_head_size)) for _ in
             range(self.feat_num)])

        # pos-attr  attr-pos  W  for inter-seq att
        self.query_pa = nn.ModuleList([copy.deepcopy(nn.Linear(hidden_size, self.attribute_all_head_size[_])) for _ in
             range(self.feat_num)])
        self.query_ap = nn.ModuleList([copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.all_head_size)) for _ in
             range(self.feat_num)])

        # attr x - attr y  W  for inter-seq att
        self.query_attr_x_y = nn.ModuleList(
            [copy.deepcopy(
                nn.ModuleList(
                    [copy.deepcopy(nn.Linear(attribute_hidden_size[x], self.attribute_all_head_size[y])) for y in
                     range(self.feat_num)])
            ) for x in range(self.feat_num) ]
        )

        # Q K V projection for feat_num attr
        self.query_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in
             range(self.feat_num)])
        self.key_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in
             range(self.feat_num)])
        self.value_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in
             range(self.feat_num)])

        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Linear(self.max_len * ((2 + self.feat_num) ** 2), self.max_len)
        elif self.fusion_type == 'gate':
            self.fusion_layer = VanillaAttention(self.max_len, self.max_len)

        self.ada_fuse = ada_fuse

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.dense_attr = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], attribute_hidden_size[_])) for _
             in range(self.feat_num)])
        self.LayerNorm_attr = nn.ModuleList(
            [copy.deepcopy(nn.LayerNorm(attribute_hidden_size[_], eps=layer_norm_eps)) for _
             in range(self.feat_num)])

        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):  # partition of each attention head
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # [batch size, head num, seq len, emb dim]
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_attribute(self, x, i):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attribute_attention_head_size[i])
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attribute_table, position_embedding, hidden_state_attr, attention_mask, fusion_w, fusion_wa):
        # Item Q K V
        item_query_layer = self.transpose_for_scores(self.query(input_tensor))
        item_key_layer = self.transpose_for_scores(self.key(input_tensor))
        item_value_layer = self.transpose_for_scores(self.value(input_tensor))
        item_query_layer = item_query_layer.unsqueeze(1)  # [B,1,H,L,D]
        item_key_layer = item_key_layer.unsqueeze(1)

        # pos Q K
        pos_query_layer = self.transpose_for_scores(self.query_p(position_embedding))
        pos_key_layer = self.transpose_for_scores(self.key_p(position_embedding))
        pos_query_layer = pos_query_layer.unsqueeze(1)
        pos_key_layer = pos_key_layer.unsqueeze(1)

        raw_attention = []
        raw_attention_a = []
        attribute_query_layers, attribute_key_layers, attribute_value_layers = [], [], []
        hidden_state_attr_query, hidden_state_attr_key, hidden_state_attr_value = [], [], []

        # attribute  Q K V
        for i, (attribute_query, attribute_key, attribute_value) in enumerate(
                zip(self.query_layers, self.key_layers, self.value_layers)):
            attribute_tensor = attribute_table[i].squeeze(-2)
            attribute_query_layers.append(self.transpose_for_scores_attribute(attribute_query(attribute_tensor), i).unsqueeze(1))
            attribute_key_layers.append(self.transpose_for_scores_attribute(attribute_key(attribute_tensor), i).unsqueeze(1))
            attribute_value_layers.append(self.transpose_for_scores_attribute(attribute_value(attribute_tensor), i).unsqueeze(1))

            hidden_state_attr_tensor = hidden_state_attr[i].squeeze(-2)
            hidden_state_attr_query.append(self.transpose_for_scores_attribute(attribute_query(hidden_state_attr_tensor), i).unsqueeze(1))
            hidden_state_attr_key.append(self.transpose_for_scores_attribute(attribute_key(hidden_state_attr_tensor), i).unsqueeze(1))  # [B,1,H,L,D] list
            hidden_state_attr_value.append(self.transpose_for_scores_attribute(attribute_value(hidden_state_attr_tensor), i).unsqueeze(1))  # [B,1,H,L,D] LIST

        # i W_attr
        item_query_a = [self.transpose_for_scores_attribute(self.query_ia[i](input_tensor), i).unsqueeze(1) for i in range(self.feat_num)]

        # attr W_i
        a_query_item = [self.transpose_for_scores(self.query_ai[i](attribute_table[i].squeeze(-2))).unsqueeze(1) for i in range(self.feat_num)]

        # attr_x W_attr_x attr_y    then  calculate raw_attention    attr_x attr_y attention for r^u_v repr
        for i in range(self.feat_num):
            for j, attr_key in enumerate(attribute_key_layers):
                if i == j:
                    continue
                # [B,1,H,L,D]
                temp = self.transpose_for_scores_attribute(self.query_attr_x_y[i][j](attribute_table[i].squeeze(-2)), j).unsqueeze(1)
                attn = torch.matmul(temp, attr_key.transpose(-1, -2))
                raw_attention.append(attn)

        # attr_x W_attr_x^\prime attr_y    then  calculate raw_attention   attr_x attr_y attention for r^u_attr  repr
        for i in range(self.feat_num):
            for j, hidden_attr_key in enumerate(hidden_state_attr_key):
                if i == j:
                    continue
                temp = self.transpose_for_scores_attribute(self.query_attr_x_y[i][j](hidden_state_attr[i]).squeeze(-2), j).unsqueeze(1)
                attn = torch.matmul(temp, hidden_attr_key.transpose(-1, -2))
                raw_attention_a.append(attn)

        # attr W_pos
        a_query_pos = [self.transpose_for_scores(self.query_ap[i](attribute_table[i].squeeze(-2))).unsqueeze(1) for i in range(self.feat_num)]

        # attr W_pos ^\prime
        a_query_pos_attr = [self.transpose_for_scores(self.query_ap[i](hidden_state_attr[i].squeeze(-2))).unsqueeze(1) for i in range(self.feat_num)]

        # p W_attr
        p_query_a = [self.transpose_for_scores_attribute(self.query_pa[i](position_embedding), i).unsqueeze(1) for i in range(self.feat_num)]

        # ii attn
        item_attn_score = torch.matmul(item_query_layer, item_key_layer.transpose(-1, -2))  # [B,1,h,L,L]
        raw_attention.append(item_attn_score)

        # i attr attention
        for i in range(self.feat_num):
            raw_attention.append(torch.matmul(item_query_a[i], attribute_key_layers[i].transpose(-1, -2)))

        # i pos attention
        raw_attention.append(torch.matmul(item_query_layer, pos_key_layer.transpose(-1, -2)))

        # attr i attention
        for i in range(self.feat_num):
            raw_attention.append(torch.matmul(a_query_item[i], item_key_layer.transpose(-1, -2)))

        # attr attr self attention  for r^u_v repr
        for i in range(self.feat_num):
            raw_attention.append(torch.matmul(attribute_query_layers[i], attribute_key_layers[i].transpose(-1, -2)))

        # attr attr self attention  for r^u_att repr
        for i in range(self.feat_num):
            raw_attention_a.append(torch.matmul(hidden_state_attr_query[i], hidden_state_attr_key[i].transpose(-1, -2)))

        # att p attention  for r^u_v repr
        for i in range(self.feat_num):
            raw_attention.append(torch.matmul(a_query_pos[i], pos_key_layer.transpose(-1, -2)))

        # attr p attention  for r^u_att repr
        for i in range(self.feat_num):
            raw_attention_a.append(torch.matmul(a_query_pos_attr[i], pos_key_layer.transpose(-1, -2)))

        # p i attention
        raw_attention.append(torch.matmul(pos_query_layer, item_key_layer.transpose(-1, -2)))

        # p attr attention  for r^u_v repr
        for i in range(self.feat_num):
            raw_attention.append(torch.matmul(p_query_a[i], attribute_key_layers[i].transpose(-1, -2)))

        # p attr attention for r^u_att repr
        for i in range(self.feat_num):
            raw_attention_a.append(torch.matmul(p_query_a[i], hidden_state_attr_key[i].transpose(-1, -2)))

        # pp attention
        pp = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))
        raw_attention.append(pp)
        raw_attention_a.append(pp)

        ac_raw_attention = torch.cat(raw_attention, dim=1)     # [B, (fea_num+2)*(fea_num+2) H, L, L]
        ac_raw_attention_a = torch.cat(raw_attention_a, dim=1)   # [B, (fea_num+1)*(fea_num+1), H, L, L]

        ac_attention = torch.permute(ac_raw_attention, (0, 2, 3, 1, 4))   # [B, H, L, (fea_num+2)*(fea_num+2), L]
        ac_attention_a = torch.permute(ac_raw_attention_a, (0, 2, 3, 1, 4))     # [B, H, L, (fea_num+1)*(fea_num+1), L]

        self.fusion_w = fusion_w.unsqueeze(0).unsqueeze(0).unsqueeze(-1)    # [1, (fea_num+2)*(fea_num+2)] -> [1,1,1, (fea_num+2)*(fea_num+2),1]
        self.fusion_wa = fusion_wa.unsqueeze(1).unsqueeze(1).unsqueeze(-1)  # [fea_num, (fea_num+1)**2] -> [fea_num,1,1,(fea_num+1)**2,1]

        if self.ada_fuse != 1:
            self.fusion_w = torch.ones_like(fusion_w.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
            self.fusion_wa = torch.ones_like(fusion_wa.unsqueeze(1).unsqueeze(1).unsqueeze(-1))

        fuse_attn_attr_list = []
        if self.fusion_type == 'sum':
            ac_attention = torch.sum(ac_attention * self.fusion_w, dim=-2)
            for fuse_wa in self.fusion_wa:
                fuse_attn_attr_list.append(torch.sum(ac_attention_a * fuse_wa.unsqueeze(0), dim=-2))
        elif self.fusion_type == 'concat':
            ac_attn_shape =  ac_attention.shape
            ac_attn_shape_a = ac_attention_a.shape
            cross_num, attn_size = ac_attn_shape[-2], ac_attn_shape[-1]
            cross_num_a, attn_size_a = ac_attn_shape_a[-2], ac_attn_shape_a[-1]

            ac_attention = torch.reshape(ac_attention*self.fusion_w, ac_attn_shape[:-2] + (cross_num * attn_size,))
            ac_attention = self.fusion_layer(ac_attention)

            for fuse_wa in self.fusion_wa:
                # ac_attention_a -> [B, H, L, (fea_num+1)*(fea_num+1) * L]
                ac_attention_a_attr = torch.reshape(ac_attention_a*fuse_wa.unsqueeze(0), ac_attn_shape_a[:-2] + (cross_num_a * attn_size_a,))
                fuse_attn_attr_list.append(self.fusion_layer(ac_attention_a_attr))

        elif self.fusion_type == 'gate':
            ac_attention, _ = self.fusion_layer(ac_attention * self.fusion_w)
            for fuse_wa in self.fusion_wa:
                ac_attention_a_attr, _ = self.fusion_layer(ac_attention_a * fuse_wa.unsqueeze(0))
                fuse_attn_attr_list.append(ac_attention_a_attr)

        ac_attention = ac_attention / math.sqrt(self.attention_head_size)
        ac_attention = ac_attention + attention_mask
        attention_probs = nn.Softmax(dim=-1)(ac_attention)
        attention_probs = self.attn_dropout(attention_probs)

        attr_attn_prob_list = []
        for fuse_attn_attr in fuse_attn_attr_list:
            fuse_attn_scaled = fuse_attn_attr / math.sqrt(self.attention_head_size)
            fuse_attn_masked = fuse_attn_scaled + attention_mask
            attr_attn_prob = nn.Softmax(dim=-1)(fuse_attn_masked)
            attr_attn_prob = self.attn_dropout(attr_attn_prob)
            attr_attn_prob_list.append(attr_attn_prob)

        context_layer = torch.matmul(attention_probs, item_value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        hidden_states_attr_list = []
        for i, attr_attn_prob in enumerate(attr_attn_prob_list):
            attr_value = hidden_state_attr_value[i].squeeze(1)  # [B,H,L,D]
            context_layer_attr = torch.matmul(attr_attn_prob, attr_value)  # [b,h,l,l] * [b,h,l,d]
            context_layer_attr = context_layer_attr.permute(0, 2, 1, 3).contiguous() # [b,l,h,d]
            new_context_layer_shape = context_layer_attr.size()[:-2] + (self.attribute_all_head_size[i],)  # [B,L,d_f]
            context_layer_attr = context_layer_attr.view(*new_context_layer_shape)
            hidden_states_attr = self.dense_attr[i](context_layer_attr)
            hidden_states_attr = self.out_dropout(hidden_states_attr)
            hidden_states_attr = self.LayerNorm_attr[i](hidden_states_attr + hidden_state_attr[i].squeeze(-2))
            hidden_states_attr_list.append(hidden_states_attr)

        return hidden_states, hidden_states_attr_list


class ACGTransformerLayer(nn.Module):

    def __init__(
        self, n_heads, hidden_size,attribute_hidden_size,feat_num, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps, fusion_type, max_len, ada_fuse
    ):
        super(ACGTransformerLayer, self).__init__()
        self.multi_head_attention = ACGMultiHeadAttention(
            n_heads, hidden_size, attribute_hidden_size, feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,
            fusion_type, max_len, ada_fuse=ada_fuse
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

        self.feed_forward_attr_list = nn.ModuleList([
            copy.deepcopy(FeedForward(attribute_hidden_size[_], intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps))
        for _ in range(feat_num)])

    def forward(self, hidden_states, attribute_embed, position_embedding, hidden_states_attr, attention_mask, fusion_w, fusion_wa):
        attention_output, att_output_attr_list = self.multi_head_attention(hidden_states, attribute_embed,
                                                                            position_embedding, hidden_states_attr, attention_mask,
                                                                            fusion_w, fusion_wa)
        feedforward_output = self.feed_forward(attention_output)

        ff_output_attr_list = []
        for i, att_output_attr in enumerate(att_output_attr_list):
            ff_output_attr = self.feed_forward_attr_list[i](att_output_attr)
            ff_output_attr_list.append(ff_output_attr)

        return feedforward_output, ff_output_attr_list


class ACGTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        attribute_hidden_size=[64],
        feat_num=1,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12,
        fusion_type = 'sum',
        max_len = None,
        ada_fuse=0
    ):
        super(ACGTransformerEncoder, self).__init__()
        self.feat_num = feat_num
        self.ada_fuse = ada_fuse

        crs_w_i = torch.empty(1, (2 + feat_num) ** 2)
        crs_w_a = torch.empty(feat_num, (1 + feat_num) ** 2)

        self.fusion_type = fusion_type

        if self.ada_fuse == 1:
            self.crs_w_i = torch.nn.init.constant_(crs_w_i, val=1.0)
            self.crs_w_a = torch.nn.init.constant_(crs_w_a, val=1.0)
            self.fusion_wi = nn.Parameter(self.crs_w_i, requires_grad=True)
            self.fusion_wa = nn.Parameter(self.crs_w_a, requires_grad=True)

        layer = ACGTransformerLayer(
            n_heads, hidden_size, attribute_hidden_size, feat_num, inner_size, hidden_dropout_prob, attn_dropout_prob,
            hidden_act, layer_norm_eps, fusion_type, max_len, ada_fuse
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attribute_hidden_states, position_embedding, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_layers_attr = []

        self.soft_fusion_w = None
        self.soft_fusion_w_a = None
        if self.ada_fuse == 1:
            self.soft_fusion_w = nn.Softmax(dim=-1)(self.fusion_wi)
            self.soft_fusion_w_a = nn.Softmax(dim=-1)(self.fusion_wa)

        hidden_states_attr = attribute_hidden_states   # attribute_hidden_states  list  each element dim: [bs, L, 1, d_f]
        for layer_module in self.layer:
            hidden_states, hidden_states_attr = layer_module(hidden_states, attribute_hidden_states, position_embedding, hidden_states_attr,
                                                             attention_mask, self.soft_fusion_w, self.soft_fusion_w_a)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_encoder_layers_attr.append(hidden_states_attr)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_encoder_layers_attr.append(hidden_states_attr)
        # print(self.fusion_w.data)
        return all_encoder_layers, all_encoder_layers_attr


class MSSR_MA(SequentialRecommender):

    def __init__(self, config, dataset):
        super(MSSR_MA, self).__init__(config, dataset)
        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.attribute_hidden_size = config['attribute_hidden_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']

        self.layer_norm_eps = config['layer_norm_eps']   # a parameter of nn.LayerNorm
        self.pooling_mode = config['pooling_mode']  # the way to calculate riched item embedding matrices

        self.selected_features = config['selected_features']
        self.device = config['device']
        self.num_feature_field = len(config['selected_features'])
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.fusion_type = config['fusion_type']

        self.lamdas = config['lamdas']
        self.attribute_predictor = config['attribute_predictor']
        self.temp = config['temp']
        self.ada_fuse = config['ada_fuse']

        self.config = config

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.feature_embed_layer_list = nn.ModuleList(
            [copy.deepcopy(FeatureSeqEmbLayer(dataset, self.attribute_hidden_size[_], [self.selected_features[_]], self.pooling_mode, self.device)) for _
             in range(len(self.selected_features))])  # is a layer that initialize feature embedding

        # self.feature_embed_layer_list2 = nn.ModuleList(
        #     [copy.deepcopy(FeatureSeqEmbLayer(dataset, self.attribute_hidden_size[_], [self.selected_features[_]],
        #                                       self.pooling_mode, self.device)) for _
        #      in range(len(self.selected_features))])

        self.trm_encoder = ACGTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            attribute_hidden_size=self.attribute_hidden_size,
            feat_num=len(self.selected_features),
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            fusion_type=self.fusion_type,
            max_len=self.max_seq_length,
            ada_fuse=self.ada_fuse
        )

        self.n_attributes = {}
        for attribute in self.selected_features:
            self.n_attributes[attribute] = len(dataset.field2token_id[attribute])
        if self.attribute_predictor == 'MLP':
            self.ap = nn.Sequential(nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                                    nn.BatchNorm1d(num_features=self.hidden_size),
                                    nn.ReLU(),
                                    # final logits
                                    nn.Linear(in_features=self.hidden_size, out_features=self.n_attributes))
        elif self.attribute_predictor == 'linear':
            if self.config['aap'] == 'wi_wc_bce':
                self.api = nn.ModuleList([copy.deepcopy(nn.Linear(in_features=self.hidden_size, out_features=self.n_attributes[_], bias=True))
                    for _ in self.selected_features])
                self.apc = nn.ModuleList([copy.deepcopy(nn.Linear(in_features=self.attribute_hidden_size[b],
                                                                  out_features=self.n_attributes[a], bias=True)) for a, b in
                                         zip(self.selected_features, range(len(self.selected_features)))])
                if self.config['aap_gate'] == 1:
                    self.aap_gate_linear_list = nn.ModuleList(
                        nn.Linear(in_features=self.hidden_size+self.attribute_hidden_size[i],
                                  out_features=1, bias=False) for i in range((len(self.selected_features)))
                    )
                    self.aap_gate_drop = nn.Dropout(self.hidden_dropout_prob)
                    self.aap_gate_sigmoid = nn.Sigmoid()
            elif self.config['aap'] == 'wiwc':
                self.api = nn.ModuleList([copy.deepcopy(nn.Linear(in_features=self.hidden_size, out_features=self.n_attributes[_], bias=True))
                                          for _ in self.selected_features])
                self.apc = nn.ModuleList([copy.deepcopy(nn.Linear(in_features=self.attribute_hidden_size[b],
                                                                  out_features=self.n_attributes[a], bias=True)) for  a, b in
                                          zip(self.selected_features, range(len(self.selected_features)))])

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
            if self.config['aap'] == 'wi_wc_bce' or self.config['aap'] == 'wiwc':
                self.attribute_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            else:
                BPRLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        if self.config['ip_mode'] == 'gating':
            self.gating_linear = nn.Linear(in_features=self.hidden_size + sum(self.attribute_hidden_size),
                                           out_features=1 + len(self.selected_features), bias=False)
            self.gating_dropout = nn.Dropout(self.hidden_dropout_prob)
            self.gating_softmax = nn.Softmax(dim=-1)

        self.batch_size = config['train_batch_size']
        if self.config['ssl'] == 1:
            self.tau = config['tau']
            self.sim = config['sim']
            self.cllmd = config['cllmd']
            self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
            self.nce_fct = nn.CrossEntropyLoss()
            if self.config['cl'] == 'siwsc':
                self.wi2a = nn.ModuleList([copy.deepcopy(nn.Linear(in_features=self.hidden_size, out_features=self.attribute_hidden_size[i],
                                      bias=False)) for i in range(len(self.selected_features))])
            elif self.config['cl'] == 'idropwc':
                self.wi2a = nn.ModuleList(
                    [copy.deepcopy(nn.Linear(in_features=self.hidden_size, out_features=self.attribute_hidden_size[i],
                                             bias=False)) for i in range(len(self.selected_features))])
                self.si_drop = nn.Dropout(p=0.5)

        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ['feature_embed_layer_list']


    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()  # [B,L]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64      [B,1,1,L]
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0         #  (-2 ** 32 + 1)
        return extended_attention_mask

    def get_seq_fea_emb(self, item_seq):
        fea_emb_layer_list = self.feature_embed_layer_list
        feature_table = []
        for feature_embed_layer in fea_emb_layer_list:
            sparse_embedding, dense_embedding = feature_embed_layer(None, item_seq)
            sparse_embedding = sparse_embedding['item']  # [bs, L, 1, d_f]  # 1 = attr num
            dense_embedding = dense_embedding['item']  # None
            # concat the sparse embedding and float embedding
            if sparse_embedding is not None:
                feature_table.append(sparse_embedding)
            if dense_embedding is not None:
                feature_table.append(dense_embedding)
        # fea num * [b,l,1,d_f]    list
        return feature_table

    def get_cd_fea_emb(self):
        item_num = self.item_embedding.weight.shape[0]  # [I]
        item_set_tensor = torch.tensor([list(range(item_num))], device=self.device)  # [1,I]
        feature_table = []
        for feature_embed_layer in self.feature_embed_layer_list:
            sparse_embedding, dense_embedding = feature_embed_layer(None, item_set_tensor)
            sparse_embedding = sparse_embedding['item']  # [1, I, 1, d_f]
            dense_embedding = dense_embedding['item']  # None
            # concat the sparse embedding and float embedding
            if sparse_embedding is not None:
                feature_table.append(sparse_embedding)
            if dense_embedding is not None:
                feature_table.append(dense_embedding)

        return_list = []
        for feature in feature_table:
            return_list.append(feature.squeeze(0).squeeze(1))  #  [1, I, 1, d_f] ->  [I, d_f]
        return return_list

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim_computer='dot'):
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)  # [2B, d]

        if sim_computer == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim_computer == 'dot':
            sim = torch.mm(z, z.T) / temp            # [2B, 2B]

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels


    def item_pred_gating(self, a, b):
        # a: repr
        # b: repr list
        concat = torch.cat([a]+b, dim=-1)
        output = self.gating_dropout(self.gating_linear(concat))
        output = self.gating_softmax(output)
        return output

    def side_pred_gating(self, a, b, i):
        # a: repr
        # b: repr
        concat = torch.cat([a,b], dim=-1)
        output = self.aap_gate_drop(self.aap_gate_linear_list[i](concat))
        return self.aap_gate_sigmoid(output)

    def forward(self, item_seq, item_seq_len):
        self.item_seq_emb = self.item_embedding(item_seq)
        position_ids = []
        item_seq_np = item_seq_len.cpu().numpy()
        for i_seq_len in item_seq_np:
            pos_list = list(range(i_seq_len))
            pos_list.reverse()
            pos_list = list(np.pad(pos_list, (0, item_seq.size(1)-i_seq_len), 'constant'))
            position_ids.append(pos_list)
        position_ids = torch.tensor(position_ids, dtype=torch.long, device=item_seq.device)
        position_embedding = self.position_embedding(position_ids)

        self.fea_seq_emb = self.get_seq_fea_emb(item_seq)  # list

        input_emb = self.item_seq_emb  # [bs, L, d]
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)  # [bs, 1, L, L]
        trm_output, trm_output_attr_list = self.trm_encoder(input_emb, self.fea_seq_emb, position_embedding, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]  # the output of last layer [bs, L, d]
        output_attr_list = trm_output_attr_list[0]
        seq_output = self.gather_indexes(output, item_seq_len - 1)  # [bs, d]
        seq_output_attr_list = []
        for output_attr in output_attr_list:
            seq_output_attr = self.gather_indexes(output_attr, item_seq_len -1)
            seq_output_attr_list.append(seq_output_attr)
        # seq output [B, D]  |  seq_output_attr_list  list  element shape [B,D]
        return seq_output, seq_output_attr_list

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # seq output:  [B, d]
        # seq_output_attr_list:   element shape [B,d_f]
        seq_output, seq_output_attr_list = self.forward(item_seq, item_seq_len)
        feature_emb_table_list = self.get_cd_fea_emb()  # list  element shape [I, d]

        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight # [I, d]

            logits_ii = torch.matmul(seq_output, test_item_emb.transpose(0, 1))   # [B,I]
            logits_aa = []  #  fea num * [B,I]
            for seq_output_attr, feature_emb_table in zip(seq_output_attr_list, feature_emb_table_list):
                logits_aa.append(torch.matmul(seq_output_attr, feature_emb_table.transpose(0, 1)))

            # gating
            if self.config['ip_mode'] == 'gating':
                # seq output [B,d]   seq_output_attr_list  [B, d_f] * fea num    concat [B, d+ d_f*fea num ] -> [B, 1+fea_num]
                gating = self.item_pred_gating(seq_output, seq_output_attr_list)
                # [B, 1+fea_num, I]
                logits_all = torch.cat([logits_ii.unsqueeze(1)] + [logits_aa[i].unsqueeze(1) for i in range(len(logits_aa))], dim=1)
                logits_final = (logits_all * gating.unsqueeze(-1)).sum(1)
            else:   # direct sum
                logits_aa_sum = torch.sum(torch.cat([logits_aa[i].unsqueeze(1) for i in range(len(logits_aa))], dim=1), dim=1)
                logits_final = logits_aa_sum + logits_ii

            loss = self.loss_fct(logits_final, pos_items)

            if self.attribute_predictor!='' and self.attribute_predictor!='not':
                loss_dic = {'item_loss':loss}
                attribute_loss_sum = 0

                if self.config['aap'] == 'wi_wc_bce':
                    for i, (api, apc) in enumerate(zip(self.api, self.apc)):
                        if self.config['aap_gate'] == 1:
                            aap_gating = self.side_pred_gating(seq_output, seq_output_attr_list[i], i)
                            attribute_logits = aap_gating * api(seq_output) + (1-aap_gating) * apc(seq_output_attr_list[i])
                        else:
                            attribute_logits = api(seq_output) + apc(seq_output_attr_list[i])
                        attribute_labels = interaction.interaction[self.selected_features[i]]  # [bs, 14]
                        attribute_labels = nn.functional.one_hot(attribute_labels, num_classes=self.n_attributes[
                            self.selected_features[i]])
                        if len(attribute_labels.shape) > 2:
                            attribute_labels = attribute_labels.sum(dim=1)
                        attribute_labels = attribute_labels.float()  # [bs, 355]
                        attribute_loss = self.attribute_loss_fct(attribute_logits, attribute_labels)
                        attribute_loss = torch.mean(
                            attribute_loss[:, 1:])  # the first col of label is about zero, useless
                        loss_dic[self.selected_features[i]] = attribute_loss

                elif self.config['aap'] == 'wiwc':
                    for i, (api, apc) in enumerate(zip(self.api, self.apc)):
                        attribute_logits = api(seq_output) + apc(seq_output_attr_list[i])
                        attribute_labels = interaction.interaction[self.selected_features[i]]  # [bs, 14]
                        attribute_labels = nn.functional.one_hot(attribute_labels, num_classes=self.n_attributes[
                            self.selected_features[i]])
                        if len(attribute_labels.shape) > 2:
                            attribute_labels = attribute_labels.sum(dim=1)
                        attribute_labels = attribute_labels.float()  # [bs, 355]
                        attribute_loss = self.attribute_loss_fct(attribute_logits, attribute_labels)
                        attribute_loss = torch.mean(
                            attribute_loss[:, 1:])  # the first col of label is about zero, useless
                        loss_dic[self.selected_features[i]] = attribute_loss

                if self.num_feature_field == 1:
                    total_loss = loss + self.lamdas[0] * loss_dic['categories']
#                    print('total_loss:{}\titem_loss:{}\tattribute_{}_loss:{}'.format(total_loss, loss, self.selected_features[0], attribute_loss))
                else:
                    for i,attribute in enumerate(self.selected_features):
                        attribute_loss_sum += self.lamdas[0] * loss_dic[attribute]
                    total_loss = loss + attribute_loss_sum
                    loss_dic['total_loss'] = total_loss
                    # s = ''
                    # for key,value in loss_dic.items():
                    #     s += '{}_{:.4f}\t'.format(key,value.item())
                    # print(s)
            else:
                total_loss = loss

            if self.config['ssl'] == 1:
                if self.config['cl'] == 'siwsc':
                    for i in range(len(self.selected_features)):
                        seq_output_w = self.wi2a[i](seq_output)   # [B,d] -> [B,d_f]
                        nce_logits, nce_labels = self.info_nce(seq_output_w, seq_output_attr_list[i], temp=self.tau,
                                                               batch_size=item_seq_len.shape[0], sim_computer=self.sim)
                        clloss = self.cllmd * self.nce_fct(nce_logits, nce_labels)
                        total_loss += clloss

                elif self.config['cl'] == 'idropwc':
                    for i in range(len(self.selected_features)):
                        seq_output_drop = self.si_drop(seq_output)
                        seq_output_w = self.wi2a[i](seq_output_drop)  # [B,d] -> [B,d_f]
                        nce_logits, nce_labels = self.info_nce(seq_output_w, seq_output_attr_list[i], temp=self.tau,
                                                           batch_size=item_seq_len.shape[0], sim_computer=self.sim)
                        clloss = self.cllmd * self.nce_fct(nce_logits, nce_labels)
                        total_loss += clloss

            return total_loss


    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _ = self.forward(item_seq, item_seq_len)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, seq_output_attr_list = self.forward(item_seq, item_seq_len)
        feature_emb_table_list = self.get_cd_fea_emb()  # list  element shape [I, d]
        test_items_emb = self.item_embedding.weight

        score_ii = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B,I]
        score_aa = []  # fea num * [B,I]
        for seq_output_attr, feature_emb_table in zip(seq_output_attr_list, feature_emb_table_list):
            score_aa.append(torch.matmul(seq_output_attr, feature_emb_table.transpose(0, 1)))

        if self.config['ip_mode'] == 'gating':
            gating = self.item_pred_gating(seq_output, seq_output_attr_list)  # [B, 1+fea_num]
            score_all = torch.cat([score_ii.unsqueeze(1)] + [score_aa[i].unsqueeze(1) for i in range(len(score_aa))], dim=1)
            scores = (score_all * gating.unsqueeze(-1)).sum(1)
        else:
            score_aa_sum = torch.sum(torch.cat([score_aa[i].unsqueeze(1) for i in range(len(score_aa))], dim=1), dim=1)
            scores = score_aa_sum + score_ii

        return scores
