# -*- coding: utf-8 -*-
'''
This scripts performs kNN search on inferenced image and text features (on single-GPU) and outputs text-to-image prediction file for evaluation.
'''

import argparse
import numpy
from tqdm import tqdm
import json

import numpy as np
import torch
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-feats', 
        type=str, 
        required=True,
        help="Specify the path of image features."
    )  
    parser.add_argument(
        '--text-feats', 
        type=str, 
        required=True,
        help="Specify the path of text features."
    )      
    parser.add_argument(
        '--top-k', 
        type=int, 
        default=10,
        help="Specify the k value of top-k predictions."
    )   
    parser.add_argument(
        '--eval-batch-size', 
        type=int, 
        default=32768,
        help="Specify the image-side batch size when computing the inner products, default to 8192"
    )    
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help="Specify the output jsonl prediction filepath."
    )         
    return parser.parse_args()


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    # print(context.shape)
    # ih, iw = context.size(2), context.size(3)
    # sourceL = ih * iw

    # # --> batch x sourceL x ndf
    # context = context.view(batch_size, -1, sourceL)
    sourceL = context.size(1)
    ih = int(np.sqrt(sourceL))
    iw = ih
    # contextT = torch.transpose(context, 1, 2).contiguous()
    contextT = context

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    # print(contextT.size(), query.size())
    attn = torch.bmm(contextT, query)  # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=1)(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax(dim=1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(torch.transpose(context, 1, 2), attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)

def word_similarity_score(img_features, words_emb, cap_lens, gamma1=0.5, gamma2=0.5, gamma3=10):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """

    att_maps = []
    similarities = []
    batch_size = img_features.shape[0]

    # Get the i-th text description
    words_num = cap_lens
    # -> 1 x nef x words_num
    word = words_emb[0, :, :words_num].unsqueeze(0).contiguous()
    # -> batch_size x nef x words_num
    word = word.repeat(batch_size, 1, 1)
    # batch x nef x 17*17
    context = img_features

    """
        word(query): batch x nef x words_num
        context: batch x nef x 17 x 17
        weiContext: batch x nef x words_num
        attn: batch x words_num x 17 x 17
    """
    weiContext, attn = func_attention(word, context, gamma1)
    att_maps.append(attn[0].unsqueeze(0).contiguous())
    # --> batch_size x words_num x nef
    word = word.transpose(1, 2).contiguous()
    weiContext = weiContext.transpose(1, 2).contiguous()
    # --> batch_size*words_num x nef
    word = word.view(batch_size * words_num, -1)
    weiContext = weiContext.view(batch_size * words_num, -1)
    #
    # -->batch_size*words_num
    row_sim = cosine_similarity(word, weiContext)
    # --> batch_size x words_num
    row_sim = row_sim.view(batch_size, words_num)

    # Eq. (10)
    row_sim.mul_(gamma2).exp_()
    row_sim = row_sim.sum(dim=1, keepdim=True)
    row_sim = torch.log(row_sim)

    # --> 1 x batch_size
    # similarities(i, j): the similarity between the i-th image and the j-th text description
    similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    similarities = similarities * gamma3

    return similarities

if __name__ == "__main__":
    args = parse_args()

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    print("Begin to load image features...")
    image_ids = []
    image_feats = []
    patch_feats = []
    with open(args.image_feats, "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            image_ids.append(obj['image_id'])
            image_feats.append(obj['image_feature'])
            patch_feats.append(obj['patch_feature'])
    image_feats_array = np.array(image_feats, dtype=np.float32)
    patch_feats_array = np.array(patch_feats, dtype=np.float32)
    print("Finished loading image features.")

    print("Begin to compute top-{} predictions for texts...".format(args.top_k))
    with open(args.output, "w") as fout:
        with open(args.text_feats, "r") as fin:
            for line in tqdm(fin):
                obj = json.loads(line.strip())
                text_id = obj['text_id']
                text_feat = obj['text_feature']
                word_feat = obj['word_feature']
                text_len = obj['text_length']
                score_tuples = []
                text_feat_tensor = torch.tensor([text_feat], dtype=torch.float).cuda() # [1, feature_dim]
                word_feat_tensor = torch.tensor([word_feat], dtype=torch.float).cuda() # [1, word_num, feature_dim]
                idx = 0
                while idx < len(image_ids):
                    img_feats_tensor = torch.from_numpy(image_feats_array[idx : min(idx + args.eval_batch_size, len(image_ids))]).cuda() # [batch_size, feature_dim]
                    patch_feats_tensor = torch.from_numpy(patch_feats_array[idx : min(idx + args.eval_batch_size, len(image_ids))]).cuda() # [batch_size, patch_num, feature_dim]
                    
                    textsim_scores = text_feat_tensor @ img_feats_tensor.t() # [1, batch_size]
                    # wordsim_scores = word_similarity_score(patch_feats_tensor, word_feat_tensor, 
                    #                                 text_len, gamma1=5.0, gamma2=0.5, gamma3=10.0)

                    # for image_id, textsim_score, wordsim_score in zip(image_ids[idx : min(idx + args.eval_batch_size, len(image_ids))], textsim_scores.squeeze(0).tolist(), wordsim_scores.squeeze(0).tolist()):
                    #     score_tuples.append((image_id, textsim_score, wordsim_score))
                    for image_id, textsim_score in zip(image_ids[idx : min(idx + args.eval_batch_size, len(image_ids))], textsim_scores.squeeze(0).tolist()):
                          score_tuples.append((image_id, textsim_score))
                    idx += args.eval_batch_size
                
                top_k_predictions = sorted(score_tuples, key=lambda x:x[1], reverse=True)[:args.top_k]
                fout.write("{}\n".format(json.dumps({"text_id": text_id, "image_ids": [entry[0] for entry in top_k_predictions]})))

                # # to compute mean-score
                # textsim_scores = []
                # wordsim_scores = []
                # for i in range(image_ids):
                #     textsim_scores.append(score_tuples[i][1])
                #     wordsim_scores.append(score_tuples[i][2])
                # textsim_scores = (textsim_scores - torch.mean(textsim_scores)) / torch.std(textsim_scores)
                # wordsim_scores = (wordsim_scores - torch.mean(wordsim_scores)) / torch.std(wordsim_scores)
                # for i in range(image_ids):
                #     score_tuples[i][1] = textsim_scores[i] + wordsim_scores[i]
                # top_k_predictions = sorted(score_tuples, key=lambda x:x[1], reverse=True)[:args.top_k]
                # fout.write("{}\n".format(json.dumps({"text_id": text_id, "image_ids": [entry[0] for entry in top_k_predictions]})))

                # # or to compute mean-rank
                # top_k_predictions_global = sorted(score_tuples, key=lambda x:x[1], reverse=True)
                # rank_list = top_k_predictions_global
                # top_k_predictions_local = sorted(score_tuples, key=lambda x:x[2], reverse=True)
                # for i in range(image_ids):
                #     image_id_tmp = top_k_predictions_global[i][0]
                #     for j in range(image_ids):
                #         if image_id_tmp == top_k_predictions_local[j][0]:
                #             img_id_tmp_rank1 = j + 1
                #     rank_list[i][1] = i + 1 + img_id_tmp_rank1
                # for i in range(image_ids):
                #     top_k_predictions = sorted(rank_list, key=lambda x:x[2], reverse=True)[:args.top_k]
                # fout.write("{}\n".format(json.dumps({"text_id": text_id, "image_ids": [entry[0] for entry in top_k_predictions]})))
    
    print("Top-{} predictions are saved in {}".format(args.top_k, args.output))
    print("Done!")
