
import time
import os
from torch.utils import data
import argparse
import json
import numpy as np
import sys
import models
import models.backbone
import models.caca_encoder_cross
import models.cda
import models.ia_cda
import models.segformer
import models.transformer_decoder_inception
from utils.LEVIR_CC_dataset import LEVIR_CC_Dataset
from utils.Dubai_CC_dataset import Dubai_CC_Dataset
from utils.WHU_CDC_dataset import WHU_CDC_Dataset
from utils.utils import *
from models.transformer_decoder_inception import FullTransformerDecoder
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Checkpoints created before the public-code cleanup store these module names.
sys.modules.setdefault("model_ablation", models)
sys.modules.setdefault("model_ablation.backbone", models.backbone)
sys.modules.setdefault("model_ablation.caca_encoder_cross", models.caca_encoder_cross)
sys.modules.setdefault(
    "model_ablation.transformer_decoder_inception",
    models.transformer_decoder_inception,
)
sys.modules.setdefault("modelsnew", models)
sys.modules.setdefault("modelsnew.backbone", models.backbone)
sys.modules.setdefault("modelsnew.caca_encoder_cross", models.caca_encoder_cross)
sys.modules.setdefault("modelsnew.cda", models.cda)
sys.modules.setdefault("modelsnew.ia_cda", models.ia_cda)
sys.modules.setdefault("modelsnew.segformer", models.segformer)
sys.modules.setdefault(
    "modelsnew.transformer_decoder_inception",
    models.transformer_decoder_inception,
)


def decode_caption(generator, feat1, feat2, args):
    """Run the selected decoding method without changing prediction logic."""
    if args.decode_method == 'beam':
        if hasattr(generator, 'sample_beam'):
            return generator.sample_beam(
                feat1,
                feat2,
                beam_size=args.beam_size,
                length_penalty=args.length_penalty,
            )

        return FullTransformerDecoder.sample_beam(
            generator,
            feat1,
            feat2,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
        )

    return generator.sample(feat1, feat2)


def print_inference_statistics(total_times, decode_times, token_counts):
    """Print synchronized, paper-ready latency statistics."""
    total_times = np.asarray(total_times, dtype=np.float64)
    decode_times = np.asarray(decode_times, dtype=np.float64)
    token_counts = np.asarray(token_counts, dtype=np.int64)

    if total_times.size == 0:
        print("Inference efficiency: no timed samples.")
        return

    total_std = total_times.std(ddof=1) if total_times.size > 1 else 0.0
    decode_std = decode_times.std(ddof=1) if decode_times.size > 1 else 0.0
    total_tokens = int(token_counts.sum())
    decode_ms_per_token = (
        decode_times.sum() * 1000.0 / total_tokens
        if total_tokens > 0
        else float('nan')
    )

    print("\nInference efficiency (data loading, transfer, text conversion, and metrics excluded):")
    print("Samples: {}".format(total_times.size))
    print("Mean total latency: {:.3f} ms/sample".format(total_times.mean() * 1000.0))
    print("Std total latency: {:.3f} ms/sample".format(total_std * 1000.0))
    print("Median total latency: {:.3f} ms/sample".format(np.median(total_times) * 1000.0))
    print("P95 total latency: {:.3f} ms/sample".format(np.percentile(total_times, 95) * 1000.0))
    print("Mean decoding latency: {:.3f} ms/sample".format(decode_times.mean() * 1000.0))
    print("Std decoding latency: {:.3f} ms/sample".format(decode_std * 1000.0))
    print("Average generated length: {:.3f} tokens/sample".format(token_counts.mean()))
    print("Decoding latency: {:.3f} ms/token".format(decode_ms_per_token))
    print("Throughput: {:.3f} samples/s".format(total_times.size / total_times.sum()))

def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    
    try:
        checkpoint = torch.load(args.checkpoint, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint)

    extractor = checkpoint['extractor']
    encoder = checkpoint['encoder']
    generator = checkpoint['generator']

    print("dataset: {}".format(args.data_name))
    print("extractor: segformer-mit_b1")
    print("encoder: {}".format(encoder.__class__.__name__))
    print("generator: {}".format('Transformer'))
    print("decode method: {}".format(args.decode_method))
    if args.decode_method == 'beam':
        print("beam size: {}".format(args.beam_size))
        print("length penalty: {}".format(args.length_penalty))
        if args.test_batchsize != 1:
            raise ValueError("Beam search requires --test_batchsize 1")

    extractor.eval()
    extractor = extractor.cuda()
    encoder.eval()
    encoder = encoder.cuda()
    generator.eval()
    generator = generator.cuda()

    if args.data_name == 'LEVIR_CC':
        nochange_list = ["the scene is the same as before ", "there is no difference ",
                         "the two scenes seem identical ", "no change has occurred ",
                         "almost nothing has changed "]
        test_loader = data.DataLoader(
            LEVIR_CC_Dataset(data_path=args.data_folder, list_path=args.list_path, split='test', 
                             max_length=args.max_length, allow_unknown=args.allow_unknown, 
                             vocab_file=args.vocab_file, token_path=args.token_folder),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.data_name == 'Dubai_CC':
        nochange_list = ["Nothing has changed ", "There is no difference ", "All remained the same "
                         "Everything remains the same ", "Nothing has changed in this area ",
                         "Nothing changed in this area ", "No changes in this area ", "No change was made ",
                         "There is no change to mention ", "No changes to mention ", "No changed to mention ",
                         "No difference in this area ", "No change to mention ", "No change was made ",
                         "No change occurred in this area ", "The area appears the same "]
        test_loader = data.DataLoader(
            Dubai_CC_Dataset(data_path=args.data_folder, list_path=args.list_path, split='test', 
                             max_length=args.max_length, allow_unknown=args.allow_unknown, 
                             vocab_file=args.vocab_file, token_path=args.token_folder),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.data_name == 'WHU_CDC':
        nochange_list = ["the scene is the same as before ", "there is no difference ",
                         "the two scenes seem identical ", "no change has occurred ",
                         "almost nothing has changed "]
        test_loader = data.DataLoader(
            WHU_CDC_Dataset(data_path=args.data_folder, list_path=args.list_path, split='test', 
                            max_length=args.max_length, allow_unknown=args.allow_unknown, 
                            vocab_file=args.vocab_file, token_path=args.token_folder),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    l_resize1 = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    l_resize2 = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    
    if args.warmup_iters > 0:
        warmup_batch = next(iter(test_loader))
        warm_imgA, warm_imgB = warmup_batch[0], warmup_batch[1]
        warm_imgA = warm_imgA.cuda(non_blocking=True)
        warm_imgB = warm_imgB.cuda(non_blocking=True)
        if args.data_name == 'Dubai_CC':
            warm_imgA = l_resize1(warm_imgA)
            warm_imgB = l_resize2(warm_imgB)

        with torch.inference_mode():
            for _ in range(args.warmup_iters):
                warm_feat1, warm_feat2 = extractor(warm_imgA, warm_imgB)
                warm_feat1, warm_feat2 = encoder(warm_feat1, warm_feat2)
                decode_caption(generator, warm_feat1, warm_feat2, args)
        torch.cuda.synchronize()

        del warm_imgA, warm_imgB, warm_feat1, warm_feat2, warmup_batch

    test_wall_start = time.perf_counter()
    total_inference_times = []
    decode_times = []
    generated_token_counts = []
    references = list()  
    hypotheses = list()  
    change_references = list()
    change_hypotheses = list()
    nochange_references = list()
    nochange_hypotheses = list()
    change_acc = 0
    nochange_acc = 0

    with torch.inference_mode():
        for ind, (imgA, imgB, token_all, token_all_len, _, _, _) in tqdm(enumerate(test_loader)):

            imgA = imgA.cuda(non_blocking=True)
            imgB = imgB.cuda(non_blocking=True)
            if args.data_name == 'Dubai_CC':
                imgA = l_resize1(imgA)
                imgB = l_resize2(imgB)
            token_all = token_all.squeeze(0).cuda(non_blocking=True)
            '''
            imageA_resized = F.interpolate(imgA, size=(224, 224), mode='bilinear', align_corners=True)
            imageB_resized = F.interpolate(imgB, size=(224, 224), mode='bilinear', align_corners=True)
            '''
            torch.cuda.synchronize()
            total_start = time.perf_counter()

            feat1, feat2 = extractor(imgA, imgB)
            feat1, feat2 = encoder(feat1, feat2)
            torch.cuda.synchronize()
            decode_start = time.perf_counter()

            if args.decode_method == 'beam':
                if hasattr(generator, 'sample_beam'):
                    seq, _ = generator.sample_beam(
                        feat1,
                        feat2,
                        beam_size=args.beam_size,
                        length_penalty=args.length_penalty,
                    )
                else:
                    seq, _ = FullTransformerDecoder.sample_beam(
                        generator,
                        feat1,
                        feat2,
                        beam_size=args.beam_size,
                        length_penalty=args.length_penalty,
                    )
            else:
                seq, _ = generator.sample(feat1, feat2)

            torch.cuda.synchronize()
            total_end = time.perf_counter()
            total_inference_times.append(total_end - total_start)
            decode_times.append(total_end - decode_start)

            img_token = token_all.tolist()

            img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'],
                                                                       word_vocab['<NULL>']}], img_token))

            references.append(img_tokens)

            pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
            generated_token_counts.append(len(pred_seq))
            hypotheses.append(pred_seq)
            assert len(references) == len(hypotheses)

            pred_caption = ""
            ref_caption = ""
            for i in pred_seq:
                pred_caption += (list(word_vocab.keys())[i]) + " "
            ref_caption = ""
            for i in img_tokens[0]:
                ref_caption += (list(word_vocab.keys())[i]) + " "
            ref_captions = ""
            for i in img_tokens:
                for j in i:
                    ref_captions += (list(word_vocab.keys())[j]) + " "
                ref_captions += ".    "

            if ref_caption in nochange_list:
                nochange_references.append(img_tokens)
                nochange_hypotheses.append(pred_seq)
                if pred_caption in nochange_list:
                    nochange_acc = nochange_acc + 1
            else:
                change_references.append(img_tokens)
                change_hypotheses.append(pred_seq)
                if pred_caption not in nochange_list:
                    change_acc = change_acc + 1

        test_wall_time = time.perf_counter() - test_wall_start

        print_inference_statistics(
            total_inference_times,
            decode_times,
            generated_token_counts,
        )
        print("End-to-end test wall time: {:.3f} s".format(test_wall_time))

        print('len(nochange_references):', len(nochange_references))
        print('len(change_references):', len(change_references))

        if len(nochange_references) > 0:
            print('nochange_metric:')
            nochange_metric = get_eval_score(nochange_references, nochange_hypotheses)
            Bleu_1 = nochange_metric['Bleu_1']
            Bleu_2 = nochange_metric['Bleu_2']
            Bleu_3 = nochange_metric['Bleu_3']
            Bleu_4 = nochange_metric['Bleu_4']
            Meteor = nochange_metric['METEOR']
            Rouge = nochange_metric['ROUGE_L']
            Cider = nochange_metric['CIDEr']
            print('BLEU-1: {0:.4f}\t' 'BLEU-2: {1:.4f}\t' 'BLEU-3: {2:.4f}\t'
                  'BLEU-4: {3:.4f}\t' 'Meteor: {4:.4f}\t' 'Rouge: {5:.4f}\t' 'Cider: {6:.4f}\t'
                  .format(Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider))
            print("nochange_acc:", nochange_acc / len(nochange_references))
        if len(change_references) > 0:
            print('change_metric:')
            change_metric = get_eval_score(change_references, change_hypotheses)
            Bleu_1 = change_metric['Bleu_1']
            Bleu_2 = change_metric['Bleu_2']
            Bleu_3 = change_metric['Bleu_3']
            Bleu_4 = change_metric['Bleu_4']
            Meteor = change_metric['METEOR']
            Rouge = change_metric['ROUGE_L']
            Cider = change_metric['CIDEr']
            print('BLEU-1: {0:.4f}\t' 'BLEU-2: {1:.4f}\t' 'BLEU-3: {2:.4f}\t'
                  'BLEU-4: {3:.4f}\t' 'Meteor: {4:.4f}\t' 'Rouge: {5:.4f}\t' 'Cider: {6:.4f}\t'
                  .format(Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider))
            print("change_acc:", change_acc / len(change_references))

        score_dict = get_eval_score(references, hypotheses)
        Bleu_1 = score_dict['Bleu_1']
        Bleu_2 = score_dict['Bleu_2']
        Bleu_3 = score_dict['Bleu_3']
        Bleu_4 = score_dict['Bleu_4']
        Meteor = score_dict['METEOR']
        Rouge = score_dict['ROUGE_L']
        Cider = score_dict['CIDEr']
        print('Testing:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.4f}\t' 'BLEU-2: {2:.4f}\t' 'BLEU-3: {3:.4f}\t'
              'BLEU-4: {4:.4f}\t' 'Meteor: {5:.4f}\t' 'Rouge: {6:.4f}\t' 'Cider: {7:.4f}\t'
              .format(test_wall_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the full ICCACA model')
    parser.add_argument('--data_folder', required=True, help='image directory')
    parser.add_argument('--list_path', required=True, help='directory containing split lists and vocab')
    parser.add_argument('--token_folder', required=True, help='token directory')
    parser.add_argument('--vocab_file', default='vocab', help='vocab file')
    parser.add_argument('--max_length', type=int, default=41, help='max length of each caption sentence')
    parser.add_argument('--allow_unknown', type=int, default=1, help='whether unknown tokens are allowed')
    parser.add_argument('--data_name', choices=['LEVIR_CC', 'Dubai_CC', 'WHU_CDC'], required=True)
    parser.add_argument('--checkpoint', required=True, help='path to a full-model checkpoint')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--workers', type=int, default=8,
                        help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--test_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--decode_method', choices=['greedy', 'beam'], default='greedy',
                        help='caption decoding method')
    parser.add_argument('--beam_size', type=int, default=3, help='beam width used by Beam Search')
    parser.add_argument('--length_penalty', type=float, default=0.6,
                        help='length penalty used to rank completed beams')
    parser.add_argument('--warmup_iters', type=int, default=20,
                        help='number of untimed GPU warm-up iterations')



    args = parser.parse_args()
    main(args)


