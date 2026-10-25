import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
from utils.LEVIR_CC_dataset import LEVIR_CC_Dataset
from utils.Dubai_CC_dataset import Dubai_CC_Dataset
from utils.WHU_CDC_dataset import WHU_CDC_Dataset
from utils.SECOND_CC_AUG_dataset import SECOND_CC_AUG_Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from utils.utils import get_eval_score, accuracy
from models.backbone import Extractor
from models.transformer_decoder_inception import FullTransformerDecoder
from models.caca_encoder_cross import BidirectionalCAKAEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(2)

def main(args):
    log_file = os.path.join(args.save_path, "train_log.txt")
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
    if args.data_name not in {"LEVIR_CC", "Dubai_CC", "WHU_CDC", "SECOND_CC_AUG"}:
        raise ValueError(f"Unsupported dataset: {args.data_name}")
    start_epoch = 0
    best_bleu4 = -1.0
    with open(os.path.join(args.list_path + args.vocab_file + ".json"), "r") as f:
        word_map = json.load(f)
    extractor = Extractor()
    extractor.fine_tuning(args.fine_tune)

    encoder = BidirectionalCAKAEncoder(
        num_layers=args.n_layer,
        channels=args.encoder_dim,
        hidden_dim=args.hidden_dim,
    )

    generator = FullTransformerDecoder(
        encoder_dim=args.encoder_dim,
        feature_dim=args.feature_dim,
        vocab_size=len(word_map),
        max_lengths=args.max_length,
        word_vocab=word_map,
        n_head=args.n_heads,
        n_layers=args.decoder_n_layers,
        dropout=args.dropout,
    )

    print("\n================ Model summary ================")

    print("SegFormer-MiT-B1 -> IA-CDA -> CAKA -> Inception decoder")

    print("=============================================\n")

    extractor_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    generator_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    total_params = extractor_params + encoder_params + generator_params
    print(f"Extractor parameters: {extractor_params}")
    print(f"Encoder parameters: {encoder_params}")
    print(f"Decoder parameters: {generator_params}")
    print(f"Total parameters: {total_params}")
    print("\n")
    parameter_file = os.path.join(args.save_path, "parameters.txt")
    if not os.path.exists(os.path.dirname(parameter_file)):
        os.makedirs(os.path.dirname(parameter_file))
    with open(parameter_file, "a") as f:
        f.write("\n========== Training Arguments ==========\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("========================================\n")
    
    with open(parameter_file, "a") as f:
        f.write(f"Extractor parameters: {extractor_params}\n")
        f.write(f"Encoder parameters: {encoder_params}\n")
        f.write(f"Decoder parameters: {generator_params}\n")
        f.write(f"Total parameters: {total_params}\n")
        f.flush()
    extractor_optimizer = torch.optim.Adam(extractor.parameters(), lr=args.cnn_lr)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.encoder_lr)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.decoder_lr)
    extractor = extractor.cuda()
    encoder = encoder.cuda()
    generator = generator.cuda()
    print(f"Dataset: {args.data_name}")
    print("Extractor: segformer-mit_b1")
    print("Encoder: CAKA")
    print("Decoder: Transformer")
    criterion = nn.CrossEntropyLoss().cuda()
    if args.data_name == "LEVIR_CC":
        train_dataloader = data.DataLoader(
            LEVIR_CC_Dataset(
                args.data_path, args.list_path, split="train",
                max_length=args.max_length, allow_unknown=args.allow_unknown,
                vocab_file=args.vocab_file, token_path=args.token_folder,
            ),
            batch_size=args.train_batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        print(f"Training batches: {len(train_dataloader)}")
        valid_dataloader = data.DataLoader(
            LEVIR_CC_Dataset(
                args.data_path, args.list_path, split="val",
                max_length=args.max_length, allow_unknown=args.allow_unknown,
                vocab_file=args.vocab_file, token_path=args.token_folder,
            ),
            batch_size=args.valid_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        print(f"Validation batches: {len(valid_dataloader)}")
        test_dataloader = data.DataLoader(
            LEVIR_CC_Dataset(
                args.data_path, args.list_path, split="test",
                max_length=args.max_length, allow_unknown=args.allow_unknown,
                vocab_file=args.vocab_file, token_path=args.token_folder,
            ),
            batch_size=args.valid_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        print(f"Test batches: {len(test_dataloader)}")
    elif args.data_name == "Dubai_CC":
        train_dataloader = data.DataLoader(
            Dubai_CC_Dataset(
                args.data_path, args.list_path, split="train",
                max_length=args.max_length, allow_unknown=args.allow_unknown,
                vocab_file=args.vocab_file, token_path=args.token_folder,
            ),
            batch_size=args.train_batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        print("train_dataloader length {}".format(len(train_dataloader)))
        valid_dataloader = data.DataLoader(
            Dubai_CC_Dataset(
                args.data_path, args.list_path, split="val",
                max_length=args.max_length, allow_unknown=args.allow_unknown,
                vocab_file=args.vocab_file, token_path=args.token_folder,
            ),
            batch_size=args.valid_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        print("val_dataloader length {}".format(len(valid_dataloader)))

        test_dataloader = data.DataLoader(
            Dubai_CC_Dataset(
                args.data_path,
                args.list_path,
                split="test",
                max_length=args.max_length,
                allow_unknown=args.allow_unknown,
                vocab_file=args.vocab_file,
                token_path=args.token_folder,
            ),
            batch_size=args.valid_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print("test_dataloader length {}".format(len(test_dataloader)))

    elif args.data_name == "WHU_CDC":
        train_dataloader = data.DataLoader(
            WHU_CDC_Dataset(
                args.data_path, args.list_path, split="train",
                max_length=args.max_length, allow_unknown=args.allow_unknown,
                vocab_file=args.vocab_file, token_path=args.token_folder,
            ),
            batch_size=args.train_batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        print("train_dataloader length {}".format(len(train_dataloader)))
        valid_dataloader = data.DataLoader(
            WHU_CDC_Dataset(
                args.data_path, args.list_path, split="val",
                max_length=args.max_length, allow_unknown=args.allow_unknown,
                vocab_file=args.vocab_file, token_path=args.token_folder,
            ),
            batch_size=args.valid_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        print("val_dataloader length {}".format(len(valid_dataloader)))
        
        test_dataloader = data.DataLoader(
            WHU_CDC_Dataset(
                args.data_path, args.list_path, split="test",
                max_length=args.max_length, allow_unknown=args.allow_unknown,
                vocab_file=args.vocab_file, token_path=args.token_folder,
            ),
            batch_size=args.valid_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print("test_dataloader length {}".format(len(test_dataloader)))

    elif args.data_name == "SECOND_CC_AUG":
        train_dataloader = data.DataLoader(
            SECOND_CC_AUG_Dataset(
                args.data_path,
                args.list_path,
                split="train",
                max_length=args.max_length,
                allow_unknown=args.allow_unknown,
                vocab_file=args.vocab_file,
                token_path=args.token_folder,
            ),
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print("train_dataloader length {}".format(len(train_dataloader)))

        valid_dataloader = data.DataLoader(
            SECOND_CC_AUG_Dataset(
                args.data_path,
                args.list_path,
                split="val",
                max_length=args.max_length,
                allow_unknown=args.allow_unknown,
                vocab_file=args.vocab_file,
                token_path=args.token_folder,
            ),
            batch_size=args.valid_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print("val_dataloader length {}".format(len(valid_dataloader)))

        test_dataloader = data.DataLoader(
            SECOND_CC_AUG_Dataset(
                args.data_path,
                args.list_path,
                split="test",
                max_length=args.max_length,
                allow_unknown=args.allow_unknown,
                vocab_file=args.vocab_file,
                token_path=args.token_folder,
            ),
            batch_size=args.valid_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print("test_dataloader length {}".format(len(test_dataloader)))


    extractor_lr_scheduler = torch.optim.lr_scheduler.StepLR(extractor_optimizer, step_size=5, gamma=0.5)
    encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=5, gamma=0.5)
    generator_lr_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=5, gamma=0.5)

    l_resizeA = torch.nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True)
    l_resizeB = torch.nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True)
    index_i = 0
    hist = np.zeros((args.num_epochs * len(train_dataloader), 3))
    for epoch in range(start_epoch, args.num_epochs):
        for id, (imgA, imgB, _, _, token, token_len, _) in enumerate(train_dataloader):
            start_time = time.time()
            extractor.train()
            encoder.train()
            generator.train()
            if extractor_optimizer is not None:
                extractor_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            if args.data_name == "Dubai_CC":
                imgA = l_resizeA(imgA)
                imgB = l_resizeB(imgB)
            token = token.squeeze(1).cuda()
            token_len = token_len.cuda()
            feat_A, feat_B = extractor(imgA, imgB)
            feat_A, feat_B = encoder(feat_A, feat_B)
            score, caps_sorted, decode_lengths, sort_ind = generator(feat_A, feat_B, token, token_len)
            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(score, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            loss = criterion(scores, targets)
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(encoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_value_(generator.parameters(), args.grad_clip)

                if extractor_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(extractor.parameters(), args.grad_clip)

            generator_optimizer.step()
            encoder_optimizer.step()
            if extractor_optimizer is not None:
                extractor_optimizer.step()
            hist[index_i, 0] = time.time() - start_time
            hist[index_i, 1] = loss.item()
            hist[index_i, 2] = accuracy(scores, targets, 5)
            index_i += 1
            with open(log_file, "a") as f:
                if index_i % args.print_freq == 0:
                    train_message = (
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Batch Time: {3:.3f}\t"
                        "Loss: {4:.4f}\t"
                        "Top-5 Accuracy: {5:.3f}".format(
                            epoch,
                            index_i,
                            args.num_epochs * len(train_dataloader),
                            np.mean(hist[index_i - args.print_freq : index_i - 1, 0]) * args.print_freq,
                            np.mean(hist[index_i - args.print_freq : index_i - 1, 1]),
                            np.mean(hist[index_i - args.print_freq : index_i - 1, 2]),
                        )
                    )
                    print(train_message)
                    f.write(train_message + "\n")
                    f.flush()
        if extractor is not None:
            extractor.eval()
        encoder.eval()
        generator.eval()
        val_start_time = time.time()
        references = list()
        hypotheses = list()
        with torch.no_grad():
            for id, (imgA, imgB, token_all, token_len, _, _, _) in tqdm(enumerate(valid_dataloader)):
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                if args.data_name == "Dubai_CC":
                    imgA = l_resizeA(imgA)
                    imgB = l_resizeB(imgB)
                token_all = token_all.squeeze(0).cuda()
                if extractor is not None:
                    feat_A, feat_B = extractor(imgA, imgB)
                feat_A, feat_B = encoder(feat_A, feat_B)
                sequence, _ = generator.sample(feat_A, feat_B)
                img_token = token_all.tolist()
                img_tokens = list(
                    map(
                        lambda c: [
                            w for w in c if w not in {word_map["<START>"], word_map["<END>"], word_map["<NULL>"]}
                        ],
                        img_token,
                    )
                )
                references.append(img_tokens)
                pred_sequence = [
                    w for w in sequence if w not in {word_map["<START>"], word_map["<END>"], word_map["<NULL>"]}
                ]
                hypotheses.append(pred_sequence)
                assert len(references) == len(hypotheses)
                if id % args.print_freq == 0:
                    pred_caption = ""
                    ref_caption = ""
                    for i in pred_sequence:
                        pred_caption += (list(word_map.keys())[i]) + " "
                    ref_caption = ""
                    for i in img_tokens:
                        for j in i:
                            ref_caption += (list(word_map.keys())[j]) + " "
                        ref_caption += ".    "
            val_time = time.time() - val_start_time
            score_dict = get_eval_score(references, hypotheses)
            Bleu_1 = score_dict["Bleu_1"]
            Bleu_2 = score_dict["Bleu_2"]
            Bleu_3 = score_dict["Bleu_3"]
            Bleu_4 = score_dict["Bleu_4"]
            Meteor = score_dict["METEOR"]
            Rouge_L = score_dict["ROUGE_L"]
            Cider = score_dict["CIDEr"]
            val_message = (
                "Validation:\nTime: {0:.3f}\tBLEU-1: {1:.4f}\tBLEU-2: {2:.4f}\tBLEU-3: {3:.4f}\t"
                "BLEU-4: {4:.4f}\tMeteor: {5:.4f}\tRouge: {6:.4f}\tCider: {7:.4f}".format(
                    val_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge_L, Cider
                )
            )
            print(val_message)
            with open(log_file, "a") as f:
                f.write(val_message + "\n")
                f.flush()
        test_start_time = time.time()
        test_references = []
        test_hypotheses = []
        with torch.no_grad():
            for id, (imgA, imgB, token_all, token_len, _, _, _) in tqdm(enumerate(test_dataloader)):
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                if args.data_name == "Dubai_CC":
                    imgA = l_resizeA(imgA)
                    imgB = l_resizeB(imgB)
                token_all = token_all.squeeze(0).cuda()
                feat_A, feat_B = extractor(imgA, imgB)
                feat_A, feat_B = encoder(feat_A, feat_B)
                sequence, _ = generator.sample(feat_A, feat_B)
                img_token = token_all.tolist()
                img_tokens = list(
                    map(
                        lambda c: [
                            w for w in c if w not in {word_map["<START>"], word_map["<END>"], word_map["<NULL>"]}
                        ],
                        img_token,
                    )
                )
                test_references.append(img_tokens)
                pred_sequence = [
                    w for w in sequence if w not in {word_map["<START>"], word_map["<END>"], word_map["<NULL>"]}
                ]
                test_hypotheses.append(pred_sequence)
                assert len(test_references) == len(test_hypotheses)
        test_time = time.time() - test_start_time
        test_score_dict = get_eval_score(test_references, test_hypotheses)
        test_Bleu_1 = test_score_dict["Bleu_1"]
        test_Bleu_2 = test_score_dict["Bleu_2"]
        test_Bleu_3 = test_score_dict["Bleu_3"]
        test_Bleu_4 = test_score_dict["Bleu_4"]
        test_Meteor = test_score_dict["METEOR"]
        test_Rouge_L = test_score_dict["ROUGE_L"]
        test_Cider = test_score_dict["CIDEr"]
        test_message = (
            "Testing:\nTime: {0:.3f}\tBLEU-1: {1:.4f}\tBLEU-2: {2:.4f}\tBLEU-3: {3:.4f}\t"
            "BLEU-4: {4:.4f}\tMeteor: {5:.4f}\tRouge: {6:.4f}\tCider: {7:.4f}".format(
                test_time, test_Bleu_1, test_Bleu_2, test_Bleu_3, test_Bleu_4, test_Meteor, test_Rouge_L, test_Cider
            )
        )
        print(test_message)
        with open(log_file, "a") as f:
            f.write(test_message + "\n")
            f.flush()
        generator_lr_scheduler.step()
        print("Decoder learning rate:", generator_optimizer.param_groups[0]["lr"])
        encoder_lr_scheduler.step()
        print("Encoder learning rate:", encoder_optimizer.param_groups[0]["lr"])
        if extractor_lr_scheduler is not None:
            extractor_lr_scheduler.step()
            print("Extractor learning rate:", extractor_optimizer.param_groups[0]["lr"])
        average_metric = (Bleu_4 + Meteor + Rouge_L + Cider) / 4
        if Bleu_4 > best_bleu4:
            best_bleu4 = Bleu_4
            print("Save Best Model")
            state = {"extractor": extractor, "encoder": encoder, "generator": generator}
            model_name = (
                f"{args.data_name}_"
                f"BatchSize_{args.train_batch_size}_"
                "segformer-mit_b1_full_"
                f"Epoch_{epoch}_"
                f"Bleu-4_{round(10000 * Bleu_4)}_"
                f"{round(10000 * Cider)}_"
                f"{round(10000 * average_metric)}_"
                f"{args.n_layer}_test.pth"
            )

            save_path = os.path.join(args.save_path,model_name)
            torch.save(state,save_path)
            print("Best model saved:")
            print(save_path)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="DGAT for RSICC")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device index")
    parser.add_argument("--fine_tune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n_layer", type=int, default=1, help="number of encoder layers")
    parser.add_argument("--decoder_n_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("--vocab_file", type=str, default="vocab", help="vocabulary filename without extension")
    parser.add_argument(
        "--encoder_dim",
        type=int,
        default=512,
        help="SegFormer-MiT-B1 output dimension",
    )
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--data_name", choices=["LEVIR_CC", "Dubai_CC", "WHU_CDC", "SECOND_CC_AUG"], required=True)
    parser.add_argument("--data_path", required=True, help="image directory")
    parser.add_argument("--list_path", required=True, help="directory containing split lists and vocab")
    parser.add_argument("--token_folder", required=True, help="token directory")
    parser.add_argument("--max_length", type=int, default=26, help="maximum caption length")
    parser.add_argument("--allow_unknown", type=int, default=1, help="allow unknown tokens")
    parser.add_argument("--train_batch_size", type=int, default=32, help="training batch size")
    

    parser.add_argument("--valid_batch_size", type=int, default=1, help="validation batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="data-loader workers")
    parser.add_argument("--cnn_lr", type=float, default=1e-4, help="extractor learning rate")
    parser.add_argument("--encoder_lr", type=float, default=1e-4, help="encoder learning rate")
    parser.add_argument("--decoder_lr", type=float, default=1e-4, help="decoder learning rate")
    parser.add_argument("--num_epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--grad_clip", type=float, default=None, help="gradient clipping value")
    parser.add_argument("--print_freq", type=int, default=100, help="training log interval")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--save_path", type=str, default="./checkpoints/", help="checkpoint directory")
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    main(args)

