import argparse
import time
import random
import math
import torch
import torch.nn.functional as F

from model import TopGenVAE
from data import Corpus, get_iterator, PAD_TOKEN
from loss import seq_recon_loss, bow_recon_loss
from loss import total_kld


parser = argparse.ArgumentParser(description='Text VAE')
parser.add_argument('--data', type=str, default='./data/ptb',
                    help="location of the data folder")
parser.add_argument('--max_vocab', type=int, default=20000,
                    help="maximum vocabulary size for the input")
parser.add_argument('--max_length', type=int, default=200,
                    help="maximum sequence length for the input")
parser.add_argument('--embed_size', type=int, default=200,
                    help="size of the word embedding")
parser.add_argument('--label_embed_size', type=int, default=8,
                    help="size of the label embedding")
parser.add_argument('--hidden_size', type=int, default=200,
                    help="number of hidden units for RNN")
parser.add_argument('--code_size', type=int, default=32,
                    help="latent code dimension")
parser.add_argument('--num_topics', type=int, default=32,
                    help="number of topics")
parser.add_argument('--epochs', type=int, default=48,
                    help="maximum training epochs")
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help="batch size")
parser.add_argument('--dropout', type=float, default=0.2,
                    help="dropout applied to layers (0 = no dropout)")
parser.add_argument('--alpha', type=float, default=1.0,
                    help="weight of the mutual information term")
parser.add_argument('--beta', type=float, default=1.0,
                    help="weight of the total correlation term")
parser.add_argument('--gamma', type=float, default=1.0,
                    help="weight of the dimension-wise kl term")
parser.add_argument('--lr', type=float, default=1e-3,
                    help="learning rate")
parser.add_argument('--wd', type=float, default=1e-5,
                    help="weight decay used for regularization")
parser.add_argument('--epoch_size', type=int, default=2000,
                    help="number of training steps in an epoch")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed")
parser.add_argument('--kla', action='store_true',
                    help="use kl annealing")
parser.add_argument('--nocuda', action='store_true',
                    help="do not use CUDA")
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)


def evaluate(data_iter, model, pad_id):
    model.eval()
    data_iter.init_epoch()
    size = len(data_iter.data())
    seq_loss = 0.0
    bow_loss = 0.0
    kld_z = 0.0
    kld_t = 0.0
    seq_words = 0
    bow_words = 0
    for batch in data_iter:
        texts, lengths = batch.text
        batch_size = texts.size(0)
        inputs = texts[:, :-1].clone()
        targets = texts[:, 1:].clone()
        results = model(inputs, lengths-1, pad_id)
        batch_seq = seq_recon_loss(
            results.seq_outputs, targets, pad_id
        )
        batch_bow = bow_recon_loss(
            results.bow_outputs, results.bow_targets
        )
        batch_kld_z = total_kld(results.posterior_z)
        batch_kld_t = total_kld(results.posterior_t,
                                results.prior_t).to(inputs.device)
        seq_loss += batch_seq.item() / size
        bow_loss += batch_bow.item() / size        
        kld_z += batch_kld_z.item() / size
        kld_t += batch_kld_t.item() / size
        seq_words += torch.sum(lengths-1).item()
        bow_words += torch.sum(results.bow_targets)
    seq_ppl = math.exp(seq_loss * size / seq_words)
    bow_ppl = math.exp(bow_loss * size / bow_words)
    return (seq_loss, bow_loss, kld_z, kld_t,
            seq_ppl, bow_ppl)


def train(data_iter, model, pad_id, optimizer, epoch):
    model.train()
    data_iter.init_epoch()
    size = min(len(data_iter.data()), args.epoch_size * args.batch_size)
    seq_loss = 0.0
    bow_loss = 0.0
    kld_z = 0.0
    kld_t = 0.0    
    seq_words = 0
    bow_words = 0
    for i, batch in enumerate(data_iter):
        if i == args.epoch_size:
            break
        texts, lengths = batch.text
        batch_size = texts.size(0)
        inputs = texts[:, :-1].clone()
        targets = texts[:, 1:].clone()
        results = model(inputs, lengths-1, pad_id)
        batch_seq = seq_recon_loss(
            results.seq_outputs, targets, pad_id
        )
        batch_bow = bow_recon_loss(
            results.bow_outputs, results.bow_targets
        )
        batch_kld_z = total_kld(results.posterior_z)
        batch_kld_t = total_kld(results.posterior_t,
                                results.prior_t).to(inputs.device)
        
        seq_loss += batch_seq.item() / size
        bow_loss += batch_bow.item() / size        
        kld_z += batch_kld_z.item() / size
        kld_t += batch_kld_t.item() / size        
        seq_words += torch.sum(lengths-1).item()
        bow_words += torch.sum(results.bow_targets)
        kld_weight = weight_schedule(args.epoch_size * (epoch - 1) + i) if args.kla else 1.
        optimizer.zero_grad()
        kld_term = (batch_kld_z + batch_kld_t * args.beta) / batch_size
        loss = (batch_seq + batch_bow * args.beta) / batch_size + kld_weight * kld_term
        loss.backward()
        optimizer.step()
    seq_ppl = math.exp(seq_loss * size / seq_words)
    bow_ppl = math.exp(bow_loss * size / bow_words)
        
    return (seq_loss, bow_loss, kld_z, kld_t,
            seq_ppl, bow_ppl)


def train_alt(data_iter, model, pad_id, optimizer, epoch):
    model.train()
    data_iter.init_epoch()
    size = min(len(data_iter.data()), args.epoch_size * args.batch_size)
    seq_loss = 0.0
    bow_loss = 0.0
    kld_z = 0.0
    kld_t = 0.0    
    seq_words = 0
    bow_words = 0
    for i, batch in enumerate(data_iter):
        if i == args.epoch_size:
            break
        texts, lengths = batch.text
        batch_size = texts.size(0)
        inputs = texts[:, :-1].clone()
        targets = texts[:, 1:].clone()
        results = model(inputs, lengths-1, pad_id)
        batch_bow = bow_recon_loss(
            results.bow_outputs, results.bow_targets
        )
        batch_kld_t = total_kld(results.posterior_t,
                                results.prior_t).to(inputs.device)
        
        bow_loss += batch_bow.item() / size        
        kld_t += batch_kld_t.item() / size        
        bow_words += torch.sum(results.bow_targets)
        optimizer.zero_grad()
        kld_term = batch_kld_t
        loss = batch_bow + kld_term
        loss.backward()
        optimizer.step()
    data_iter.init_epoch()        
    for i, batch in enumerate(data_iter):
        if i == args.epoch_size:
            break
        texts, lengths = batch.text
        batch_size = texts.size(0)
        inputs = texts[:, :-1].clone()
        targets = texts[:, 1:].clone()
        results = model(inputs, lengths-1, pad_id)
        batch_seq = seq_recon_loss(
            results.seq_outputs, targets, pad_id
        )
        batch_kld_z = total_kld(results.posterior_z)
        
        seq_loss += batch_seq.item() / size
        kld_z += batch_kld_z.item() / size
        seq_words += torch.sum(lengths-1).item()
        kld_weight = weight_schedule(args.epoch_size * (epoch - 1) + i) if args.kla else 1.
        optimizer.zero_grad()
        kld_term = batch_kld_z
        loss = batch_seq + kld_weight * kld_term
        loss.backward()
        optimizer.step()
    seq_ppl = math.exp(seq_loss * size / seq_words)
    bow_ppl = math.exp(bow_loss * size / bow_words)
    return (seq_loss, bow_loss, kld_z, kld_t,
            seq_ppl, bow_ppl)


def interpolate(i, start, duration):
    return max(min((i - start) / duration, 1), 0)


def weight_schedule(t):
    """Scheduling of the KLD annealing weight. """
    return interpolate(t, 6000, 40000)


def get_savepath(args):
    dataset = args.data.rstrip('/').split('/')[-1]
    path = './saves/emb{0:d}.hid{1:d}.z{2:d}.t{3:d}{4}{5}.beta{6:.1f}.{7}.pt'.format(
        args.embed_size, args.hidden_size, args.code_size, args.num_topics,
        '.wd{:.0e}'.format(args.wd) if args.wd > 0 else '',
        '.kla' if args.kla else '', args.beta, dataset)
    return path


def main(args):
    print("Loading data")
    dataset = args.data.rstrip('/').split('/')[-1]
    if dataset in ['yahoo', 'yelp']:
        with_label = True
    else:
        with_label = False
    corpus = Corpus(
        args.data, max_vocab_size=args.max_vocab,
        max_length=args.max_length, with_label=with_label
    )
    pad_id = corpus.word2idx[PAD_TOKEN]
    vocab_size = len(corpus.word2idx)
    print("\ttraining data size: ", len(corpus.train))
    print("\tvocabulary size: ", vocab_size)
    print("Constructing model")
    print(args)
    device = torch.device('cpu' if args.nocuda else 'cuda')
    model = TopGenVAE(
        vocab_size, args.embed_size, args.hidden_size, args.code_size,
        args.num_topics, args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_loss = None

    train_iter = get_iterator(corpus.train, args.batch_size, True,  device)
    valid_iter = get_iterator(corpus.valid, args.batch_size, False, device)
    test_iter  = get_iterator(corpus.test,  args.batch_size, False, device)
    print("\nStart training")
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            (tr_seq_loss, tr_bow_loss, tr_kld_z, tr_kld_t,
             tr_seq_ppl, tr_bow_ppl) = train(
                 train_iter, model, pad_id, optimizer, epoch
             )
            (va_seq_loss, va_bow_loss, va_kld_z, va_kld_t,
             va_seq_ppl, va_bow_ppl) = evaluate(
                 valid_iter, model, pad_id
             )
            print('-' * 90)
            meta = "| epoch {:2d} | time {:5.2f}s ".format(epoch, time.time()-epoch_start_time)
            print(meta + "| train loss {:5.2f} {:5.2f} ({:5.2f} {:5.2f}) "
                  "| train ppl {:5.2f} {:5.2f}".format(
                      tr_seq_loss, tr_bow_loss, tr_kld_z, tr_kld_t,
                      tr_seq_ppl, tr_bow_ppl))
            print(len(meta)*' ' + "| valid loss {:5.2f} {:5.2f} ({:5.2f} {:5.2f}) "
                  "| valid ppl {:5.2f} {:5.2f}".format(
                      va_seq_loss, va_bow_loss, va_kld_z, va_kld_t,
                      va_seq_ppl, va_bow_ppl), flush=True)
            epoch_loss = va_seq_loss + va_bow_loss + va_kld_z + va_kld_t
            if best_loss is None or epoch_loss < best_loss:
                best_loss = epoch_loss
                with open(get_savepath(args), 'wb') as f:
                    torch.save(model, f)
                
    except KeyboardInterrupt:
        print('-' * 90)
        print('Exiting from training early')


    with open(get_savepath(args), 'rb') as f:
        model = torch.load(f)
    (te_seq_loss, te_bow_loss, te_kld_z, te_kld_t,
     te_seq_ppl, te_bow_ppl) = evaluate(test_iter, model, pad_id)
    print('=' * 90)
    print("| End of training | test loss {:5.2f} {:5.2f} ({:5.2f} {:5.2f}) "
          "| test ppl {:5.2f} {:5.2f}".format(
              te_seq_loss, te_bow_loss, te_kld_z, te_kld_t,
              te_seq_ppl, te_bow_ppl))
    print('=' * 90)


if __name__ == '__main__':
    main(args)
