import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.distributions import Normal, LogNormal

    
class Seq2Counts(nn.Module):
    """Converts sequences of tokens to bag of words representations. """
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, inputs, ignore_index):
        # inputs dim: batch_size x max_len
        counts = torch.zeros(
            (inputs.size(0), self.vocab_size),
            dtype=torch.float,
            device=inputs.device
        )
        ones = torch.ones_like(
            inputs, dtype=torch.float,
        )
        counts.scatter_add_(1, inputs, ones)
        counts[:, ignore_index] = 0
        return counts


class SeqEncoder(nn.Module):
    """Sequence encoder. Used to calculate q(z|x). """
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers=1, batch_first=True
        )

    def forward(self, inputs, lengths):
        inputs = pack(
            self.drop(inputs), lengths, batch_first=True
        )
        _, hn = self.rnn(inputs)
        return hn


class BowEncoder(nn.Module):
    """Bag of words encoder. Used to calculate q(t|x). """
    def __init__(self, vocab_size, hidden_size, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, inputs):
        # inputs is bow of size: (batch_size x vocab_size)
        h1 = F.softplus(self.fc1(inputs))
        h2 = F.softplus(self.fc2(h1))
        h = self.drop(h2)
        return h


class Hidden2Normal(nn.Module):
    """
    Converts hidden state from the SeqEncoder to normal 
    distribution. Calculates q(z|x).

    """
    def __init__(self, hidden_size, code_size):
        super().__init__()
        self.fcmu = nn.Linear(hidden_size * 2, code_size)
        self.fclv = nn.Linear(hidden_size * 2, code_size)
        self.bnmu = nn.BatchNorm1d(num_topics)
        self.bnlv = nn.BatchNorm1d(num_topics)

    def forward(self, hidden):
        # hidden size: tuple of (1 x batch_size x hidden_size) 
        h = torch.cat(hidden, dim=2).squeeze(0)
        mu = self.bnmu(self.fcmu(h))
        lv = self.bnlv(self.fclv(h))
        dist = Normal(mu, (0.5 * lv).exp())
        return dist


class Hidden2LogNormal(nn.Module):
    """
    Converts hidden state from BowEncoder to log-normal
    distribution. Calculates q(t|x,z) = q(t|x)

    """
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.fcmu = nn.Linear(hidden_size, num_topics)
        self.fclv = nn.Linear(hidden_size, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics)
        self.bnlv = nn.BatchNorm1d(num_topics)

    def forward(self, hidden):
        # hidden size: (batch_size x hidden_size)
        mu = self.bnmu(self.fcmu(hidden))
        lv = self.bnlv(self.fclv(hidden))
        dist = LogNormal(mu, (0.5 * lv).exp())
        return dist

    
class Code2LogNormal(nn.Module):
    """Calculates p(t|z). """
    def __init__(self, code_size, hidden_size, num_topics):
        super().__init__()
        self.fc1 = nn.Linear(code_size, hidden_size)
        self.fcmu = nn.Linear(hidden_size, num_topics)
        self.fclv = nn.Linear(hidden_size, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics)
        self.bnlv = nn.BatchNorm1d(num_topics)

    def forward(self, inputs):
        h1 = F.softplus(self.fc1(inputs))
        mu = self.bnmu(self.fcmu(h1))
        lv = self.bnlv(self.fclv(h1))
        dist = LogNormal(mu, (0.5 * lv).exp())
        return dist


class BilinearFuser(nn.Module):
    """
    Fuse z and t to initialize the hidden state for SeqDecoder.
    z and t are fused with a bilinear layer.

    """
    def __init__(self, code_size, num_topics, hidden_size):
        super().__init__()
        self.bilinear = nn.Bilinear(code_size, num_topics, hidden_size * 2)

    def forward(self, z, t):
        hidden = F.tanh(self.bilinear(z, t)).unsqueeze(0)
        return [x.contiguous() for x in torch.chunk(hidden, 2, 2)]

    
class ConcatFuser(nn.Module):
    """
    Fuse z and t to initialize the hidden state for SeqDecoder.
    z and t are fused by applying a linear layer to the concatenation. 

    """
    def __init__(self, code_size, num_topics, hidden_size):
        super().__init__()
        self.fc = nn.Linear(code_size + num_topics, hidden_size * 2)

    def forward(self, z, t):
        code = torch.cat([z, t], dim=1)
        hidden = F.tanh(self.fc(code)).unsqueeze(0)
        return [x.contiguous() for x in torch.chunk(hidden, 2, 2)]


class SeqDecoder(nn.Module):
    """
    Decodes into sequences. Calculates p(x|z,t).

    """
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers=1, batch_first=True
        )
        
    def forward(self, inputs, lengths=None, init_hidden=None):
        # inputs size: batch_size x sequence_length x embed_size
        inputs = self.drop(inputs)
        if lengths is not None:
            inputs = pack(inputs, lengths, batch_first=True)
        outputs, hidden = self.rnn(inputs, init_hidden)
        if lengths is not None:
            outputs, _ = unpack(outputs, batch_first=True)
        outputs = self.drop(outputs)
        return outputs, hidden


class BowDecoder(nn.Module):
    """
    Decodes into log-probabilities across the vocabulary. 
    Calculates p(x_{bow}|t).

    """
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.fc = nn.Linear(num_topics, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        # inputs of size (batch_size x num_topics)
        # returns log probs of each token
        inputs = self.drop(inputs)
        return F.log_softmax(self.bn(self.fc(inputs)), dim=1)


class Results:
    """Holds model results. """
    pass


class TopGenVAE(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 code_size, num_topics, dropout):
        super().__init__()
        self.lookup = nn.Embedding(vocab_size, embed_size)
        self.encode_seq = SeqEncoder(embed_size, hidden_size, dropout)
        self.encode_bow = BowEncoder(vocab_size, hidden_size, dropout)
        self.decode_seq = SeqDecoder(embed_size, hidden_size, dropout)
        self.decode_bow = BowDecoder(vocab_size, num_topics, dropout)
        self.seq2counts = Seq2Counts(vocab_size)
        self.h2norm = Hidden2Normal   (hidden_size, code_size)
        self.h2lgnm = Hidden2LogNormal(hidden_size, num_topics)
        self.fuse = ConcatFuser(code_size, num_topics, hidden_size)
        # output layer
        self.fcout = nn.Linear(hidden_size, vocab_size)

    def _encode(self, inputs, lengths, pad_id):
        enc_emb = self.lookup(inputs)
        enc_bow = self.seq2counts(inputs, pad_id)
        hn = self.encode_seq(enc_emb, lengths)
        h3 = self.encode_bow(enc_bow)
        dist_norm = self.h2norm(hn)
        dist_lgnm = self.h2lgnm(h3)
        return dist_norm, dist_lgnm, enc_bow
    
    def forward(self, inputs, lengths, pad_id):
        dist_norm, dist_lgnm, enc_bow = self._encode(inputs, lengths, pad_id)
        dec_emb = self.lookup(inputs)
        if self.training:
            z = dist_norm.rsample()
            t = dist_lgnm.rsample()
        else:
            z = dist_norm.mean
            t = dist_lgnm.mean
        log_probs = self.decode_bow(t)
        hidden = self.fuse(z, t)
        outputs, _ = self.decode_seq(dec_emb, lengths, hidden)
        outputs = self.fcout(outputs)
        results = Results()
        results.z = z
        results.t = t
        results.bow_targets = enc_bow
        results.seq_outputs = outputs
        results.bow_outputs = log_probs
        results.posterior_z = dist_norm
        results.posterior_t = dist_lgnm
        return results

    def generate(self, z, t, max_length, sos_id):
        batch_size = z.size(0)
        hidden = self.fuse(z, t)
        generated = torch.zeros((batch_size, max_length), dtype=torch.long, device=z.device)
        dec_inputs = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=z.device)
        for k in range(max_length):
            dec_emb = self.lookup(dec_inputs)
            outputs, hidden = self.decode(dec_emb, init_hidden=hidden)
            outputs = self.fcout(outputs)
            dec_inputs = outputs.max(2)[1]
            generated[:, k] = dec_inputs[:, 0].clone()
        return generated
    
    def reconstruct(self, inputs, lengths, pad_id, max_length, sos_id,
                    fix_z=True, fix_t=True):
        dist_norm, dist_lgnm, _ = self._encode(inputs, lengths, pad_id)
        if fix_z:
            z = dist_norm.mean
        else:
            z = dist_norm.sample()
        if fix_t:
            t = dist_lgnm.mean
        else:
            t = dist_lgnm.sample()
        return self.generate(z, t, max_length, sos_id)

    def sample(self, num_samples, max_length, sos_id, device):
        """Randomly sample latent code to sample texts. 
        Note that num_samples should not be too large. 

        """
        code_size = self.fuse.in1_features
        num_topics = self.fuse.in2_features
        z = torch.randn(1, num_samples, code_size, device=device)
        t = torch.randn(1, num_samples, num_topics, device=device).exp()
        return self.generate(z, t, max_length, sos_id)

    def get_topics(self, inputs, pad_id):
        enc_bow = self.seq2counts(inputs, pad_id)
        h3 = self.encode_bow(enc_bow)
        dist_lgnm = self.h2lgnm(h3)
        t = dist_lgnm.mean
        return t / t.sum(1, keepdim=True)
        
    def interpolate(self, input_pairs, length_pairs, pad_id, max_length, sos_id, num_pts=4):
        z_pairs = []
        t_pairs = []
        for inputs, lengths in zip(input_pairs, length_pairs):
            dist_norm, dist_lgnm, _ = self._encode(inputs, lengths)
            z = dist_norm.mean
            t = dist_lgnm.mean
            z_pairs.append(z)
            t_pairs.append(t)
        generated = []
        for i in range(num_pts+2):
            z = _interpolate(z_pairs, i, num_pts+2)
            t = _interpolate(t_pairs, i, num_pts+2)
            generated.append(self.generate(z, t, max_length, sos_id))
        return generated


def _interpolate(pairs, i, n):
    x1, x2 = [x.clone() for x in pairs]
    return x1 * (n - 1 - i) / (n - 1) + x2 * i / (n - 1)
