import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.distributions import Normal, LogNormal, Dirichlet

    
class SeqToBow(nn.Module):
    """Converts sequences of tokens to bag of words representations. """
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, inputs, ignore_index):
        # inputs dim: batch_size x max_len
        bow = torch.zeros(
            (inputs.size(0), self.vocab_size),
            dtype=torch.float,
            device=inputs.device
        )
        ones = torch.ones_like(
            inputs, dtype=torch.float,
        )
        bow.scatter_add_(1, inputs, ones)
        bow[:, ignore_index] = 0
        return bow


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


class HiddenToNormal(nn.Module):
    """
    Converts hidden state from the SeqEncoder to normal 
    distribution. Calculates q(z|x).

    """
    def __init__(self, hidden_size, code_size):
        super().__init__()
        self.fcmu = nn.Linear(hidden_size * 2, code_size)
        self.fclv = nn.Linear(hidden_size * 2, code_size)
        self.bnmu = nn.BatchNorm1d(code_size)
        self.bnlv = nn.BatchNorm1d(code_size)

    def forward(self, hidden):
        # hidden size: tuple of (1 x batch_size x hidden_size) 
        h = torch.cat(hidden, dim=2).squeeze(0)
        mu = self.bnmu(self.fcmu(h))
        lv = self.bnlv(self.fclv(h))
        dist = Normal(mu, (0.5 * lv).exp())
        return dist


class HiddenToLogNormal(nn.Module):
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


class HiddenToDirichlet(nn.Module):
    """
    Converts hidden state from BowEncoder to dirichlet
    distribution. Calculates q(t|x,z) = q(t|x)

    """
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_topics)
        self.bn = nn.BatchNorm1d(num_topics)

    def forward(self, hidden):
        # hidden size: (batch_size x hidden_size)
        alphas = self.bn(self.fc(hidden)).exp().cpu()
        # Dirichlet only supports cpu backprop for now
        dist = Dirichlet(alphas)
        return dist


class CodeToLogNormal(nn.Module):
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


class CodeToDirichlet(nn.Module):
    """Calculates p(t|z). """
    def __init__(self, code_size, hidden_size, num_topics):
        super().__init__()
        self.fc1 = nn.Linear(code_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_topics)
        self.bn = nn.BatchNorm1d(num_topics)

    def forward(self, inputs):
        h1 = F.softplus(self.fc1(inputs))
        alphas = self.bn(self.fc2(h1)).exp().cpu()
        dist = Dirichlet(alphas)
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


class IdentityFuser(nn.Module):
    """
    Use only z to initialize the hidden state of SeqDecoder.

    """
    def __init__(self, code_size, hidden_size):
        super().__init__()
        self.fc = nn.Linear(code_size, hidden_size * 2)

    def forward(self, z):
        hidden = F.tanh(self.fc(z)).unsqueeze(0)
        return [x.contiguous() for x in torch.chunk(hidden, 2, 2)]


class SeqDecoder(nn.Module):
    """
    Decodes into sequences. Calculates p(x|z).

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
        self.seq2bow = SeqToBow(vocab_size)
        self.encode_seq = SeqEncoder(embed_size, hidden_size, dropout)
        self.encode_bow = BowEncoder(vocab_size, hidden_size, dropout)
        self.h2z = HiddenToNormal   (hidden_size, code_size)
        self.h2t = HiddenToDirichlet(hidden_size, num_topics)
        self.z2t = CodeToDirichlet(code_size, hidden_size, num_topics)
        self.fuse = IdentityFuser(code_size, hidden_size)
        self.decode_seq = SeqDecoder(embed_size, hidden_size, dropout)
        self.decode_bow = BowDecoder(vocab_size, num_topics, dropout)
        # output layer
        self.fcout = nn.Linear(hidden_size, vocab_size)

    def _encode_z(self, inputs, lengths):
        enc_emb = self.lookup(inputs)
        hn = self.encode_seq(enc_emb, lengths)
        posterior_z = self.h2z(hn)
        return posterior_z

    def _encode_t(self, inputs, pad_id):
        bow_targets = self.seq2bow(inputs, pad_id)
        h3 = self.encode_bow(bow_targets)
        posterior_t = self.h2t(h3)
        return posterior_t, bow_targets
    
    def forward(self, inputs, lengths, pad_id):
        posterior_z = self._encode_z(inputs, lengths)
        posterior_t, bow_targets = self._encode_t(inputs, pad_id)
        dec_emb = self.lookup(inputs)
        if self.training:
            z = posterior_z.rsample()
            t = posterior_t.rsample().to(z.device)
        else:
            z = posterior_z.mean
            t = posterior_t.mean.to(z.device)
        bow_outputs = self.decode_bow(t)
        hidden = self.fuse(z)
        outputs, _ = self.decode_seq(dec_emb, lengths, hidden)
        seq_outputs = self.fcout(outputs)
        results = Results()
        results.z = z
        results.t = t
        results.bow_targets = bow_targets
        results.seq_outputs = seq_outputs
        results.bow_outputs = bow_outputs
        results.posterior_z = posterior_z
        results.prior_t = self.z2t(z)
        results.posterior_t = posterior_t
        return results

    def generate(self, z, max_length, sos_id):
        batch_size = z.size(0)
        hidden = self.fuse(z)
        generated = torch.zeros((batch_size, max_length), dtype=torch.long, device=z.device)
        dec_inputs = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=z.device)
        for k in range(max_length):
            dec_emb = self.lookup(dec_inputs)
            outputs, hidden = self.decode_seq(dec_emb, init_hidden=hidden)
            outputs = self.fcout(outputs)
            dec_inputs = outputs.max(2)[1]
            generated[:, k] = dec_inputs[:, 0].clone()
        return generated
    
    def reconstruct(self, inputs, lengths, pad_id, max_length, sos_id,
                    fix_z=True, fix_t=True):
        posterior_z, posterior_t, _ = self._encode(inputs, lengths, pad_id)
        if fix_z:
            z = posterior_z.mean
        else:
            z = posterior_z.sample()
        if fix_t:
            t = posterior_t.mean.to(z.device)
        else:
            t = posterior_t.sample().to(z.device)
        return self.generate(z, max_length, sos_id)

    def sample(self, num_samples, max_length, sos_id, device):
        """Randomly sample latent code to sample texts. 
        Note that num_samples should not be too large. 

        """
        code_size = self.z2t.fc1.in_features
        z = torch.randn(1, num_samples, code_size, device=device)
        prior_t = self.z2t(z)
        t = prior_t.sample().to(device)
        return self.generate(z, max_length, sos_id)

    def get_topics(self, inputs, pad_id):
        posterior_t = self._encode_t(inputs, pad_id)
        t = posterior_t.mean.to(inputs.device)
        return t / t.sum(1, keepdim=True)
        
    def interpolate(self, input_pairs, length_pairs, pad_id, max_length, sos_id, num_pts=4):
        z_pairs = []
        t_pairs = []
        for inputs, lengths in zip(input_pairs, length_pairs):
            posterior_z, posterior_t, _ = self._encode(inputs, lengths)
            z = posterior_z.mean
            t = posterior_t.mean.to(z.device)
            z_pairs.append(z)
            t_pairs.append(t)
        generated = []
        for i in range(num_pts+2):
            z = _interpolate(z_pairs, i, num_pts+2)
            t = _interpolate(t_pairs, i, num_pts+2)
            generated.append(self.generate(z, max_length, sos_id))
        return generated


def _interpolate(pairs, i, n):
    x1, x2 = [x.clone() for x in pairs]
    return x1 * (n - 1 - i) / (n - 1) + x2 * i / (n - 1)
