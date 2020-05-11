from collections import defaultdict
import json
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

import music21

import torch
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader

class NESDataset(Dataset):
    def __init__(self, phase='train'):
        self.root = (Path(__file__) / '../../../../data/nesmdb_midi').resolve()
        self.num_voices = 4
        self.phase = phase

        if not self.root.exists():
            raise ValueError(
                f"The dataset could not be found in this folder: {self.root}.\n"
                "Move it to that folder or create a link if you already have it elsewhere.\n"
                "Otherwise run ./VQCPCB/datasets/init_nes_dataset.sh to download and extract it (72 MB)."
            )

        self.processed = self.root / 'processed'

        self.processed.mkdir(exist_ok=True)

        self.dict_path = self.root / 'pitches.json'
        if self.dict_path.exists():
            with open(self.dict_path, 'r') as f:
                pitches_dict = json.load(f)
        else:
            pitches_dict = {'NULL': 0}
        self.pitches_dict = defaultdict(lambda: len(self.pitches_dict.keys()), pitches_dict)

        if not (self.processed / '.done').exists():
            print(f'Preprocessing {self}...')
            t0 = time.time()
            self.preprocess_dataset()
            t1 = time.time()
            d = t1 - t0
            print('Done. Time elapsed:', '{:.0f} s.'.format(d) if d < 60 else '{:.0f} min {:.0f} s.'.format(*divmod(d, 60)))
            with open(self.processed / '.done', 'wb') as f:     # indicates that the dataset has been fully preprocessed
                f.write(bytes(0))

        self.paths = list((self.processed / phase).glob('*.npy'))

    def __getitem__(self, idx):
        path = self.paths[idx]
        # print(path)
        score = np.load(path, allow_pickle=False)

        if False:#self.phase == 'train':
            ## data augmentation

            # 1. transpose melodic voices by a random number of semitones between -6 and 5
            pitch_shift = np.random.randint(-6, 6)
            score[:,:3,0] += pitch_shift        # TODO: correct pitch -> skip rests

            # 2. adjust the speed of the piece by a random percentage between +/- 5%
            time_speed = 1 + (np.random.random()-0.5) / 10

            score[:,:,1] *= time_speed
            score[:,:,3] *= time_speed

            actual_num_voices = sum(score[0,:,0] == 0)
            if actual_num_voices > 1 and np.random.random() < 0.5:
                # 3. half of the time, remove one of the instruments from the ensemble
                score[:,np.random.randint(actual_num_voices)] = 0

                # 4. half of the time, shuffle the score-to-instrument alignment for the melodic instruments only
                melodic_voices = max(actual_num_voices, 3)
                # v1 = np.random.randint(melodic_voices)
                # v2 = (v1 + 1) % melodic_voices
                # # score[:,v1] # TODO:

        # embed pitches
        for v in range(self.num_voices):
            score[:,v,0] = [self.pitches_dict[p] for p in score[:,v,0]]

        return torch.from_numpy(score)


    def __len__(self):
        return len(self.paths)

    def __repr__(self):
        return f'<NESDataset(dir={self.processed})>'

    def __str__(self):
        return 'NES-MDB Dataset'



    # DATA LOADING

    def split(self, sequences, interval=1):
        # 30 s with batch_size = 1
        r"""Splits a sequence of events into a sequence of blocks, where each block
        contains all events for one voice during a time interval

        Args:
            sequences: list of batch_size torch.Tensors, shape (seq_length, num_voices, 4)
                batch of musical events sequences
            interval: float (default=1)
                interval of sequences

        Returns:
            torch.Tensor, shape (batch_size, num_blocks, block_size, num_voices, 4)
                sequences of blocks
            list of int
                number of blocks per sequence
        """
        num_sequences = len(sequences)

        all_blocks = []
        num_blocks_per_voice = []

        for seq in sequences:
            all_voices_blocks = [] # concatenation of blocks of all voices
            for voice in seq.transpose(0,1):
                indices = (voice[:,-1] / interval).long()
                bins, counts = torch.unique_consecutive(indices, return_counts=True)
                split_size = torch.zeros(bins.max()+2, dtype=torch.int64)
                split_size[bins] = counts
                # print('indices', voice[:,-1])
                # print('bins', bins)
                # print('counts', counts)
                # print('ssize', split_size)
                blocks = torch.split(voice, split_size.tolist())
                all_voices_blocks.extend(blocks) # len(all_voices_blocks) = sum_voices
                num_blocks_per_voice.append(len(blocks))

            all_blocks.extend(all_voices_blocks)

        per_voice_blocks = torch.split(rnn.pad_sequence(all_blocks, batch_first=True), num_blocks_per_voice)

        padded_blocks = rnn.pad_sequence(per_voice_blocks, batch_first=True)
        padded_blocks_per_voice = padded_blocks.view(num_sequences, -1, *padded_blocks.shape[1:])

        lengths = torch.tensor(num_blocks_per_voice).view(num_sequences, -1).max(dim=1).values

        return padded_blocks_per_voice.permute(0, 2, 3, 1, 4), lengths

    def split1(self, sequences, interval):
        # 3 min 30 s to load the whole training set
        batch = []
        lengths = []
        for seq in sequences:
            seq_blocks = []
            for voice in seq.transpose(0,1):
                blocks = []
                block = []
                limit = interval
                for event in voice:
                    while event[-1] >= limit:
                        blocks.append(block)
                        limit += interval
                        block = []
                    block.append(event)
                blocks.append(block) # list of blocks for a given voice of a given sequence
                seq_blocks.append(blocks) # list of list of blocks for all voices of a given sequence
            # list of num_voices tensor of variable shape (num_blocks, block_size, 4)
            batch.append(seq_blocks)
        seq_length, num_voices, d = sequences[0].shape
        batch_size = len(sequences)
        num_blocks = max(len(blocks) for seq_blocks in batch for blocks in seq_blocks)
        block_size = max(len(block) for seq_blocks in batch for blocks in seq_blocks for block in blocks)
        padded = torch.zeros(batch_size, num_blocks, block_size, num_voices, d)

        for i, seq_blocks in enumerate(batch):
            for j, blocks in enumerate(seq_blocks):
                for k, block in enumerate(blocks):
                    for l, event in enumerate(block):
                        padded[i, k, l, j, :] = event

        padded[...,-1] %= interval
        lengths = [max(len(blocks) for blocks in seq_blocks) for seq_blocks in batch]
        return padded, lengths


    def collate_fn(self, sequences):
        # print(*[seq.shape for seq in sequences])
        padded_sequences, lengths = self.split(sequences)

        packed_sequences = rnn.pack_padded_sequence(
            padded_sequences,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        return packed_sequences


    def data_loaders(self, **kwargs):
        return DataLoader(
            self,
            collate_fn=self.collate_fn,
            **kwargs
        )




    # PREPROCESSING

    def preprocess_dataset(self):
        for src_file in tqdm(sorted(list(self.root.glob('**/*.mid'))), leave=False):
            target_dir = self.processed / src_file.relative_to(self.root).parent
            target_name = src_file.stem + '.npy'
            target_file = target_dir / target_name

            # skip already preprocessed files
            if target_file.exists():
                continue

            target_dir.mkdir(exist_ok=True)

            score = music21.converter.parse(src_file)
            tensor = self.parse_score(score)
            # save the computed tensor
            np.save(target_file, tensor, allow_pickle=False)

            with open(self.dict_path, 'w') as f:
                json.dump(self.pitches_dict, f, indent=2) # NOTE: pitches_dict must be saved after each step so that preprocessing can be interrupted

# n_voices: train Counter({4: 2439, 3: 1509, 2: 450, 1: 102})
#           test  Counter({4: 175, 3: 94, 2: 89, 1: 14})
#           valid Counter({4: 207, 3: 127, 2: 59, 1: 9})

    def parse_score(self, score):
        r"""Computes the per-voice list of musical events of a midi input

        Args:
            score: music21.stream.Score

        Returns:
            torch.tensor, shape (num_events, num_voices, 4)
        """
        # get tempo
        metronome = score.flat.getElementsByClass(music21.tempo.MetronomeMark)
        try: bpm = next(metronome).getQuarterBPM()
        except StopIteration: bpm = 120

        # computes features (pitch, duration, velocity, timeshift) of notes
        events_per_voice = self.num_voices * [torch.zeros(1,4)]
        for i, part in enumerate(score.parts):
            events = [self.compute_event(n, i) for n in part.flat.notesAndRests]
            events = torch.tensor([e for event in events for e in event]) # flatten

            # rescale to seconds instead of beats
            events[:,1] *= 60 / bpm
            events[:,3] *= 60 / bpm

            events_per_voice[i] = events

        return rnn.pad_sequence(events_per_voice, padding_value=-1)




    def compute_event(self, n, i):
        """Returns the (pitch, duration, velocity, offset) of a note and fills
        pitches_dict on the fly

        Args:
            n: music21.GeneralNote
                note to extract features
            i: int
                voice index

        Returns:
            tuple
                features of the note (pitch, duration, velocity, offset)
        """
        d, o = float(n.quarterLength), float(n.offset)

        if n.isRest:
            k = 129*i + 128
            _ = self.pitches_dict[k]    # add key to dict if not exists
            return [(k, d, 0, o)]

        notes = []
        v = n.volume.velocity
        for p in n.pitches:
            k = 129*i + p.midi
            _ = self.pitches_dict[k]
            notes.append((k, d, v, o))
        return notes



























if __name__ == '__main__':
    # compute statistics about the dataset
    sequences = [
        torch.tensor([0, 0.5, 0.75, 1, 1.5, 2.2, 2.75, 5.2]).reshape(-1, 1, 1),
        torch.tensor([0.25, 1, 2, 3, 3.5]).reshape(-1, 1, 1)
    ]

    dataset = NESDataset()

    t0 = time.time()
    batch, lengths = dataset.split(sequences, 1)
    t1 = time.time()
    print(batch.squeeze())
    print(batch.shape, t1-t0)
    # raise IndexError

    from collections import Counter
    import matplotlib.pyplot as plt

    def plot_hist(counters, titles=None):
        if titles is None:
            titles = len(counters)*['']
        for i, (c, title) in enumerate(zip(counters, titles)):
            # remove outliers since it causes problems at display
            to_remove = []
            for k, v in c.items():
                if v < 5: to_remove.append(k)
            for k in to_remove: del c[k]


            plt.subplot(2, 3, i+1)
            plt.bar(c.keys(), c.values(), color='b')
            plt.yscale('log')
            plt.title(title)
        plt.show()


    loader = dataset.data_loaders(batch_size=1)

    t0 = time.time()
    for s in tqdm(loader):
        pass
        # print(s)
    t1 = time.time()
    print(t1-t0)

    raise IndexError
    c_pitch = Counter()
    c_duration = Counter()
    c_velocity = Counter()
    c_timeshift = Counter()
    c_length = Counter()
    c_voices = Counter()

    for score in tqdm(dataset):
        c_pitch += Counter(score[...,0].reshape(-1))
        c_duration += Counter(score[...,1].reshape(-1))
        c_velocity += Counter(score[...,2].reshape(-1))
        c_timeshift += Counter(score[...,3].reshape(-1))
        c_length[len(score[0])] += 1
        c_voices[sum(score[:,0,0] != 0)] += 1

    # remove null values
    c_pitch[0] = 0
    c_velocity[0] = 0


    plot_hist(
        [c_pitch, c_duration, c_velocity, c_timeshift, c_length, c_voices],
        [
            'pitch', 'duration', 'velocity', 'timeshift',
            'Number of events',
            'Number of voices'
        ]
    )
