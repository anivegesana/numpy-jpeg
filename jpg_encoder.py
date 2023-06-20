SHOW_PLOTS = False
import time; tqq = time.time()

import itertools

from collections import Counter
from collections.abc import Iterable
from struct import pack

from bitarray import bitarray
import bitarray.util as bitarray_util
import numpy as np
if SHOW_PLOTS: import matplotlib.pyplot as plt

from scipy.fft import dctn
from numpy.lib.stride_tricks import as_strided
from numpy.typing import NDArray


####(a)####
with open('silicon_valley.ppm', 'rb') as f:
    words = []
    while len(words) < 4: words.extend(f.readline().split())
    _, w, h, _ = words
    # _, w, h, _ = f.readline().split()
    w, h = int(w), int(h)
    img = np.fromfile(f, dtype=np.uint8)

rgb = img.reshape(h, w, 3).transpose(1, 0, 2)

if SHOW_PLOTS:
    plt.imshow(rgb.transpose(1, 0, 2).astype(np.uint8))
    plt.show()

####(b)####
COLOR_COEFF = np.array([[ 0.299   ,  0.587   ,  0.114   ],
                        [-0.168736, -0.331264,  0.5     ],
                        [ 0.5     , -0.418688, -0.081312]]).T
COLOR_INTRCEPT = np.array([[[0, 128, 128]]])
ICOLOR_COEFF = np.array([[ 1.0,  0.0    ,  1.402   ],
                         [ 1.0, -0.34414, -0.71414 ],
                         [ 1.0,  1.772  ,  0.0     ]]).T

ycbcr = np.clip(rgb @ COLOR_COEFF + COLOR_INTRCEPT, 0, 255)

if SHOW_PLOTS:
    plt.subplot(221)
    plt.title('Luminance')
    plt.imshow(ycbcr.transpose(1, 0, 2).astype(np.uint8)[..., 0])
    plt.subplot(222)
    plt.title('Chrominance Blue')
    plt.imshow(ycbcr.transpose(1, 0, 2).astype(np.uint8)[..., 1])
    plt.subplot(223)
    plt.title('Chrominance Red')
    plt.imshow(ycbcr.transpose(1, 0, 2).astype(np.uint8)[..., 2])
    plt.show()

####(c)####
def downsample(channels, factor=20):
    ycbcr_chr = ycbcr.copy()
    ycbcr_chr[channels] = (ycbcr_chr[channels] / factor).round() * factor
    rgb_chr = (ycbcr_chr - COLOR_INTRCEPT) @ ICOLOR_COEFF
    rgb_chr = np.clip(rgb_chr, 0, 255)
    return rgb_chr

rgb_lum = downsample(slice(None, 1))
rgb_chr = downsample(slice(1, None))


if SHOW_PLOTS:
    plt.subplot(221)
    plt.title('Original')
    plt.imshow(rgb.transpose(1, 0, 2).astype(np.uint8))
    plt.subplot(222)
    plt.title('Luminance Perturbed')
    plt.imshow(rgb_lum.transpose(1, 0, 2).astype(np.uint8))
    plt.subplot(223)
    plt.title('Chrominance Perturbed')
    plt.imshow(rgb_chr.transpose(1, 0, 2).astype(np.uint8))
    plt.show()


####(d)####
# pad to multiple of 8
ycbcr -= 128
new_w = ((w + 7) >> 3) << 3
new_h = ((h + 7) >> 3) << 3
lpad, tpad = (new_w - w), (new_h - h)
ycbcr = np.pad(
    ycbcr,
    ((lpad, 0), (tpad, 0), (0, 0)),
    'edge'
)

# split into 8x8 patches
sw, sh, sc = ycbcr.strides
patches = as_strided(
    ycbcr,
    (new_w >> 3, new_h >> 3, 3, 8, 8),
    (sw << 3, sh << 3, sc, sw, sh)
)

# Minimum Coded Units (MCU)
mcus = dctn(patches, axes=(-2, -1), norm='ortho', overwrite_x=True) # patches is connected to ycbcr

# TODO: Here is the standard quantization matrix for you to play around with
# for question (f). Vary a `quality` parameter that controls how much we
# compress the image.

quality = 100

quality = np.clip(quality, 1, 100, dtype=float)
qualityScale = 5000 / quality if quality < 50 else 200 - quality * 2

Q_LUM = np.array([[  16,  11,  10,  16,  24,  40,  51,  61],
                  [  12,  12,  14,  19,  26,  58,  60,  55],
                  [  14,  13,  16,  24,  40,  57,  69,  56],
                  [  14,  17,  22,  29,  51,  87,  80,  62],
                  [  18,  22,  37,  56,  68, 109, 103,  77],
                  [  24,  35,  55,  64,  81, 104, 113,  92],
                  [  49,  64,  78,  87, 103, 121, 120, 101],
                  [  72,  92,  95,  98, 112, 100, 103,  99]], dtype=np.uint8)

Q_CHROM = np.array([[  17,  18,  24,  47,  99,  99,  99,  99],
                    [  18,  21,  26,  66,  99,  99,  99,  99],
                    [  24,  26,  56,  99,  99,  99,  99,  99],
                    [  47,  66,  99,  99,  99,  99,  99,  99],
                    [  99,  99,  99,  99,  99,  99,  99,  99],
                    [  99,  99,  99,  99,  99,  99,  99,  99],
                    [  99,  99,  99,  99,  99,  99,  99,  99],
                    [  99,  99,  99,  99,  99,  99,  99,  99]], dtype=np.uint8)

Q_LUM = np.clip(np.floor((Q_LUM * qualityScale + 50) / 100), 1, 225).astype(np.uint8)
Q_CHROM = np.clip(np.floor((Q_CHROM * qualityScale + 50) / 100), 1, 225).astype(np.uint8)

Q = np.stack([Q_LUM, Q_CHROM, Q_CHROM])

# Or if that is a little confusing, use minimal quality loss for part (d).
# This is the same as `quality` = 100 in the above code.
# Q = np.ones((3, 8, 8), dtype=np.uint8)
# Q_LUM = Q[0]
# Q_CHROM = Q[1]


mcus_quant = (mcus / Q).round().astype(np.int16)

if SHOW_PLOTS:
    dc_val = mcus_quant[..., 0, 0].transpose(1, 0, 2)
    plt.subplot(221)
    plt.title('DC Luminance')
    plt.imshow(dc_val[..., 0])
    plt.subplot(222)
    plt.title('DC Chrominance Blue')
    plt.imshow(dc_val[..., 1])
    plt.subplot(223)
    plt.title('DC Chrominance Red')
    plt.imshow(dc_val[..., 2])
    plt.show()



####(e)####

norm_order = np.arange(64).reshape(8, 8)
l1_norm = sum(np.meshgrid(np.arange(8), np.arange(8)))
diag_order = (l1_norm + (np.arange(8) / 10)[:, np.newaxis]).flatten().argsort()
ZIGZAG_ORDER = np.where(l1_norm & 1, norm_order, norm_order.T).flatten()[diag_order]

# TODO: Change the height and width of the image to match what you read in (a)
h = h
w = w

# TODO: Change the `mcus_quant` variable to match the variables in the code you wrote thus far.
mcus_quant_zz = mcus_quant.transpose(1, 0, 2, 4, 3).reshape(-1, *mcus_quant.shape[2:-2], 64)[..., ZIGZAG_ORDER]

dc = mcus_quant_zz[..., 0]
ac = mcus_quant_zz[..., 1:]

class JPEGHuff(dict):
    __slots__ = ('bits', 'huffval')
    @classmethod
    def for_encoding(cls, symbols):
        dummy_entry = 300 #None

        counts = Counter(symbols)
        counts[dummy_entry] = 0

        # TODO: Add code to create your canoncial Huffman table with the frequencies in counts.
        # The `300` symbol just forces the code `111...11` not to be taken.
        # This is a requirement from the JPEG standard and just makes it easier to detect
        # corrupted images.
        # It is not used in encoding and you can set it to whatever you want as long
        # as it isn't a valid symbol. If you are stumped on Q1, the implementation in
        # the `bitarray` can be used for this part.

        hufftable = bitarray_util.canonical_huffman(counts)[0]

        # Drop the dummy entry.
        del hufftable[dummy_entry]

        # Calculate the number of nodes on each layer of the Huffman tree.
        # https://en.wikibooks.org/wiki/JPEG_-_Idea_and_Practice/The_Huffman_coding
        lens = [len(code) for code in hufftable.values()]

        # Drop index 0 because it is not possible.
        bits = np.bincount(lens, minlength=17)[1:].astype(np.uint8)

        # Get the symbols sorted in the order of Huffman code
        sorted_symbols = sorted(hufftable, key=hufftable.get)

        # Convert symbols to ndarray. You may need to change this depending
        # on how you stored your AC symbols.
        huffval = np.array(sorted_symbols, dtype=np.uint8)

        # The maximum depth allowed by JPEG for its Huffman tables is 16 bits. If
        # it is deeper than 16 bits, we need to rearrange the tree and make it more
        # suboptimal, but so that no code is longer than 16 bits. For this reason,
        # the Huffman table may be pregenerated using statistical data and fixed
        # for the implementation. You can look in the JPEG standard to find these
        # standardized Huffman tables.
        if bits.size > 16:
            # Add dummy node for simplicity.
            bits[-1] += 1

            # Remove nodes until the tree is depth limited. Remember our debt to
            # society.
            debt = 0
            for i in reversed(range(16, len(bits))):
                assert bits[i] % 2 == 0
                amount = bits[i] // 2
                bits[i] = 0
                bits[i - 1] += amount
                debt += amount
            bits = bits[:16]

            # Calculate all places that we can put nodes and how much room there
            # is in each bin.
            budget = np.cumsum((bits * ((1 << np.arange(16)[::-1]) >> 1))[::-1])
            loc = -np.searchsorted(budget, debt) - 1
            budget = budget[::-1]
            first_amount = debt - budget[loc+1]

            # Put removed leaves in locations that are close to the cut off based
            # on bin counts
            bits[loc] -= first_amount
            bits[loc+1] += 2 * first_amount
            loc += 1
            for loc in range(loc, -1):
                bits[loc+1] += 2 * bits[loc]
                bits[loc] = 0

            # Remove dummy node
            bits[-1] -= 1

            # Recalculate Huffman table.
            return cls.for_decoding(bits, huffval)

        self = cls(hufftable)
        self.bits = bits
        self.huffval = huffval
        return self

    @classmethod
    def for_decoding(cls, bits, huffval):
        # Load the canonical Huffman table into memory.
        self = cls()
        self.bits = bits
        self.huffval = huffval
        code = None
        for k, v in zip(huffval, np.repeat(np.arange(1, 17), bits)):
            if code is None:
                code = 0
            else:
                code = (code + 1) << (v - code_len)
            code_len = v
            self[int(k)] = bitarray(np.binary_repr(code, code_len))
        return self


def onescomp(i):
    width = i.bit_length()
    if i < 0:
        i = (i & ((1 << width) - 1)) - 1
    elif i == 0: return ''
    # assert i >= 0
    # assert len(bin(i)[2:].zfill(width)) == width
    return bin(i)[2:].zfill(width)


def rle(arr: NDArray[np.int16]):
    symbols: list[int] = [] # tuple[int, int]
    values: list[int] = []
    ptrs: list[int] = []
    repeat = itertools.repeat
    for bc in range(arr.shape[0]):
        x = arr[bc, :]
        last_ptr = len(symbols)
        nz = np.flatnonzero(x)
        runs = np.diff(nz, prepend=[-1]) - 1
        for idx, run_length in zip(nz, runs):
            # Runs of zeros that are 16 or longer need to be broken up into smaller runs.
            fills, run_length = divmod(run_length, 16)
            if fills != 0:
                symbols.extend(repeat(0xf0, fills))
                values.extend(repeat(0, fills))

            # Nonzero elements are stored
            elem = int(x[idx])
            symbols.append((int(run_length) << 4) | elem.bit_length()) # (run_length, size)
            values.append(elem)
        if x[-1] == 0:
            symbols.append(0) # (0, 0)
            values.append(0)
        ptrs.append(last_ptr)
    ptrs.append(len(symbols))
    return symbols, values, ptrs


# dc_diff[0] = dc[0]; dc_diff[n] = dc[n] - dc[n-1]
dc_diff = np.diff(dc, prepend=[[0] * 3], axis=0)

dcy_symbols = [int(elem).bit_length() for elem in dc_diff[:, 0].flat]
dcc_symbols = [int(elem).bit_length() for elem in dc_diff[:, 1:].flat]

acy_symbols, acy_values, acy_ptrs = rle(ac[:, 0, :])
acc_symbols, acc_values, acc_ptrs = rle(ac[:, 1:, :].reshape(-1, 63))

DCY_HUFF = JPEGHuff.for_encoding(dcy_symbols)
DCC_HUFF = JPEGHuff.for_encoding(dcc_symbols)
ACY_HUFF = JPEGHuff.for_encoding(acy_symbols)
ACC_HUFF = JPEGHuff.for_encoding(acc_symbols)


# The header

APP0 = bytes.fromhex(
    'FFD8'          # SOI (Start of Image)

    'FFE0'          # APP0 segment
    '0010'          # Length of segment
    '4A46494600'    # 'JFIF' in ASCII
    '0101'          # Version 1.01
    '00 0001 0001'  # Pixel density
    '00 00'         # Thumbnail size
    ''              # Thumbnail. Not present
)

def encode_dqt(q, ischrom):
    return pack('4s?64s',
        b'\xFF\xDB' # DQT (Define Quantization Table) segment
        b'\x00\x43',# Length of segment (always 67)
        ischrom,    # 0 - Luminance, 1 - Chrominance
        bytes(q)    # Quantization table
    )

DQT_Y = encode_dqt(Q_LUM, False)
DQT_C = encode_dqt(Q_CHROM, True)

SOF0 = pack('>5s2H10s',
    bytes.fromhex(
        'FFC0'      # SOF0 (Start Of Frame Baseline DCT) segment
        '0011'      # Length of segment (11 for greyscale, 17 [0x11] for color)
        '08'        # The number 8 for no apparent reason
    ),
    h, w,           # Height and width
    bytes.fromhex(
        '03'        # Number of channels (3 for color)
        '01 11 00'  # Choose quantization tables for the different channels
        '02 11 01'  # https://web.archive.org/web/20120403212223/http://class.ee.iastate.edu/ee528/Reading%20material/JPEG_File_Format.pdf
        '03 11 01'
    )
)


def encode_huff(hufftable, isac, ischrom):
    bits = hufftable.bits
    huffval = hufftable.huffval

    assert bits.size == 16
    assert bits.sum() == huffval.size

    return pack(f'>B16s{len(huffval)}s',
        (isac << 4) |       # 0 - DC, 1 - AC
        ischrom,            # 0 - Luminance, 1 - Chrominance
        bytes(bits),        # the list BITS (16 bytes)
        bytes(huffval)      # the list HUFFVAL (nhv bytes)
    )

DHT_DCY = encode_huff(DCY_HUFF, False, False)
DHT_DCC = encode_huff(DCC_HUFF, False, True)
DHT_ACY = encode_huff(ACY_HUFF, True, False)
DHT_ACC = encode_huff(ACC_HUFF, True, True)
DHT = DHT_DCY + DHT_DCC + DHT_ACY + DHT_ACC
DHT = pack(f'>2sH{len(DHT)}s',
    b'\xFF\xC4',        # DHT (Define Huffman Table) segment
    2 + len(DHT),       # Length of segment
    DHT
)

SOS = bytes.fromhex(
    'FFDA'          # SOS (Start of Scan) segment
    '000C'          # Length of segment (6 + 2 * number of components)

    '03'            # Number of channels (3 for color)

    '01 00'         # Choose Huffman tables for the different channels
    '02 11'
    '03 11'

    '003F00'        # More random numbers for no reason
)


HEADER = APP0 + DQT_Y + DQT_C + SOF0 + DHT + SOS

# The payload

payload = bitarray()
acy_itr = zip(acy_symbols, acy_values)
acc_itr = zip(acc_symbols, acc_values)
acy_ptr = iter(np.diff(acy_ptrs))
acc_ptr = iter(np.diff(acc_ptrs))


# Huffman encode the payload
for dc_bc in (it := np.nditer(dc_diff, flags=['multi_index'])):
    b, c = it.multi_index

    if c == 0:
        dc_table, ac_table, itr, ptr = DCY_HUFF, ACY_HUFF, acy_itr, acy_ptr
    else:
        dc_table, ac_table, itr, ptr = DCC_HUFF, ACC_HUFF, acc_itr, acc_ptr

    num_symbols = next(ptr)

    dc_comp = onescomp(int(dc_bc))
    payload.extend(dc_table[len(dc_comp)])
    payload.extend(dc_comp)

    for symbol, value in itertools.islice(itr, num_symbols):
        payload.extend(ac_table[symbol])
        payload.extend(onescomp(value))

# Padding and byte stuffing
payload.extend('1' * payload.padbits)
payload = bytes(payload).replace(b'\xff', b'\xff\x00')

# Save the entire JPEG to a file
with open('silicon_valley.jpg', 'wb') as f:
    f.write(HEADER)
    f.write(payload)
    f.write(b'\xFF\xD9')    # EOI (End Of Image)

#######
# 3e Analysis
if SHOW_PLOTS:
    import PIL.Image, numpy as np, matplotlib.pyplot as plt
    im1 = np.array(PIL.Image.open('silicon_valley.ppm'))
    im2 = np.array(PIL.Image.open('silicon_valley.jpg'))
    diff = im1.astype(int) - im2.astype(int)
    plt.imsave('p3e.pdf', diff / diff.max())
    np.abs(((im1 - im2) ** 2).mean())
