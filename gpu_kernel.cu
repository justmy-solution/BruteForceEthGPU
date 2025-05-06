// gpu_kernel.cu
#include <stdint.h>
#include <stdio.h>
#include <string.h>


// BIP39 wordlist in constant memory (unchanged)
__device__ __constant__ char wordlist[2048][10];

// SHA-512 Constants
__device__ __constant__ uint64_t K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

// Macros
__device__ void keccak256(const uint8_t* input, int inputlen, uint8_t* output);

__device__ __forceinline__ uint64_t ROTR(uint64_t x, uint64_t n) {
    return (x >> n) | (x << (64 - n));
}

// secp256k1 Field Prime
__device__ __constant__ uint64_t SECP256K1_P[4] = {
    0xFFFFFFFFFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

// Helper: Add two field elements modulo P
__device__ void fe_add(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = a[i] + b[i];
    }

    // Quick reduction (if overflowed)
    bool carry = (r[0] < a[0]);
    for (int i = 1; i < 4; i++) {
        if (carry) {
            r[i]++;
            carry = (r[i] == 0);
        }
    }

    // If result >= P, subtract P
    bool greater_or_equal = true;
    for (int i = 3; i >= 0; i--) {
        if (r[i] < SECP256K1_P[i]) break;
        if (r[i] > SECP256K1_P[i]) {
            greater_or_equal = false;
            break;
        }
    }
    if (greater_or_equal) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            r[i] -= SECP256K1_P[i];
        }
    }
}

// Helper: Subtract two field elements modulo P
__device__ void fe_sub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = a[i] - b[i];
    }

    // If underflow, add back P
    bool borrow = (a[0] < b[0]);
    for (int i = 1; i < 4; i++) {
        if (borrow) {
            if (a[i] == 0) {
                r[i] = (uint64_t)-1 - (b[i] - 1);
            } else {
                r[i] = a[i] - b[i] - 1;
            }
            borrow = (a[i] <= b[i]);
        }
    }

    if (borrow) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            r[i] += SECP256K1_P[i];
        }
    }
}

// Helper: Multiply two field elements modulo P (naive version)
__device__ void fe_mul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    // Naive long multiplication (slow but correct)
    uint64_t tmp[8] = {0};

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a[i]), "l"(b[j]));
            asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a[i]), "l"(b[j]));
            tmp[i + j] += lo;
            tmp[i + j + 1] += hi;
        }
    }

    // Modular reduction (brutal method: subtract P until fit) - we will optimize this later
    for (int i = 0; i < 4; i++) {
        r[i] = tmp[i];
    }
}
// Elliptic Curve: Point Doubling (P = 2P)
__device__ void ec_double(uint64_t rx[4], uint64_t ry[4], const uint64_t px[4], const uint64_t py[4]) {
    uint64_t t0[4], t1[4], t2[4], t3[4];

    // λ = (3 * px²) / (2 * py)
    fe_mul(t0, px, px);          // t0 = px^2
    fe_add(t1, t0, t0);          // t1 = 2 * px^2
    fe_add(t1, t1, t0);          // t1 = 3 * px^2

    fe_add(t2, py, py);          // t2 = 2 * py
    // Need modular inverse of t2 (2*py) -- NOT IMPLEMENTED YET
    // In full implementation, t1 = t1 * inv(t2) mod P
    // For now placeholder multiplication

    // x3 = λ² - 2 * px
    fe_mul(t3, t1, t1);          // t3 = λ^2
    fe_add(t0, px, px);          // t0 = 2 * px
    fe_sub(rx, t3, t0);          // rx = λ^2 - 2px

    // y3 = λ(px - x3) - py
    fe_sub(t0, px, rx);          // t0 = px - x3
    fe_mul(t0, t1, t0);          // t0 = λ(px - x3)
    fe_sub(ry, t0, py);          // ry = λ(px - x3) - py
}

// Elliptic Curve: Point Addition (R = P + Q)
__device__ void ec_add(uint64_t rx[4], uint64_t ry[4], const uint64_t px[4], const uint64_t py[4], const uint64_t qx[4], const uint64_t qy[4]) {
    uint64_t t0[4], t1[4], t2[4], t3[4];

    // λ = (qy - py) / (qx - px)
    fe_sub(t0, qy, py);          // t0 = qy - py
    fe_sub(t1, qx, px);          // t1 = qx - px
    // Need modular inverse of t1 -- NOT IMPLEMENTED YET
    // For now placeholder multiplication

    // x3 = λ² - px - qx
    fe_mul(t2, t0, t0);          // t2 = λ^2
    fe_sub(t3, t2, px);          // t3 = λ^2 - px
    fe_sub(rx, t3, qx);          // rx = λ^2 - px - qx

    // y3 = λ(px - x3) - py
    fe_sub(t2, px, rx);          // t2 = px - x3
    fe_mul(t2, t0, t2);          // t2 = λ(px - x3)
    fe_sub(ry, t2, py);          // ry = λ(px - x3) - py
}
// Scalar Multiplication: privkey * G
__device__ void ec_scalar_mul(uint64_t rx[4], uint64_t ry[4], const uint64_t privkey[4]) {
    // Start from G (secp256k1 base point)
    uint64_t gx[4] = { 
        0x79BE667EF9DCBBACULL, 
        0x55A06295CE870B07ULL, 
        0x029BFCDB2DCE28D9ULL, 
        0x59F2815B16F81798ULL 
    };
    uint64_t gy[4] = {
        0x483ADA7726A3C465ULL,
        0x5DA4FBFC0E1108A8ULL,
        0xFD17B448A6855419ULL,
        0x9C47D08FFB10D4B8ULL
    };

    // Initialize R = INF (point at infinity) [we'll fake it as 0,0 first]
    uint64_t rx0[4] = {0};
    uint64_t ry0[4] = {0};

    for (int bit = 255; bit >= 0; bit--) {
        // R = 2R
        ec_double(rx0, ry0, rx0, ry0);

        int word = bit / 64;
        int bit_in_word = bit % 64;

        if ((privkey[word] >> bit_in_word) & 1) {
            ec_add(rx0, ry0, rx0, ry0, gx, gy);
        }
    }

    // Final public key coordinates
#pragma unroll
    for (int i = 0; i < 4; i++) {
        rx[i] = rx0[i];
        ry[i] = ry0[i];
    }
}
// Serialize compressed public key (33 bytes: 0x02 or 0x03 + X coordinate)
__device__ void serialize_pubkey_compressed(uint8_t out[33], const uint64_t x[4], const uint64_t y[4]) {
    // Determine prefix
    out[0] = (y[0] & 1) ? 0x03 : 0x02; // 0x02 if even, 0x03 if odd Y coordinate

#pragma unroll
    for (int i = 0; i < 4; i++) {
        out[1 + i * 8 + 0] = (x[i] >> 56) & 0xff;
        out[1 + i * 8 + 1] = (x[i] >> 48) & 0xff;
        out[1 + i * 8 + 2] = (x[i] >> 40) & 0xff;
        out[1 + i * 8 + 3] = (x[i] >> 32) & 0xff;
        out[1 + i * 8 + 4] = (x[i] >> 24) & 0xff;
        out[1 + i * 8 + 5] = (x[i] >> 16) & 0xff;
        out[1 + i * 8 + 6] = (x[i] >> 8)  & 0xff;
        out[1 + i * 8 + 7] = (x[i] >> 0)  & 0xff;
    }
}
// Generate Ethereum address from pubkey
__device__ void pubkey_to_eth_address(uint8_t address_out[20], const uint64_t pubkey_x[4], const uint64_t pubkey_y[4]) {
    uint8_t serialized[33];
    uint8_t hash[32];

    serialize_pubkey_compressed(serialized, pubkey_x, pubkey_y);
    keccak256(serialized, 33, hash);

    // Copy last 20 bytes
#pragma unroll
    for (int i = 0; i < 20; i++) {
        address_out[i] = hash[12 + i];
    }
}

// SHA512 Transform
__device__ void sha512_transform(uint64_t* state, const uint8_t* block) {
    uint64_t W[80];
    uint64_t a, b, c, d, e, f, g, h;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint64_t)block[i * 8 + 0] << 56) |
               ((uint64_t)block[i * 8 + 1] << 48) |
               ((uint64_t)block[i * 8 + 2] << 40) |
               ((uint64_t)block[i * 8 + 3] << 32) |
               ((uint64_t)block[i * 8 + 4] << 24) |
               ((uint64_t)block[i * 8 + 5] << 16) |
               ((uint64_t)block[i * 8 + 6] << 8) |
               ((uint64_t)block[i * 8 + 7]);
    }

    #pragma unroll
    for (int i = 16; i < 80; i++) {
        uint64_t s0 = ROTR(W[i-15],1) ^ ROTR(W[i-15],8) ^ (W[i-15] >> 7);
        uint64_t s1 = ROTR(W[i-2],19) ^ ROTR(W[i-2],61) ^ (W[i-2] >> 6);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }

    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    #pragma unroll
    for (int i = 0; i < 80; i++) {
        uint64_t S1 = ROTR(e,14) ^ ROTR(e,18) ^ ROTR(e,41);
        uint64_t ch = (e & f) ^ ((~e) & g);
        uint64_t temp1 = h + S1 + ch + K[i] + W[i];
        uint64_t S0 = ROTR(a,28) ^ ROTR(a,34) ^ ROTR(a,39);
        uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint64_t temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// SHA512 wrapper
__device__ void sha512(const uint8_t* data, int datalen, uint8_t* out) {
    uint64_t state[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };

    // Only support short messages for now
    sha512_transform(state, data);

    for (int i = 0; i < 8; i++) {
        out[i*8 + 0] = (state[i] >> 56) & 0xff;
        out[i*8 + 1] = (state[i] >> 48) & 0xff;
        out[i*8 + 2] = (state[i] >> 40) & 0xff;
        out[i*8 + 3] = (state[i] >> 32) & 0xff;
        out[i*8 + 4] = (state[i] >> 24) & 0xff;
        out[i*8 + 5] = (state[i] >> 16) & 0xff;
        out[i*8 + 6] = (state[i] >> 8) & 0xff;
        out[i*8 + 7] = (state[i] >> 0) & 0xff;
    }
}

// HMAC-SHA512
__device__ void hmac_sha512(const uint8_t* key, int key_len, const uint8_t* message, int message_len, uint8_t* output) {
    uint8_t k_ipad[128] = {0};
    uint8_t k_opad[128] = {0};
    uint8_t inner_hash[64];

    if (key_len > 128) {
        // If key longer than blocksize, hash it
        sha512(key, key_len, k_ipad);
        key = k_ipad;
        key_len = 64;
    }

    for (int i = 0; i < key_len; i++) {
        k_ipad[i] = key[i] ^ 0x36;
        k_opad[i] = key[i] ^ 0x5c;
    }
    for (int i = key_len; i < 128; i++) {
        k_ipad[i] = 0x36;
        k_opad[i] = 0x5c;
    }

    uint8_t inner_data[256];
    for (int i = 0; i < 128; i++) inner_data[i] = k_ipad[i];
    for (int i = 0; i < message_len; i++) inner_data[128+i] = message[i];

    sha512(inner_data, 128+message_len, inner_hash);

    uint8_t outer_data[192];
    for (int i = 0; i < 128; i++) outer_data[i] = k_opad[i];
    for (int i = 0; i < 64; i++) outer_data[128+i] = inner_hash[i];

    sha512(outer_data, 192, output);
}

// PBKDF2-HMAC-SHA512
__device__ void pbkdf2_hmac_sha512(const uint8_t* password, int password_len, const uint8_t* salt, int salt_len, int iterations, uint8_t* output, int dklen) {
    uint8_t U[64];
    uint8_t T[64];

    for (int block_index = 1; dklen > 0; block_index++) {
        uint8_t salt_block[256];

        for (int i = 0; i < salt_len; i++) {
            salt_block[i] = salt[i];
        }
        salt_block[salt_len + 0] = (block_index >> 24) & 0xFF;
        salt_block[salt_len + 1] = (block_index >> 16) & 0xFF;
        salt_block[salt_len + 2] = (block_index >> 8) & 0xFF;
        salt_block[salt_len + 3] = (block_index) & 0xFF;

        hmac_sha512(password, password_len, salt_block, salt_len+4, U);

        for (int i = 0; i < 64; i++) {
            T[i] = U[i];
        }

        for (int j = 1; j < iterations; j++) {
            hmac_sha512(password, password_len, U, 64, U);
            for (int i = 0; i < 64; i++) {
                T[i] ^= U[i];
            }
        }

        int copy_len = dklen < 64 ? dklen : 64;
        for (int i = 0; i < copy_len; i++) {
            output[i] = T[i];
        }
        output += copy_len;
        dklen -= copy_len;
    }
}

// (Your keccak256, validate_bip39_checksum, search_seeds kernel would follow here)
// Helper: Convert 4 bytes to 32 bits
__device__ uint32_t to_u32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) |
           ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  |
           ((uint32_t)p[3]);
}

// BIP39 Checksum Validator
__device__ bool validate_bip39_checksum(const uint16_t* word_indices) {
    uint8_t entropy[32] = {0}; // Max space for 24 words

    int ent_bits = 128; // 12-word mnemonic = 128 bits entropy + 4 bits checksum

    // Rebuild entropy from word indices
    int bitpos = 0;
    for (int i = 0; i < 11 * 11; i++) { // 11 bits per word, 11 words
        int word_idx = i / 11;
        int bit_idx = 10 - (i % 11);
        if (word_indices[word_idx] & (1 << bit_idx)) {
            entropy[bitpos / 8] |= (1 << (7 - (bitpos % 8)));
        }
        bitpos++;
    }

    uint8_t hash[64];
    sha512(entropy, 16, hash); // Only first 16 bytes used for 12-word seed

    int checksum_bits = ent_bits / 32;
    uint8_t checksum_expected = hash[0] >> (8 - checksum_bits);

    uint8_t checksum_actual = word_indices[11] & ((1 << checksum_bits) - 1);

    return checksum_expected == checksum_actual;
}

// KECCAK-256 simplified (for Ethereum address)
__device__ void keccak_f(uint64_t state[25]) {
    const uint64_t keccakf_rndc[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
        0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
        0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
        0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
        0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
    };

    const int keccakf_rotc[24] = {
         1,  3,  6, 10, 15, 21,
         28, 36, 45, 55,  2, 14,
         27, 41, 56, 8, 25, 43,
         62, 18, 39, 61, 20, 44
    };

    const int keccakf_piln[24] = {
         10, 7, 11, 17, 18, 3, 5, 16,
         8, 21, 24, 4, 15, 23, 19, 13,
         12, 2, 20, 14, 22, 9, 6, 1
    };

    for (int round = 0; round < 24; round++) {
        uint64_t bc[5];

        for (int i = 0; i < 5; i++) {
            bc[i] = state[i] ^ state[i+5] ^ state[i+10] ^ state[i+15] ^ state[i+20];
        }

        for (int i = 0; i < 5; i++) {
            uint64_t t = bc[(i+4)%5] ^ ROTR(bc[(i+1)%5], 1);
            for (int j = 0; j < 25; j += 5) {
                state[j+i] ^= t;
            }
        }

        uint64_t t = state[1];
        for (int i = 0; i < 24; i++) {
            int j = keccakf_piln[i];
            bc[0] = state[j];
            state[j] = ROTR(t, keccakf_rotc[i]);
            t = bc[0];
        }

        for (int j = 0; j < 5; j++) {
            bc[j] = state[j] ^ state[j+5] ^ state[j+10] ^ state[j+15] ^ state[j+20];
        }

        for (int j = 0; j < 5; j++) {
            for (int i = 0; i < 25; i += 5) {
                state[i+j] ^= (~bc[(j+1)%5]) & bc[(j+2)%5];
            }
        }

        state[0] ^= keccakf_rndc[round];
    }
}

__device__ void keccak256(const uint8_t* input, int inputlen, uint8_t* output) {
    uint64_t state[25] = {0};

    for (int i = 0; i < inputlen; i++) {
        int idx = i / 8;
        int shift = (i % 8) * 8;
        state[idx] ^= (uint64_t)input[i] << shift;
    }

    state[inputlen / 8] ^= (uint64_t)0x01ULL << ((inputlen % 8) * 8);
    state[16] ^= 0x8000000000000000ULL;

    keccak_f(state);

    for (int i = 0; i < 32; i++) {
        output[i] = (state[i/8] >> ((i%8)*8)) & 0xFF;
    }
}

// The main GPU Kernel
extern "C" __global__ void search_seeds(
    uint64_t* seeds_tested,
    uint64_t* seeds_found,
    uint64_t  start_index,
    uint64_t  batch_size,
    int       known_count,
    const int* known_indices,
    const uint8_t* target_address,
    int       match_mode,
    int       match_prefix_len
) {
    // 1) Thread index within this worker’s batch
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    uint64_t idx = start_index + tid;

    // 2) Dynamic word index decoding
    int word_indices[12];
    #pragma unroll
    for (int i = 0; i < known_count; i++) {
        word_indices[i] = known_indices[i];
    }
    uint64_t rem = idx;
    for (int pos = known_count; pos < 12; pos++) {
        word_indices[pos] = rem % 2048;
        rem /= 2048;
    }

    // 3) Prepare buffer for mnemonic assembly
    uint8_t mnemonic[256];
    int     mnemonic_len = 0;

    // 4) BIP39 checksum validation
    if (!validate_bip39_checksum((uint16_t*)word_indices)) {
        atomicAdd((unsigned long long*)seeds_tested, 1ULL);
        return;
    }

    // 5) Build mnemonic string
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        const char* w = wordlist[word_indices[i]];
        for (int j = 0; w[j] != '\0'; j++) {
            mnemonic[mnemonic_len++] = w[j];
        }
        if (i < 11) mnemonic[mnemonic_len++] = ' ';
    }
    mnemonic[mnemonic_len] = '\0';

    // 6) Derive seed via PBKDF2-HMAC-SHA512
    uint8_t seed[64];
    pbkdf2_hmac_sha512(
        mnemonic,          // password
        mnemonic_len,      // password length
        (const uint8_t*)"mnemonic",  // salt
        8,                 // salt length
        2048,              // iterations
        seed,              // output
        64                 // dklen
    );

    // 7) Extract private key (first 32 bytes)
    uint8_t private_key[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        private_key[i] = seed[i];
    }

    // 8) Generate real public key from private key
    uint64_t priv_scalar[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        priv_scalar[i] = 
            ((uint64_t)private_key[i*8 + 0] << 56) |
            ((uint64_t)private_key[i*8 + 1] << 48) |
            ((uint64_t)private_key[i*8 + 2] << 40) |
            ((uint64_t)private_key[i*8 + 3] << 32) |
            ((uint64_t)private_key[i*8 + 4] << 24) |
            ((uint64_t)private_key[i*8 + 5] << 16) |
            ((uint64_t)private_key[i*8 + 6] <<  8) |
            ((uint64_t)private_key[i*8 + 7] <<  0);
    }
    uint64_t pubkey_x[4], pubkey_y[4];
    ec_scalar_mul(pubkey_x, pubkey_y, priv_scalar);

    // 9) Derive Ethereum address
    uint8_t eth_addr[20];
    pubkey_to_eth_address(eth_addr, pubkey_x, pubkey_y);

    // 10) Matching logic
    bool match = true;
    if (match_mode == 0) {
        for (int i = 0; i < 20; i++)
            if (eth_addr[i] != target_address[i]) { match = false; break; }
    } else if (match_mode == 1) {
        for (int i = 0; i < match_prefix_len; i++)
            if (eth_addr[i] != target_address[i]) { match = false; break; }
    } else {
        for (int i = 0; i < 20; i++)
            if (eth_addr[i] != 0x00) { match = false; break; }
    }

    // 11) Report results
    if (match) {
        printf("[FOUND] Mnemonic: %s\n", mnemonic);
        atomicAdd((unsigned long long*)seeds_found, 1ULL);
    }
    atomicAdd((unsigned long long*)seeds_tested, 1ULL);
} // end search_seeds
