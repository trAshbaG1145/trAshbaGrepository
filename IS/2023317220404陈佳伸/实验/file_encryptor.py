#!/usr/bin/env python3

"""
3DES-CBC 文件加解密工具 (file_encryptor.py)

使用方法:
  加密: python file_encryptor.py encrypt -i <原始文件> -o <加密文件>
  解密: python file_encryptor.py decrypt -i <加密文件> -o <解密文件>
"""

import os
import argparse
import hashlib
import getpass
import sys

# =============================================================================
# 步骤 1.1: DES 常量定义
# =============================================================================

# 初始置换表 (IP)
IP = [58, 50, 42, 34, 26, 18, 10, 2,
      60, 52, 44, 36, 28, 20, 12, 4,
      62, 54, 46, 38, 30, 22, 14, 6,
      64, 56, 48, 40, 32, 24, 16, 8,
      57, 49, 41, 33, 25, 17, 9, 1,
      59, 51, 43, 35, 27, 19, 11, 3,
      61, 53, 45, 37, 29, 21, 13, 5,
      63, 55, 47, 39, 31, 23, 15, 7]

# 最终置换表 (FP or IP-1)
FP = [40, 8, 48, 16, 56, 24, 64, 32,
      39, 7, 47, 15, 55, 23, 63, 31,
      38, 6, 46, 14, 54, 22, 62, 30,
      37, 5, 45, 13, 53, 21, 61, 29,
      36, 4, 44, 12, 52, 20, 60, 28,
      35, 3, 43, 11, 51, 19, 59, 27,
      34, 2, 42, 10, 50, 18, 58, 26,
      33, 1, 41, 9, 49, 17, 57, 25]

# 扩展置换表 (E)
E = [32, 1, 2, 3, 4, 5,
     4, 5, 6, 7, 8, 9,
     8, 9, 10, 11, 12, 13,
     12, 13, 14, 15, 16, 17,
     16, 17, 18, 19, 20, 21,
     20, 21, 22, 23, 24, 25,
     24, 25, 26, 27, 28, 29,
     28, 29, 30, 31, 32, 1]

# P-盒置换表 (P)
P = [16, 7, 20, 21, 29, 12, 28, 17,
     1, 15, 23, 26, 5, 18, 31, 10,
     2, 8, 24, 14, 32, 27, 3, 9,
     19, 13, 30, 6, 22, 11, 4, 25]

# 密钥置换选择 1 (PC-1)
PC1 = [57, 49, 41, 33, 25, 17, 9,
       1, 58, 50, 42, 34, 26, 18,
       10, 2, 59, 51, 43, 35, 27,
       19, 11, 3, 60, 52, 44, 36,
       63, 55, 47, 39, 31, 23, 15,
       7, 62, 54, 46, 38, 30, 22,
       14, 6, 61, 53, 45, 37, 29,
       21, 13, 5, 28, 20, 12, 4]

# 密钥置换选择 2 (PC-2)
PC2 = [14, 17, 11, 24, 1, 5,
       3, 28, 15, 6, 21, 10,
       23, 19, 12, 4, 26, 8,
       16, 7, 27, 20, 13, 2,
       41, 52, 31, 37, 47, 55,
       30, 40, 51, 45, 33, 48,
       44, 49, 39, 56, 34, 53,
       46, 42, 50, 36, 29, 32]

# 密钥调度左移位数
KEY_SHIFTS = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

# S-盒 (8 个 S-盒)
S_BOXES = [
    [[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
     [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
     [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
     [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],
    [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
     [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
     [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
     [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],
    [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
     [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
     [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
     [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],
    [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
     [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
     [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
     [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],
    [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
     [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
     [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
     [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],
    [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
     [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
     [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
     [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],
    [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
     [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
     [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
     [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],
    [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
     [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
     [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
     [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]]
]

# =============================================================================
# 步骤 1.2: DES 辅助函数
# =============================================================================

def bytes_to_bit_list(data: bytes) -> list[int]:
    """将字节串转换为 0/1 的位列表"""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def bit_list_to_bytes(bits: list[int]) -> bytes:
    """将 0/1 的位列表转换为字节串"""
    if len(bits) % 8 != 0:
        raise ValueError("位列表长度必须是 8 的倍数")
    bytes_out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i+j]
        bytes_out.append(byte)
    return bytes(bytes_out)

def permute(bit_list: list[int], table: list[int]) -> list[int]:
    """根据置换表对位列表进行置换"""
    return [bit_list[i - 1] for i in table]

def xor(bits1: list[int], bits2: list[int]) -> list[int]:
    """对两个位列表进行异或操作"""
    return [b1 ^ b2 for b1, b2 in zip(bits1, bits2)]

def bits_to_int(bits: list[int]) -> int:
    """位列表转整数"""
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val

def int_to_bits(val: int, num_bits: int) -> list[int]:
    """整数转固定长度的位列表"""
    bits = [int(b) for b in bin(val)[2:]]
    return [0] * (num_bits - len(bits)) + bits

# =============================================================================
# 步骤 1.3: DES 密钥调度
# =============================================================================

def generate_subkeys(key_64bit: bytes) -> list[list[int]]:
    """从 64 位密钥生成 16 轮的 48 位子密钥"""
    if len(key_64bit) != 8:
        raise ValueError("DES 密钥必须是 8 字节 (64 位)")
        
    key_bits = bytes_to_bit_list(key_64bit)
    key_56bit = permute(key_bits, PC1)
    
    C = key_56bit[:28]
    D = key_56bit[28:]
    
    subkeys = []
    for i in range(16):
        shift = KEY_SHIFTS[i]
        C = C[shift:] + C[:shift]
        D = D[shift:] + D[:shift]
        subkey = permute(C + D, PC2)
        subkeys.append(subkey)
        
    return subkeys

# =============================================================================
# 步骤 1.4: DES 加解密核心 (Feistel 网络)
# =============================================================================

def des_core(block_64bit: bytes, subkeys: list[list[int]]) -> bytes:
    """DES 核心算法（加密或解密），取决于 subkeys 的顺序"""
    if len(block_64bit) != 8:
        raise ValueError("DES 数据块必须是 8 字节 (64 位)")

    bits = bytes_to_bit_list(block_64bit)
    bits = permute(bits, IP)
    
    L = bits[:32]
    R = bits[32:]
    
    for i in range(16):
        L_prev = L
        R_prev = R
        subkey = subkeys[i]
        
        # f(R, K) 函数
        R_expanded = permute(R_prev, E)
        R_xored = xor(R_expanded, subkey)
        
        s_box_output = []
        for j in range(8):
            chunk = R_xored[j*6 : (j+1)*6]
            row = bits_to_int([chunk[0], chunk[5]])
            col = bits_to_int(chunk[1:5])
            val = S_BOXES[j][row][col]
            s_box_output.extend(int_to_bits(val, 4))
        
        f_result = permute(s_box_output, P)
        
        L = R_prev
        R = xor(L_prev, f_result)
    
    final_bits = R + L
    final_permuted = permute(final_bits, FP)
    
    return bit_list_to_bytes(final_permuted)

def des_encrypt_block(block: bytes, key: bytes) -> bytes:
    subkeys = generate_subkeys(key)
    return des_core(block, subkeys)

def des_decrypt_block(block: bytes, key: bytes) -> bytes:
    subkeys = generate_subkeys(key)
    return des_core(block, subkeys[::-1]) # 解密时子密钥逆序

# =============================================================================
# 步骤 2: 填充模式 (PKCS7)
# =============================================================================

BLOCK_SIZE_DES = 8

def pad_pkcs7(data: bytes, block_size: int = BLOCK_SIZE_DES) -> bytes:
    """应用 PKCS7 填充"""
    padding_len = block_size - (len(data) % block_size)
    padding = bytes([padding_len]) * padding_len
    return data + padding

def unpad_pkcs7(data: bytes, block_size: int = BLOCK_SIZE_DES) -> bytes:
    """移除 PKCS7 填充"""
    if not data:
        raise ValueError("解密数据为空，无法去填充")
        
    padding_len = data[-1]
    
    if padding_len > block_size or padding_len == 0:
        raise ValueError("填充数据无效 (长度错误)")
        
    if data[-padding_len:] != bytes([padding_len]) * padding_len:
        raise ValueError("填充数据无效 (内容错误)")
        
    return data[:-padding_len]

# =============================================================================
# 步骤 3 & 4: 3DES-CBC 模式实现
# =============================================================================

def xor_bytes(b1: bytes, b2: bytes) -> bytes:
    """对两个等长字节串进行异或"""
    return bytes(x ^ y for x, y in zip(b1, b2))

def triple_des_encrypt_block(block: bytes, key1: bytes, key2: bytes, key3: bytes) -> bytes:
    block = des_encrypt_block(block, key1)
    block = des_decrypt_block(block, key2)
    block = des_encrypt_block(block, key3)
    return block

def triple_des_decrypt_block(block: bytes, key1: bytes, key2: bytes, key3: bytes) -> bytes:
    block = des_decrypt_block(block, key3)
    block = des_encrypt_block(block, key2)
    block = des_decrypt_block(block, key1)
    return block

def triple_des_cbc_encrypt(plaintext_padded: bytes, key_24byte: bytes, iv: bytes) -> bytes:
    if len(key_24byte) != 24:
        raise ValueError("3DES 密钥必须是 24 字节")
    if len(iv) != BLOCK_SIZE_DES:
        raise ValueError(f"IV 必须是 {BLOCK_SIZE_DES} 字节")
    if len(plaintext_padded) % BLOCK_SIZE_DES != 0:
        raise ValueError("加密数据必须先填充到块大小的整数倍")
        
    k1, k2, k3 = key_24byte[0:8], key_24byte[8:16], key_24byte[16:24]
    
    ciphertext = b''
    previous_block = iv
    
    for i in range(0, len(plaintext_padded), BLOCK_SIZE_DES):
        block = plaintext_padded[i : i + BLOCK_SIZE_DES]
        block_to_encrypt = xor_bytes(block, previous_block)
        encrypted_block = triple_des_encrypt_block(block_to_encrypt, k1, k2, k3)
        ciphertext += encrypted_block
        previous_block = encrypted_block
        
    return ciphertext

def triple_des_cbc_decrypt(ciphertext: bytes, key_24byte: bytes, iv: bytes) -> bytes:
    if len(key_24byte) != 24:
        raise ValueError("3DES 密钥必须是 24 字节")
    if len(iv) != BLOCK_SIZE_DES:
        raise ValueError(f"IV 必须是 {BLOCK_SIZE_DES} 字节")
    if len(ciphertext) % BLOCK_SIZE_DES != 0:
        raise ValueError("密文长度必须是块大小的整数倍")

    k1, k2, k3 = key_24byte[0:8], key_24byte[8:16], key_24byte[16:24]
    
    plaintext_padded = b''
    previous_block = iv
    
    for i in range(0, len(ciphertext), BLOCK_SIZE_DES):
        block = ciphertext[i : i + BLOCK_SIZE_DES]
        decrypted_block = triple_des_decrypt_block(block, k1, k2, k3)
        plaintext_block = xor_bytes(decrypted_block, previous_block)
        plaintext_padded += plaintext_block
        previous_block = block
        
    return plaintext_padded

# =============================================================================
# 步骤 5: 文件加解密主函数
# =============================================================================

def encrypt_file(input_file: str, output_file: str, key_24byte: bytes):
    """使用 3DES-CBC 加密文件"""
    try:
        iv = os.urandom(BLOCK_SIZE_DES)
        
        with open(input_file, 'rb') as f_in:
            plaintext = f_in.read()
            
        plaintext_padded = pad_pkcs7(plaintext, BLOCK_SIZE_DES)
        ciphertext = triple_des_cbc_encrypt(plaintext_padded, key_24byte, iv)
        
        with open(output_file, 'wb') as f_out:
            f_out.write(iv)
            f_out.write(ciphertext)
            
        print(f"文件 '{input_file}' 加密成功, 已保存为 '{output_file}'")
        
    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file}' 未找到。")
    except Exception as e:
        print(f"加密失败: {e}")

def decrypt_file(input_file: str, output_file: str, key_24byte: bytes):
    """使用 3DES-CBC 解密文件"""
    try:
        with open(input_file, 'rb') as f_in:
            iv = f_in.read(BLOCK_SIZE_DES)
            if len(iv) < BLOCK_SIZE_DES:
                raise ValueError("加密文件不完整或格式错误 (无法读取 IV)")
            
            ciphertext = f_in.read()
            
        plaintext_padded = triple_des_cbc_decrypt(ciphertext, key_24byte, iv)
        plaintext = unpad_pkcs7(plaintext_padded, BLOCK_SIZE_DES)
        
        with open(output_file, 'wb') as f_out:
            f_out.write(plaintext)
            
        print(f"文件 '{input_file}' 解密成功, 已保存为 '{output_file}'")
        
    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file}' 未找到。")
    except ValueError as e:
        print(f"解密失败: {e}。 (提示：这通常意味着密钥错误或文件已损坏)")
    except Exception as e:
        print(f"解密失败: {e}")

# =============================================================================
# 步骤 6: 命令行界面 (CLI)
# =============================================================================

def get_key_from_password(password: str) -> bytes:
    """使用 SHA-256 从密码派生 24 字节的密钥"""
    # 使用 SHA-256 生成 32 字节的哈希值
    hasher = hashlib.sha256()
    hasher.update(password.encode('utf-8'))
    hash_32byte = hasher.digest()
    
    # 截取前 24 字节 (192 位) 作为 3DES 密钥
    return hash_32byte[:24]

def main():
    parser = argparse.ArgumentParser(description="3DES-CBC 文件加解密工具")
    subparsers = parser.add_subparsers(dest='command', required=True, help="选择 'encrypt' 或 'decrypt'")

    # 加密子命令
    enc_parser = subparsers.add_parser('encrypt', help="加密一个文件")
    enc_parser.add_argument('-i', '--input', required=True, help="要加密的原始文件路径")
    enc_parser.add_argument('-o', '--output', required=True, help="加密后保存的文件路径")

    # 解密子命令
    dec_parser = subparsers.add_parser('decrypt', help="解密一个文件")
    dec_parser.add_argument('-i', '--input', required=True, help="要解密的加密文件路径")
    dec_parser.add_argument('-o', '--output', required=True, help="解密后保存的文件路径")

    args = parser.parse_args()

    if args.command == 'encrypt':
        password = getpass.getpass("请输入加密密码: ")
        password_confirm = getpass.getpass("请再次输入密码确认: ")
        
        if password != password_confirm:
            print("两次输入的密码不一致。")
            sys.exit(1)
            
        key = get_key_from_password(password)
        encrypt_file(args.input, args.output, key)
        
    elif args.command == 'decrypt':
        password = getpass.getpass("请输入解密密码: ")
        key = get_key_from_password(password)
        decrypt_file(args.input, args.output, key)

if __name__ == "__main__":
    main()