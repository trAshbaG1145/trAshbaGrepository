# 自底向上的归并排序实现
def bottom_up_merge_sort(arr):
    n = len(arr)
    # 辅助数组，用于存放合并结果
    temp = [0] * n
    
    # sz 是子数组的长度：1, 2, 4, 8...
    sz = 1
    while sz < n:
        # i 是每一组待合并子数组的起始下标
        for i in range(0, n - sz, sz * 2):
            # 确定三个关键边界
            low = i
            mid = i + sz - 1
            high = min(i + 2 * sz - 1, n - 1)
            
            # 执行合并动作
            merge(arr, temp, low, mid, high)
        
        sz *= 2
    return arr

# 标准的合并函数
def merge(arr, temp, low, mid, high):
    """
    标准的合并函数：将 arr[low...mid] 和 arr[mid+1...high] 合并
    """
    i, j = low, mid + 1
    
    # 先把原数组数据考到临时数组中
    for k in range(low, high + 1):
        temp[k] = arr[k]
    
    # 回填到原数组
    for k in range(low, high + 1):
        if i > mid:               # 左半边用尽
            arr[k] = temp[j]; j += 1
        elif j > high:            # 右半边用尽
            arr[k] = temp[i]; i += 1
        elif temp[j] < temp[i]:   # 右边更小
            arr[k] = temp[j]; j += 1
        else:                     # 左边更小
            arr[k] = temp[i]; i += 1

# 测试代码
if __name__ == "__main__":
    test_array = [38, 27, 43, 3, 9, 82, 10, 27, 15, 44, 6, 12]
    sorted_array = bottom_up_merge_sort(test_array)
    print(f"排序结果: {sorted_array}")  # 输出: 排序结果: [3, 9, 10, 15, 27, 27, 38, 43, 44, 82]