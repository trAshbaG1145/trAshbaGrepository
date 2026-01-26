# 插入排序算法实现
def insertion_sort(nums):
    # 从第二个元素开始插入（下标 1）
    for i in range(1, len(nums)):
        key = nums[i]  # 哨兵，保存当前待插入的值
        j = i - 1
        
        # 将比 key 大的元素全部向右移动一位
        while j >= 0 and nums[j] > key:
            nums[j + 1] = nums[j]
            j -= 1
        
        # 找到合适位置后插入
        nums[j + 1] = key
    return nums

# 测试代码
test_arr = [12, 5, 3, 8, 6, 4, 7, 10, 2, 1, 9, 11, 15, 14, 13, 0, -1, -5]
print(f"排序前: {test_arr}")
insertion_sort(test_arr)
print(f"排序后: {test_arr}")