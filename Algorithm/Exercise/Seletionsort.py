# 选择排序
def selection_sort(nums):
    n = len(nums)
    # 外层循环：i 表示当前需要填充的“最小元素”位置
    for i in range(n - 1):
        # 1. 初始假设当前位置 i 就是最小值所在的索引
        min_idx = i
        
        # 2. 内层循环：从 i+1 到最后，寻找真正的最小值
        for j in range(i + 1, n):
            if nums[j] < nums[min_idx]:
                min_idx = j
        
        # 3. 交换：如果找到的最小值索引不是 i，则交换
        # 注意：Python 支持 a, b = b, a 的简写，但在考试中可能需要写临时变量
        if min_idx != i:
            nums[i], nums[min_idx] = nums[min_idx], nums[i]
            
    return nums

# 测试代码
test_arr = [12, 5, 3, 8, 6, 4, 7, 10, 2, 1, 9, 11, 15, 14, 13, 0, -1, -5]
print(f"排序前: {test_arr}")
selection_sort(test_arr)
print(f"排序后: {test_arr}")