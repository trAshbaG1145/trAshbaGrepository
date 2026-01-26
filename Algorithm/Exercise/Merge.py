# 归并排序算法实现
def merge(left_arr, right_arr):
    result = []
    i = j = 0  # 两个指针，分别指向两个数组的起点
    
    # 第一阶段：两两比较
    while i < len(left_arr) and j < len(right_arr):
        if left_arr[i] <= right_arr[j]:
            result.append(left_arr[i])
            i += 1
        else:
            result.append(right_arr[j])
            j += 1
            
    # 第二、三阶段：将剩下的元素直接合并（两者只会执行其一）
    result.extend(left_arr[i:])
    result.extend(right_arr[j:])
    
    return result

# 测试代码
test_left = [1, 5, 8, 9, 10, 12, 15, 20]
test_right = [2, 4, 6, 8, 10, 11, 13, 14, 16, 18]
merged_result = merge(test_left, test_right)
print(f"合并结果: {merged_result}")  # 输出: 合并结果: [1, 2, 4, 5, 6, 8, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 18, 20]