# 二分查找算法实现
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:  # 注意是 <=
        # 防止溢出的写法：mid = left + (right - left) // 2
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1  # 未找到
    
# 示例用法
if __name__ == "__main__":
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    target = 5
    result = binary_search(nums, target)
    print(f"Target {target} found at index: {result}")  # 输出: Target 5 found at index: 4