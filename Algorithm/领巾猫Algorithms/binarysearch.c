#include <stdio.h>
int search(int low, int high, int target, int nums[])
{
    if (low > high)
    {
        return -1;
    }
    else
    {
        int mid = (low + high) / 2;
        if (nums[mid] == target)
        {
            return mid;
        }
        else if (target < nums[mid])
        {
            return search(low, mid - 1, target, nums);
        }
        else
        {
            return search(mid + 1, high, target, nums);
        }
    }
}
int main()
{
    int length, target;
    scanf("%d", &length);
    int num[length];
    for (int i = 0; i < length; i++)
    {
        scanf("%d", &num[i]);
    }
    scanf("%d", &target);
    int res = search(0, length - 1, target, num);
    printf("%d", res);
}