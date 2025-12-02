#include <stdio.h>
#include <stdlib.h>
int compare(const void *a, const void *b)
{
    return *(int *)a - *(int *)b;
}
void backtrack(int nums[], int n, int start, int current_sum, int target, int *count)
{
    if (current_sum == target)
    {
        (*count)++;
        return;
    }
    if (current_sum > target)
    {
        return;
    }
    for (int i = start; i < n; i++)
    {
        if (i > start && nums[i] == nums[i - 1])
        {
            continue;
        }
        backtrack(nums, n, i + 1, current_sum + nums[i], target, count);
    }
}

int main()
{
    int n, target;
    scanf("%d %d", &n, &target);
    int *nums = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &nums[i]);
    }
    qsort(nums, n, sizeof(int), compare);
    int count = 0;
    backtrack(nums, n, 0, 0, target, &count);
    printf("%d\n", count);
    return 0;
}