#include <string.h>
#include <stdio.h>
#define MAX_MIN_PRIME 1000010
#define MAX_PRIME 100010

int min_prime[MAX_MIN_PRIME];
int prime[MAX_PRIME];
int g[MAX_MIN_PRIME];
int splitArray(int nums[], int numsSize)
{
    if (numsSize == 0)
        return 0;
    int n = numsSize;
    int m = 2;
    for (int i = 0; i < n; ++i)
    {
        if (nums[i] > m)
        {
            m = nums[i];
        }
    }
    memset(min_prime, 0, sizeof(min_prime));
    memset(prime, 0, sizeof(prime));
    int prime_cnt = 0;
    for (int i = 2; i <= m; ++i)
    {
        if (min_prime[i] == 0)
        {
            prime[++prime_cnt] = i;
            min_prime[i] = i;
        }
        for (int j = 1; j <= prime_cnt; ++j)
        {
            if (i > m / prime[j])
                break;
            min_prime[i * prime[j]] = prime[j];
            if (i % prime[j] == 0)
                break;
        }
    }
    prime[0] = prime_cnt;
    for (int j = 1; j <= prime[0]; ++j)
    {
        g[prime[j]] = n;
    }
    int first_num = nums[0];
    for (int x = first_num; x > 1; x /= min_prime[x])
    {
        int p = min_prime[x];
        g[p] = 0;
    }
    int ans = 1;
    for (int i = 0; i < n; ++i)
    {
        ans = i + 1;
        int curr_num = nums[i];
        for (int x = curr_num; x > 1; x /= min_prime[x])
        {
            int p = min_prime[x];
            if (g[p] + 1 < ans)
            {
                ans = g[p] + 1;
            }
        }
        if (i == n - 1)
            break;
        int next_num = nums[i + 1];
        for (int x = next_num; x > 1; x /= min_prime[x])
        {
            int p = min_prime[x];
            if (ans < g[p])
            {
                g[p] = ans;
            }
        }
    }

    return ans;
}

int main()
{
    int n;
    scanf("%d", &n);
    int nums[n];
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &nums[i]);
    }
    int t = splitArray(nums, n);
    printf("%d", t);
}