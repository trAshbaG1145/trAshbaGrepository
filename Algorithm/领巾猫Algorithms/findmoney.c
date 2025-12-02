#include <stdio.h>
#include <string.h>
#define MAXN 10
#define MAXM 20001
#define INF 114514

int main()
{
    int n, T[MAXN], Coins[MAXN], m;
    scanf("%d", &n);
    for (int i = 0; i < n; i++)
    {
        scanf("%d %d", &T[i], &Coins[i]);
    }
    scanf("%d", &m);

    int dp[MAXM];
    for (int i = 1; i <= m; i++)
        dp[i] = INF;
    dp[0] = 0;
    for (int j = 0; j < n; j++)
    {
        for (int i = m; i >= 0; i--)
        {
            for (int k = 1; k <= Coins[j]; k++)
            {
                if (i - k * T[j] >= 0)
                {
                    if (dp[i] > dp[i - k * T[j]] + k)
                        dp[i] = dp[i - k * T[j]] + k;
                }
            }
        }
    }
    if (dp[m] == INF)
        printf("-1\n");
    else
        printf("%d\n", dp[m]);
    return 0;
}