#include <stdio.h>
#include <iostream>
#include <string.h>
#include <algorithm>
#include <math.h>
#include <queue>
using namespace std;
#define M 251
int FIRSTM[M], SECONDM[M], dp[M][M];
const int INF = 1 << 30;
int n, sum;
int main()
{
    while (cin >> n)
    {
        int i, j;
        sum = 0;
        for (i = 0; i < n; i++)
        {
            cin >> FIRSTM[i];
            sum += FIRSTM[i];
        }
        for (i = 0; i < n; i++)
            cin >> SECONDM[i];
        for (i = 1; i <= n; i++)
        {
            for (j = 0; j <= sum; j++)
                if (j < FIRSTM[i - 1])
                    dp[i][j] = dp[i - 1][j] + SECONDM[i - 1];
                else if (dp[i - 1][j - FIRSTM[i - 1]] > dp[i - 1][j] + SECONDM[i - 1])
                    dp[i][j] = dp[i - 1][j] + SECONDM[i - 1];
                else
                    dp[i][j] = dp[i - 1][j - FIRSTM[i - 1]];
        }
        int temp, ans = INF;
        for (i = 0; i <= sum; i++)
        {
            temp = dp[n][i] > i ? dp[n][i] : i;
            if (temp < ans)
                ans = temp;
        }
        cout << ans << endl;
    }
}
