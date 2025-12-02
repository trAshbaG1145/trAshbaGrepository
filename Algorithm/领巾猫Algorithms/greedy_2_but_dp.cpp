#include <iostream>
#include <algorithm>
#include <climits>
using namespace std;

int Coins[6];
int CoinValue[6] = {5, 10, 20, 50, 100, 200};
int INF = INT_MAX / 2;

int main()
{
    while (true)
    {
        bool allZero = true;
        for (int i = 0; i < 6; i++)
        {
            scanf("%d", &Coins[i]);
            if (Coins[i] != 0)
                allZero = false;
        }
        if (allZero)
            break;
        double payInput;
        scanf("%lf", &payInput);
        int Pay = static_cast<int>(payInput * 100);

        int dp[1001];
        for (int i = 0; i <= 1000; i++)
            dp[i] = INF;
        dp[0] = 0;
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < Coins[i]; j++)
            {
                for (int k = 1000; k >= CoinValue[i]; k--)
                {
                    if (dp[k - CoinValue[i]] != INF)
                    {
                        dp[k] = min(dp[k], dp[k - CoinValue[i]] + 1);
                    }
                }
            }
        }
        int dp2[1001];
        for (int i = 0; i <= 1000; i++)
            dp2[i] = INF;
        dp2[0] = 0;
        for (int i = 0; i < 6; i++)
        {
            for (int j = CoinValue[i]; j <= 1000; j++)
            {
                if (dp2[j - CoinValue[i]] != INF)
                {
                    dp2[j] = min(dp2[j], dp2[j - CoinValue[i]] + 1);
                }
            }
        }
        int result = INF;
        for (int i = Pay; i <= 1000; i++)
        {
            if (dp[i] != INF && dp2[i - Pay] != INF)
            {
                result = min(result, dp[i] + dp2[i - Pay]);
            }
        }
        if (result == INF)
        {
            printf("impossible\n");
        }
        else
        {
            printf("%d\n", result);
        }
    }
    return 0;
}