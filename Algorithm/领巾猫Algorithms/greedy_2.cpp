#include <iostream>
#include <algorithm>
#include <climits>
using namespace std;

int Coins[6];
int CoinValue[6] = {5, 10, 20, 50, 100, 200};
int INF = INT_MAX;

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

        int result = INF;

        for (int extra = 0; extra <= 200; extra += 5)
        {
            int total = Pay + extra;
            int coinsUsed = 0;
            int tempCoins[6];
            for (int i = 0; i < 6; i++)
                tempCoins[i] = Coins[i];

            for (int i = 5; i >= 0; i--)
            {
                if (tempCoins[i] == 0)
                    continue;
                int canUse = min(tempCoins[i], total / CoinValue[i]);
                coinsUsed += canUse;
                total -= canUse * CoinValue[i];
                tempCoins[i] -= canUse;
            }
            if (total == 0)
            {
                int change = extra;
                int changeCoins = 0;

                for (int i = 5; i >= 0; i--)
                {
                    if (change == 0)
                        break;
                    int canUse = change / CoinValue[i];
                    changeCoins += canUse;
                    change -= canUse * CoinValue[i];
                }
                if (change == 0)
                {
                    result = min(result, coinsUsed + changeCoins);
                }
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