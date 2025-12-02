#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define MAXB 5
#define MAXK 6

typedef struct discount
{
    int j;
    int CK[2 * MAXB];
    int price;
} discount;
int dp[MAXK][MAXK][MAXK][MAXK][MAXK];
int mincost(int need[], int B, discount offers[], int S, int price[])
{
    int *p;
    if (B == 1)
        p = &dp[need[0]][0][0][0][0];
    else if (B == 2)
        p = &dp[need[0]][need[1]][0][0][0];
    else if (B == 3)
        p = &dp[need[0]][need[1]][need[2]][0][0];
    else if (B == 4)
        p = &dp[need[0]][need[1]][need[2]][need[3]][0];
    else
        p = &dp[need[0]][need[1]][need[2]][need[3]][need[4]];
    if (*p != -1)
        return *p;
    int res = 0;
    for (int i = 0; i < B; i++)
        res += need[i] * price[i];
    for (int s = 0; s < S; s++)
    {
        int ok = 1, newneed[MAXB];
        memcpy(newneed, need, sizeof(int) * B);
        for (int j = 0; j < offers[s].j; j++)
        {
            int c = offers[s].CK[2 * j];
            int k = offers[s].CK[2 * j + 1];
            if (newneed[c] < k)
            {
                ok = 0;
                break;
            }
            newneed[c] -= k;
        }
        if (ok)
        {
            int tmp = mincost(newneed, B, offers, S, price) + offers[s].price;
            if (tmp < res)
                res = tmp;
        }
    }
    *p = res;
    return res;
}

int main()
{
    memset(dp, -1, sizeof(dp));
    int B;
    scanf("%d", &B);
    int code[MAXB], need[MAXB], price[MAXB], code2idx[1000 + 1] = {0};
    for (int i = 0; i < B; i++)
    {
        scanf("%d %d %d", &code[i], &need[i], &price[i]);
        code2idx[code[i]] = i;
    }
    int S;
    scanf("%d", &S);
    discount offers[S];
    for (int i = 0; i < S; i++)
    {
        scanf("%d", &offers[i].j);
        for (int j = 0; j < offers[i].j; j++)
        {
            int c, k;
            scanf("%d %d", &c, &k);
            offers[i].CK[2 * j] = code2idx[c];
            offers[i].CK[2 * j + 1] = k;
        }
        scanf("%d", &offers[i].price);
    }
    int result = mincost(need, B, offers, S, price);
    printf("%d\n", result);
    return 0;
}