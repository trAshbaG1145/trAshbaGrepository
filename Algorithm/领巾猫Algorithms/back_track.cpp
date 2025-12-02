#include <stdio.h>
#define MAXN 20
int jc(int i)
{
    int res = 1;
    for (int k = 1; k <= i; k++)
        res = res * k;
    return res;
}

int max_sum = 0;
void dfs(int num, int pair[][MAXN], int i, int used[], int sum)
{
    if (i == num)
    {
        if (sum > max_sum)
            max_sum = sum;
        return;
    }
    for (int j = 0; j < num; j++)
    {
        if (!used[j])
        {
            used[j] = 1;
            dfs(num, pair, i + 1, used, sum + pair[i][j]);
            used[j] = 0;
        }
    }
}

int main()
{
    int num;
    scanf("%d", &num);
    int male[MAXN][MAXN], female[MAXN][MAXN], pair[MAXN][MAXN];
    for (int i = 0; i < num; i++)
        for (int j = 0; j < num; j++)
            scanf("%d", &male[i][j]);
    for (int i = 0; i < num; i++)
        for (int j = 0; j < num; j++)
            scanf("%d", &female[i][j]);
    for (int i = 0; i < num; i++)
        for (int j = 0; j < num; j++)
            pair[i][j] = male[i][j] * female[j][i];
    int used[num];
    for (int i = 0; i < num; i++)
        used[i] = 0;
    max_sum = 0;
    dfs(num, pair, 0, used, 0);
    printf("%d\n", max_sum);
    return 0;
}