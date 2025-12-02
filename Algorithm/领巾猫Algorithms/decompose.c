#include <stdio.h>
#include <stdlib.h>
int decompose(int n)
{
    int cnt = 1;
    for (int i = 2; i < n; i++)
    {
        if (n % i == 0)
        {
            cnt += decompose(n / i);
        }
    }
    return cnt;
}

int main()
{
    int n;
    scanf("%d", &n);
    int ans = decompose(n);
    printf("%d\n", ans);
    return 0;
}