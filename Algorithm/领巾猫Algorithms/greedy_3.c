#include <stdio.h>
int main()
{
    int n, k;
    scanf("%d %d", &n, &k);
    int station[k + 1];
    for (int i = 0; i < k + 1; i++)
    {
        scanf("%d", &station[i]);
    }
    for (int i = 0; i < k + 1; i++)
    {
        if (station[i] > n)
        {
            printf("No Solution");
            return 0;
        }
    }
    int current_fuel = n;
    int count = 0;
    int i = 0;
    while (i < k + 1)
    {
        if (current_fuel >= station[i])
        {
            current_fuel -= station[i];
            i++;
        }
        else
        {
            count++;
            current_fuel = n - station[i];
            i++;
        }
    }
    printf("%d", count);
    return 0;
}
