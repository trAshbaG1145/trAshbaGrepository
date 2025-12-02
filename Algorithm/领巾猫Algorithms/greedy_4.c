#include <stdio.h>

int main()
{
    int num;
    scanf("%d", &num);
    if (num < 2)
    {
        printf("0\n");
        return 0;
    }
    int res[1000];
    int count = 0;
    int sum = 0;
    int current = 2;
    while (sum + current <= num)
    {
        res[count] = current;
        sum += current;
        count++;
        current++;
    }
    int rem = num - sum;
    for (int i = count - 1; i >= 0 && rem > 0; i--)
    {
        res[i] += 1;
        rem--;
    }
    int result = 1;
    for (int i = 0; i < count; i++)
    {
        result *= res[i];
    }
    printf("%d\n", result);
    return 0;
}