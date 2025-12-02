#include <stdio.h>
int main() 
{
    int n;
    scanf("%d", &n);
    int array[n];
    for (int i = 0; i < n; i++) 
    {
        scanf("%d", &array[i]);
    }
    int times[n];
    for(int k = 0; k < n; k++)
    {
        int nums = array[k];
        int count = 0;
        for(int m = 0; m < n; m++)
        {
            if(array[m] == nums)
            {
                count++;
            }
        }
        times[k] = count;
    }
    int max = 0;
    int result = array[0];
    for(int i = 0; i < n; i++)
    {
        if(times[i] > max)
        {
            max = times[i];
            result = array[i];
        }
    }
    printf("%d", result);
    return 0;
}