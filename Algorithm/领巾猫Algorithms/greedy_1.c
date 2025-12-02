#include <stdio.h>
typedef struct time
{
    int start_time;
    int stop_time;
    int solved;
} time;

int main()
{
    int count = 0;
    int n;
    scanf("%d", &n);
    time time[n];
    for (int i = 0; i < n; i++)
    {
        scanf("%d %d", &time[i].start_time, &time[i].stop_time);
        time[i].solved = 0;
    }
    for (int i = 0; i < n; i++)
    {
        if (time[i].solved == 1)
        {
            continue;
        }
        else
        {
            count++;
            time[i].solved = 1;
            int current_stop = time[i].stop_time;
            for (int q = 0; q < n; q++)
            {
                if (time[q].solved == 1)
                {
                    continue;
                }
                else
                {
                    if (time[q].start_time > current_stop)
                    {
                        time[q].solved = 1;
                        current_stop = time[q].stop_time;
                    }
                }
            }
        }
    }
    printf("%d", count);
}