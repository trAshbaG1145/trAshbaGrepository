#include <stdio.h>

int main()
{
    int a;
    int num=0;
    int prev=1;
    int preprev=0;
    scanf("%d",&a);
    if(a==1)
    {
        printf("1");
    }
    else if(a==0)
    {
        printf("0");
    }
    else {
        a=a-1;
        while(a)
        {
          num=preprev+prev;
          preprev=prev;
          prev=num;
          a--;  
        }
        printf("%d",num);
    }
}