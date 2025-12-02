#include <stdio.h>
int main()
{
    double a;
    int b;
    double result=1;
    scanf("%lf %d",&a,&b);
    if(b>0)
    {    
        while(b>0)
        {
        result=result*a;
        b--;
        }
    printf("%.4f",result);
}
    else
    {
        b=-b;
        while(b>0)
        {
        result=result*a;
        b--;
        }
        result=1/result;
    printf("%.4f",result);  
    }
}