#include <stdlib.h>
#include <stdio.h>

float last_progress;

void progress_bar(float progress, int length)
{     
    if ((int) (length*(progress - last_progress)) > 1)
    {   
        int i = 0;
        char * string = (char *) malloc( sizeof(char) * (length+1));

        for (i = 0; i < length; i++)
        {
            if (i == 0 || i == length - 1)
            {
                string[i] = '|';
            }
            else if (i < (int) (progress * length))
            {
                string[i] = '#';
            }
            else
            {
                string[i] = ' ';
            }
            string[length] = '\0';
        }
        printf("\33[2K\r%s %d 0/0", string, (int)(100*progress));
        fflush(stdout); 
        last_progress = progress;
    }
}