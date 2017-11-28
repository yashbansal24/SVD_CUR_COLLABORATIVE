#include<bits/stdc++.h>
#define USERS 6000
#define MOVIES 4000
#pragma comment(linker, "/STACK:20000000")
#pragma comment(linker, "/HEAP:20000000")

using namespace std;
int maxid=0,maxud=0;

float ratings[MOVIES][USERS];
float sim[MOVIES][USERS];
float cosim[MOVIES][USERS];
float avg[MOVIES];
int read_data()
{

        std::ifstream file("ratings.dat");
        std::string   line;
        int i=0,j,n;
        float k;

        while(i<1000209)
        {

          int                 userid;
          int                 movieid;
          float         rating;
          float         val1;

          cin >> userid  >> movieid >> rating >> val1;
          //cout<<movieid<<endl;
          ratings[movieid][userid] = rating;
          maxid = max(maxid,movieid);
          maxud = max(maxud,userid);
            //cout<<ratings[movie]
          i++;
        }

                //cout<<ratings[661][1]<<endl;





        return 0;

}

void write_file()
{
    ofstream output;

    output.open ("A.csv");
    for(int i=1;i<=1000;i++)
    {
        for(int j=1;j<=1000;j++){
            output <<ratings[i][j]<<",";


            }

            output <<endl;
    }
    output.close();

}

int main()
{
   ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    read_data();

    write_file();
    return 0;
}
