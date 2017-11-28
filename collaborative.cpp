#include<bits/stdc++.h>
#include<ctime>
#define USERS 6000
#define MOVIES 4000
#pragma comment(linker, "/STACK:20000000")
#pragma comment(linker, "/HEAP:20000000")

using namespace std;





float ratings[USERS][MOVIES], orig[USERS][MOVIES];
float norm_ratings[USERS][USERS];
float cosim[USERS][USERS];
float avg[USERS];

/*

read_data- This function is used to read the values of
user id, movie id and ratings from the file named "ratings.dat" and
to compute the average ratings for the movies.

*/
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
          ratings[userid][movieid] = orig[userid][movieid] = rating;
          i++;
        }

                cout<<"data read properly ... "<<endl;

// .... avg calculation of each movie
        for(i=1;i<=1000;i++)
        {
            n=0;
            for(j=1;j<=1000;j++)
            {
                k = ratings[i][j];
                if(k>0){
                avg[i]+=k;
                n++;
                }
            }
            if(n>0)
            avg[i]/=n;

        }
        for(i=1;i<1001;i++)
        {
            for(j=1;j<=1000;j++)
            {
                k=ratings[i][j];
                if(k>0)
                    norm_ratings[i][j]= k - avg[i];
                else
                    norm_ratings[i][j] = 0;
            }

        }
        for(i=800;i<1001;i++)
        {
            for(j=1;j<=1000;j++)
            {
                ratings[i][j] = 0;
            }

        }

        return 0;

}
/*
write_file- This is used to write the ouput to the final named "output_coll.txt"

*/
void write_file()
{
    ofstream output;

    output.open ("output_coll.txt");
    for(int i=800;i<=1000;i++)
    {
        for(int j=1;j<=1000;j++){
            output <<setprecision(15) <<fixed << (ratings[i][j])<<" ";


            }

            output <<endl;
    }
    output.close();
    float error=0,spearman=0,n2=0;
    n2 = 0;

    for(int i=800;i<=1000;i++)
    {
        for(int j=1;j<=1000;j++){


            if(orig[i][j]!=0){
            spearman+=(ratings[i][j] - orig[i][j])*(ratings[i][j] - orig[i][j]);
         error+=pow(ratings[i][j] - orig[i][j],2);
         n2++;
            }

            }
    }

    spearman = 6.0*spearman / n2;
    spearman /= (n2*n2 -1.0);
    spearman = 1 - spearman;
    error = sqrt(error);
    error/=n2;
    //error/=(1000.0);

    cout<<setprecision(10)<<fixed<<" RMS for collaborative : " <<error << endl;
    //output<<" precision on top k " << (1- error)<<endl;
    cout<<setprecision(10)<<fixed<<" Spearman for collaborative : " <<spearman << endl;
    error=0;
    for(int i=801;i<=900;i++)
    {
        for(int j=1;j<=1000;j++){
        if(orig[i][j]!=0)
         error+=pow(ratings[i][j] - orig[i][j],2);

            }
    }
    error=sqrt(error);
    error/=100.0;
    error/=1000.0;
    cout<<setprecision(10)<<fixed<<" Precision on top 100 for Collborative :  " <<1.0 -error << endl;

}
/*
compute_sim- This function computes the cosine similarities for a given movie and
user which is then used to calculate the final ratings.
*/
int compute_sim(int knn_n)
{
    int i,j,k;
    float a,b,c,d;
    vector< pair<float,int> > knn;
    /*
    This part of the code calculates the cosine similarity between users.
    This is pre-calculated for better performance.

    */
    for(i=1;i<=1000;i++)
        {
            for(j=1;j<=1000;j++)
            {
                a=b=0;
                for(k=1;k<=1000;k++)
                {
                    c=norm_ratings[i][k];
                    d = norm_ratings[j][k];
                        cosim[i][j] += c * d;
                        a+= c*c;
                        b+=d*d;


                }
                if(a>0 && b>0){
                cosim[i][j] /= sqrt(a);
                cosim[i][j] /= sqrt(b);
                }

            }
        }
        for(i=800;i<=1000;i++)
        {
            /*

            Knn is precomputed for better performances.
            We used inbuilt heap data structure in cpp.

            */
            knn.clear();
            for(k=1;k<=799;k++)
                            {
                                knn.push_back(make_pair(cosim[i][k],k));
                            }
                            make_heap(knn.begin(),knn.end());

            for(j=1;j<=1000;j++)
            {
                a=0;

                    for(k=1;k<800;k++)
                        {

                                if((cosim[i][k]>0 &&  i!=k && ratings[i][j]==0 && ratings[k][j]!=0))    // check on ratings weather it is filled or not
                                {

                                    ratings[i][j] += 1.0 * cosim[i][k] * ratings[k][j];
                                    a+=1.0 * cosim[i][k];

                                }

                        }

                        if(a!=0.0)              // no divide by zero error
                            {
                                ratings[i][j]= (float)(ratings[i][j] / (float)(a));
                            }
            }
        }

}

int main()
{
   ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    read_data();
    int start_s=clock();

    compute_sim(99);
    write_file();
    int stop_s=clock();
    cout << "time for collaborative : " << (stop_s-start_s)/(double(CLOCKS_PER_SEC)) << endl;

    return 0;
}
