#include<bits/stdc++.h>
#define USERS 6000
#define MOVIES 4000
#pragma comment(linker, "/STACK:20000000")
#pragma comment(linker, "/HEAP:20000000")

using namespace std;



float ratings[USERS][MOVIES], orig[USERS][MOVIES];
float norm_ratings[USERS][USERS];
float cosim[USERS][USERS];
float avg[USERS];

float base[USERS][MOVIES];

float avg_movies[MOVIES];


/*
read_data- reads the values of user id, movie id and
ratings from the file and stores them in a matrix.
Then overall average and average rating
for a particular user is calculated.
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


        /*Calculation of avg of user and movies*/
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
        for(i=1;i<=1000;i++)
        {
            n=0;
            for(j=1;j<=1000;j++)
            {
                k = ratings[j][i];
                if(k>0){
                avg_movies[i]+=k;
                n++;
                }
            }
            if(n>0)
            avg_movies[i]/=n;

        }
        // updation of normalized ratings
        for(i=1;i<=1000;i++)
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
        // making the testing dataset 0.
        for(i=801;i<=1000;i++)
        {
            for(j=1;j<=1000;j++)
            {
                ratings[i][j] = 0;
            }

        }
        return 0;

}
/*
write_file- The final ratings calculated are
 written into the file "output_base.txt".
*/
void write_file()
{
    ofstream output;

    output.open ("output_base.txt");

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
    cout<<setprecision(10)<<fixed<<" RMS for baseline : " <<error << endl;
    cout<<setprecision(10)<<fixed<<" Spearman for baseline : " <<spearman << endl;
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
    cout<<setprecision(10)<<fixed<<" Precision on top 100 for Baseline :  " <<1.0 -error << endl;


}
/*
compute_sim- The baseline mean is calculated
which is the average of all the ratings.
Then the formula for baseline approach is used to
calculate the unknown ratings.
Heap is used to find the k nearest neighbours
for the prediction of similarities.
*/
int compute_sim(int knn_n)
{
    int i,j,k,n=0;
    float a,b,c,d;
    double baseline=0,base_mean=0;
    vector< pair<float,int> > knn;
    /*
    This part of the code calculates the cosine similarity between users.
    This is pre-calculated for better performance.

    */
    for(i=1;i<=1000;i++)
    {

        for(j=1;j<=1000;j++)
        {
            base_mean +=ratings[i][j];
            if(ratings[i][j]>0)n++;
        }
    }
    base_mean/=n;
    for(i=1;i<=1000;i++)
    {

        for(j=1;j<=1000;j++)
        {
            base[i][j] = base_mean + (avg[i] - base_mean) +(avg_movies[j] -base_mean);
        }
    }
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
            for(j=1;j<=799;j++)
            {
                if(ratings[i][j]==0)        // check on ratings weather it is filled or not
                    {
                            a=0;




                                for(k=1;k<=knn_n+1;k++)
                                    {
                                            if(i!=knn[k].second && ratings[knn[k].second][j]!=0)
                                            {
                                            ratings[i][j] += 1.0 * cosim[i][knn[k].second] * (ratings[knn[k].second][j] - base[knn[k].second][j]);
                                            a+=1.0 * cosim[i][knn[k].second];

                                            }

                                    }


                                    if(a!=0.0){      // no divide by zero error

                                            ratings[i][j]= (float)(ratings[i][j] / (float)(a));
                                            }

                                            ratings[i][j]+=base[i][j];
                                            if(ratings[i][j]<0)ratings[i][j]=0;
                                            else if(ratings[i][j]>5)ratings[i][j]=5;
                        }


            }
        }
        return 0;

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
    cout << "time for baseline : " << (stop_s-start_s)/(double(CLOCKS_PER_SEC)) << endl;
    return 0;
}
