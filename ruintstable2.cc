#include <random>
#include <cmath>
#include <vector>
#include <stdio.h>
#include <omp.h>

using namespace std;

inline double sinc(double x) {
    double ax = fabs(x);
    if(ax < 0.006) {
	if(x == 0.) return 1;
	double x2 = x*x;
	if(ax < 2e-4)
	     return 1. - x2/6.;
	else return 1. - x2/6.*(1 - x2/20.);
    }
    /* else */
    return sin(x)/x;
}

inline double A(double x, double rho) {
  double Irho = 1.-rho;
  return pow(Irho*sinc(Irho*x),Irho)*pow(rho*sinc(rho*x),rho)/sinc(x);
}

double ruintstable(double u, double t, double rho, double c, double alpha, double p, double h, int n, int seed) {
    int hits = 0;
    double sum = 0;

    #pragma omp parallel reduction(+:hits) reduction(+:sum)
    {
        int thread_seed = omp_get_thread_num();
        std::mt19937 rng(seed + thread_seed + 1);
        std::uniform_real_distribution<> runif(0,1);
        std::exponential_distribution<> rexp(1);

        double cf = pow(cos(M_PI_2*rho),-1./rho);
        double sigma = pow(-h*c*cos(M_PI*rho/2.)*tgamma(-rho), 1./rho);
        double mu = -p*h;

        double U, W, Z, dX;

        #pragma omp for
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            double X = 0.0;
            bool hit = false;

            while (s < t) {
                // generate an increment of a stable process dX
                U = runif(rng);
                do {
                    W = rexp(rng);
                } while (W == 0.);
                dX = mu + sigma*cf*pow(A(M_PI*U,rho)/pow(W,1.-rho),1./rho);

                s += h;
                X += dX;

                if (X > u) {
                    // only record the first hit
                    if (!hit) hits++;
                    hit = true;
                }
            }

            // if hit then count this path
            if (hit)
                sum += exp(-alpha*X);

        }

     #ifdef DEBUG
        #pragma omp critical
        printf("%i %i\n", thread_seed, hits);
     #endif
    }

    sum = sum/n;

    double ee = exp(-(c*tgamma(-rho)*pow(alpha,rho)+p*alpha)*t);
    double pr = sum * ee;

//    printf("n: %i hits: %i sum: %f ee: %f pr: %f\n", n, hits, sum, ee, pr);

    return pr;
}

int main(int argc, char const *argv[])
{
    double h = argc > 1 ? atof(argv[1]): 0.01;
    int    n = argc > 2 ? atoi(argv[2]): 1<<13;
    int seed = argc > 3 ? atoi(argv[3]): 1234;

    double u = 0.1;
    double t = 5.;
    
    double rho   = 0.99; // stability index
    double c     = 0.01;
    double alpha = 1.0;

    double xi    = 0.2;
    double p = (1+xi)*(-c*rho*tgamma(-rho)*pow(alpha,rho-1));

    #ifdef DEBUG
    printf("u: %f t: %f seed: %i h: %f n: %i\n", u, t, seed, h, n);
    printf("rho: %f c: %f alpha: %f xi: %f p: %f\n", rho, c, alpha, xi, p);
    #endif


    if (rho == 1.) {
        printf("ERROR: rho = 1");
        exit(1);
    }

    double pr = ruintstable(u, t, rho, c, alpha, p, h, n, seed);

    printf("%f\n", pr);

    return 0;
}
