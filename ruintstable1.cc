/*
  Copyright (C) 2012 Dale Roberts

  This program is free software; you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation; either version 3 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
  details.

  You should have received a copy of the GNU General Public License along with
  this program; if not, see <http://www.gnu.org/licenses/>.
*/

#include <random>
#include <cmath>
#include <stdio.h>
#include <omp.h>

using namespace std;

inline double sinc(double x) {
    double ax = fabs(x);
    if (ax < 0.006) {
        if (x == 0.) return 1;
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
    double gamma = c/(1-rho) - p;
    int hits = 0;

    #pragma omp parallel reduction(+:hits)
    {
        int thread_seed = seed + omp_get_thread_num() + 1;
        std::mt19937 rng(thread_seed);
        std::uniform_real_distribution<> runif(0,1);
        std::exponential_distribution<> rexp(1);

        double cf = pow(cos(M_PI_2*rho),-1./rho);
        double sigma = pow(-h*c*cos(M_PI*rho/2.)*tgamma(-rho), 1./rho);
        double mu = -h*p;
        double U, W, V, Z, dX;

        #pragma omp for
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            double X = 0.0;

            while (s < t) {
                do {
                    U = runif(rng);
                    V = runif(rng);
                    do {
                        W = rexp(rng);
                    } while (W == 0.);
                    dX = mu + sigma*cf*pow(A(M_PI*U,rho)/pow(W,1.-rho),1./rho);
                } while (V > exp(-alpha*dX));

                s += h;
                X += dX;

                if (X > u) {
                    hits++;
                    break;
                }
            }
       }

     #ifdef DEBUG
        #pragma omp critical
        printf("%i %i\n", thread_seed, hits);
     #endif
     }

    double pr = hits / (double) n;

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
