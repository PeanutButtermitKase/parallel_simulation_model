/*
Author @ Nydia R. Varela-Rosales
run as: gcc -fopenmp -o LJparallel LJparallel.c
env OMP_NUM_THREADS=8 ./LJparallel | > out.pos
*/
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <omp.h>
#include<iostream>
using namespace std;
/* properties of the system */
const double rc   = 3.0;  // cutoff radius 
const double rcp  = 2.5;  // smooth cut off
int     N;           // number of particles 
double  rho;         // density 
double  L;           // box size
double  dt;          //  time step 
double  halfDt;      // integration time step 
double  runtime;     // simulation time
long    seed;        // random seed 
double  burninTime;  // start sampling
double  K;           // KE
double  U;           // PE
double  H;           // Total energy
double  T;           /* temperature normalized

int thread_count = 2;
/* structure for the properties of one atoms */
struct Atom
{
   double  rx, ry, rz;  
   double  px, py, pz;  
   double  fx, fy, fz;  
};

/* function to set up cubic lattice */
double latticex, latticey, latticez;
void makeLatticePosition(double a)
{
   static int i = 0;
   static int j = 0;
   static int k = 0;
   latticex = i*a - 0.5*L;
   latticey = j*a - 0.5*L;
   latticez = k*a - 0.5*L;
   i = i + 1;
   if ( i*a > L - 1.0e-6 )  
   {
      i = 0;                 
      j = j + 1;             

      if ( j*a > L - 1.0e-3 )  
      {
         j = 0;               
         k = k + 1;            

         if ( k*a > L - 1.0e-6 )  
         {
            i = 0;                
            j = 0;
            k = 0;
         }
      }
   }
}

double makePeriodic(double u)
{
   while ( u < -0.5*L )
   {
      u = u + L;
   }

   while ( u >= 0.5*L )
   {
      u = u - L;
   }

   return u;
}


void getForce(struct Atom atoms[])
{
   int     i, j;                       
   double  dx, dy, dz;                 
   double  r, r2, r2i, r6i;            
   double  fij;                        
   double  eij;                       
   double  x, alpha, dalpha;
   U = 0;

   for ( i = 0; i < N; i = i + 1 )
   {
      atoms[i].fx = 0;
      atoms[i].fy = 0;
      atoms[i].fz = 0;
   }
   // set-up variable for shared memory
   #pragma omp parallel for shared(N,dx, dy, dz,r, r2, r2i, r6i,fij,eij,x,alpha,dalpha) private(i)
   for ( i = 0; i < N-1; i = i + 1 )
   {
     #pragma omp parallel for shared(N,dx, dy, dz,r, r2, r2i, r6i,fij,eij,x,alpha,dalpha) private(j)
      for ( j = i+1; j < N; j = j + 1 )
      {
         
         dx = makePeriodic(atoms[i].rx - atoms[j].rx);
         dy = makePeriodic(atoms[i].ry - atoms[j].ry);
         dz = makePeriodic(atoms[i].rz - atoms[j].rz);
         r2 = dx*dx + dy*dy + dz*dz;
         if ( r2 < rc*rc )
         {
            r2i = 1/r2;        // LJ definition
            r6i = r2i*r2i*r2i; // LJ definition
            fij = 48*r2i*r6i*(r6i-0.5);
            eij = 4*r6i*(r6i-1);
            if ( r2 > rcp*rcp )
            {
              // sample energy
               r      = sqrt(r2);
               x      = (2*r - rcp - rc)/(rcp - rc);
               alpha  = 0.5 - 0.25*x*(x*x - 3);
               dalpha = 1.5*(x*x - 1)/(r*(rcp-rc));
               fij    = alpha*fij + dalpha*eij;
               eij    = alpha*eij;
            }
            // update forces
            atoms[i].fx = atoms[i].fx + fij*dx;
            atoms[i].fy = atoms[i].fy + fij*dy;
            atoms[i].fz = atoms[i].fz + fij*dz;
            atoms[j].fx = atoms[j].fx - fij*dx;
            atoms[j].fy = atoms[j].fy - fij*dy;
            atoms[j].fz = atoms[j].fz - fij*dz;
            U = U + eij;
         }
      }
   }

}

// seed generator
double gaussian()
{
   static int    have = 0;
   static double x2;
   double fac, y1, y2, x1;
   if ( have == 1 ) 
   {
      have = 0;
      return x2;
   }
   else
   {
      y1  = drand48();
      y2  = drand48();
      fac = sqrt(-2*log(y1));
      have = 1;
      x1 = fac*sin(2*M_PI*y2); 
      x2 = fac*cos(2*M_PI*y2); 
      return x1;               
   }
}

/* function to initialize the system */
void initialize(struct Atom atoms[])
{
  // ofstream myfile;
  // myfile.open ("lattice.pos");
   double  scale, a;
   int     i;

   /* generate positions */
   a = L/(int)(cbrt(N)+0.99999999999); /* lattice distance */

   for ( i = 0; i < N; i = i + 1 )
   {
      makeLatticePosition(a);
      atoms[i].rx = latticex;
      atoms[i].ry = latticey;
      atoms[i].rz = latticez;
   }

   /* generate momenta */
   srand48(seed);  /* initialized the random number generator used in gaussian */
   scale = sqrt(T);
   K     = 0;

   for ( i = 0; i < N; i = i + 1 )
   {
      atoms[i].px = scale*gaussian();
      atoms[i].py = scale*gaussian();
      atoms[i].pz = scale*gaussian();
      K = K
         + atoms[i].px*atoms[i].px
         + atoms[i].py*atoms[i].py
         + atoms[i].pz*atoms[i].pz;
   }

   T = K/(3*N);
   K = K/2;
   getForce(atoms);
   H = U + K;
}

/* Verlet integration step */
void integrateStep(struct Atom atoms[])
{
   int i;
   /* half-force step */
   for ( i = 0; i < N; i = i + 1 )
   {
      atoms[i].px = atoms[i].px + 0.5*dt*atoms[i].fx;
      atoms[i].py = atoms[i].py + 0.5*dt*atoms[i].fy;
      atoms[i].pz = atoms[i].pz + 0.5*dt*atoms[i].fz;
   }

   /* full free motion step */
   for ( i = 0; i < N; i = i + 1 )
   {
      atoms[i].rx = atoms[i].rx + dt*atoms[i].px;
      atoms[i].ry = atoms[i].ry + dt*atoms[i].py;
      atoms[i].rz = atoms[i].rz + dt*atoms[i].pz;
      
      cout <<"sphere 1 0x00748A "<< atoms[i].rx << " "<<  atoms[i].ry <<" "<<atoms[i].rz << " \n";
   }
   cout <<"eof\n";


   /* positions were changed, so recompute the forces */
   getForce(atoms);

   /* final force half-step */
   K = 0;

   for ( i = 0; i < N; i = i + 1 )
   {
      atoms[i].px = atoms[i].px + 0.5*dt*atoms[i].fx;
      atoms[i].py = atoms[i].py + 0.5*dt*atoms[i].fy;
      atoms[i].pz = atoms[i].pz + 0.5*dt*atoms[i].fz;
      K = K
         + atoms[i].px*atoms[i].px
         + atoms[i].py*atoms[i].py
         + atoms[i].pz*atoms[i].pz;
   }

   /* finish computing T, K, H */
   T = K/(3*N);
   K = K/2;
   H = U + K;
}

/* integration and measurement */
void run()
{
   struct  Atom atoms[N];
   int     numSteps    = (int)(runtime/dt + 0.5);
   int     burninSteps = (int)(burninTime/dt + 0.5);
   int     count;          /* counts time steps */
   int     numPoints = 0;  /* counts measurements */
   double  sumH      = 0;  /* total energy accumulated over steps */
   double  sumH2     = 0;  /* total energy squared accumulated */
   double  avgH, avgH2, fluctH;  /* average energy, square, fluctuations */

   /* draw initial conditions */
   initialize(atoms);

   /* perform burn-in steps */
   for ( count = 0; count < burninSteps; count = count + 1 )
   {
      integrateStep(atoms);
   }

   /* perform rest of time steps */
   for ( count = burninSteps; count < numSteps; count = count + 1 )
   {
      /* perform integration step */
      integrateStep(atoms);

      /* accumulate energy and its square */
      sumH  = sumH + H;
      sumH2 = sumH2 + H*H;
      numPoints = numPoints + 1;

      /* determine averages and fluctuations */
      avgH   = sumH/numPoints;
      avgH2  = sumH2/numPoints;
      fluctH = sqrt(avgH2 - avgH*avgH);
   }
}

/* main program */
int main()
{

  N = 500;
  rho = 0.23;
  T = 0.2;
  dt = 0.01;
  runtime = 100;
  seed = 10;
  burninTime = 10;
  
   L = 5;
   /* run the simulation */
   double start = omp_get_wtime();
   run();
   double end = omp_get_wtime();
   printf("Simulation time %f seconds\n", end - start);
   return 0;
}
