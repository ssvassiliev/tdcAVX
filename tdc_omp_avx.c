// Computes Full Coulomb Coupling 
// WRITTEN BY: Sergey Vassiliev, Daniel Oblinsky
// LAST MODIFIED: March 22, 2013
// GCC FLAGS: -fopenmp -msse3 -lm -Ofast
// ICC FLAGS: -openmp -xHost 
/*-----------------------------------------------------------------*/
// Changes:
// June 2017
// Rewritten to support AVX instructions set
// Automatic CPU cache size determination
/******* July 2008 *******/
// added check for noncollinear cubes
// renamed some variables
// modified main loop
// removed AtomicMass from CUBE structure - it is junk
// added lookup table for scaling to experimental dipoles
/****** Feb 2013 *******/
// Added support of noncollinear cubes
// Bugfix: both 0/1 cubes were scaled using mu0 instead of mu0 and mu1
// Added cutoff for reading in charges: Qcutoff

// Known issues:
// -  Intel compiled binary may segfault with some cubes
//  Added printout of Block & Thread from the inner loop to fix this
/*------------------------------------------------------------------*/

/********   CONSTANTS AND UNITS  ************/ 

#define Bohr2Ang 0.5291772108
#define echarge 1.602176E-19 
#define ep0 8.854188e-12
#define HtoeV 27.2116
#define JtoHartree 4.3597482e-18
#define eVtocm 8065.
#define eAngtoDebye 4.8032124

#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <sys/timeb.h>
#include <unistd.h>
//#include <xmmintrin.h>
//#include <pmmintrin.h>
#include <immintrin.h>


/******** Function prototypes ************/

void   ReadGaussianCube(int, int );
double CalcCoulombCoupling(int,int,int);
void   WriteGaussianCube(int );
double CalcDipole(int, double *);
int    ReadCubeInfo(char *);
double LookupDipoleMoment(int, char*);
inline static double vec_dot(double *,double *);
inline static void vec_diff(double *, double *, double *);
inline static void vec_scale(double *, double, double *);
inline static void Line(int);

#define MAX_NFILES 2
#define NCUB 2

/************************** GLOBAL VARIABLES **********************/
struct GAU_CUBE { /*GAUSSIAN CUBE */
  /************* VARIABLES  READ FROM THE CUBE FILE ***************/
  int    NumAtoms; // Number of atoms
  int v_count;
  short int last_v_element_count;
  double OriginX, OriginY, OriginZ; // Origin of the cube
  int    MX,MY,MZ;  // Number of voxels along XYZ directions
  double X_X, X_Y, X_Z; // Axis X vector
  double Y_X, Y_Y, Y_Z; // Axis Y vector
  double Z_X, Z_Y, Z_Z; // Axis Z vector
  double X_Length, Y_Length, Z_Length; // Length of each vector
  double Vol; // Volume correction factor
  __m128* AtomicXYZ_Num;
  __m256 *rX, *rY, *rZ;
  __m256 *rQ;
  
  double Dipole;
  double center[3];
  /*************** CALCULATED VARIABLES *******************/
  double sum, abssum;     // Sum of transition densities 
  // Coordinates of the cube origin corrected for the shift of center of mass
  long   NumOfVoxels;     // This is the MX * MY * MZ
} cube[NCUB];

/** The Global Matrix of Coulomb couplings **/
double VCoupl[MAX_NFILES][MAX_NFILES]; 
double Rcutoff, Qcutoff;

/********** VARIABLES  READ FROM CUBE INFO FILE ************/
char FileName[MAX_NFILES][30];
char RESIDUE_NAME[MAX_NFILES][4];
char Transition[MAX_NFILES][4];

int main(int argc, char** argv) {
  
  int i, NCubes;
  int cache_size;
  // TIMING 
  struct  timeb start, end;
  double elapsed_t;
  int c;
  char *cubinfo = "cube.inf";  
  Qcutoff = 0.0;

  char usage[]="\n"
    " -------------------------------------------------------------------\n"
    " * This is the program to compute full Coulomb coupling between    *\n"
    " * two transition density cubes                                    *\n"
    " * Example usage: tdc [-f cube.inf -c 1e-7 -s 2.0]                 *\n"
    " * Optional arguments:                                             *\n" 
    " * -f: name of the file containing cube information                *\n"
    " * -c: charge density cutoff value                                 *\n"
    " * -s: CPU L2+L3 cache size (KB), approximate parameter!            *\n"
    " -------------------------------------------------------------------\n"
    "  Defaults: -f = cube.inf \n"                                       
    "            -c = 0.0     \n"           
    "            -s = automatic     \n"           
    "  To run on n threads set OMP_NUM_THREADS=n \n"            
    " ------------------------------------------------------------------\n\n";

// Get CPU L3 cache size
  system("grep 'cache size' /proc/cpuinfo | head -n 1 | awk '{print $4}' > cache_size");
  FILE *fp=fopen("cache_size","rt");
  fscanf(fp,"%i",&cache_size);
  fclose(fp);
  system("rm -f cache_size");
  NCubes=ReadCubeInfo(cubinfo);
  
 opterr=0;
  while( (c=getopt(argc,argv,"c:f:s:h")) != -1) 
    {
      switch(c) 
	{
	case 'f': 
	  cubinfo=optarg;
	  break;
	case 'c': 
	  Qcutoff=atof(optarg);
	  break;
	case 's': 
	  cache_size=atof(optarg);
	  break;
	case 'h':
	  printf("%s",usage);
	  exit(0);
	}
    }

  
  // Check if dipole moments are known
  fprintf(stderr,"Experimental dipole moments:\n");
  for(i=0;i<NCubes;i++){
    cube[i].Dipole=LookupDipoleMoment(i,Transition[i]);
    fprintf(stderr,"%i:%s/%s=%.3f Debye\n",i, RESIDUE_NAME[i],
	    Transition[i],cube[i].Dipole);
  }
  fprintf(stderr,"\n");

  fprintf(stderr,"Note: charges will be divided by sqrt(2)\n");
  fprintf(stderr,"This is required only for gaussian cubes\n");	
  fprintf(stderr,"Using cutoff for charges: %e\n", Qcutoff);		 
  ReadGaussianCube(0,0);
  ReadGaussianCube(1,1);
  Line(60);

  ftime(&start);	  
  VCoupl[0][1]=CalcCoulombCoupling(0,1,cache_size);
  ftime(&end); 

  elapsed_t=(1000.0*(end.time-start.time)+(end.millitm-start.millitm))/1000;
  fprintf(stderr, "\n Job CPU time: %.3f sec\n", elapsed_t);
  fprintf(stderr,"\n");
  return 1; 
  
}


double LookupDipoleMoment(int FNum, char *Transition)
{ // Lookup table of experimental dipole moments
  // char* Transition selects QY or QX transition for Chl or Pheo
  // QX transitions rapidly relax, they should not be considered as donors

  if(!strncmp(Transition,"Qy",2)) 
    {
      if(!strncmp(RESIDUE_NAME[FNum],"CLA",3)) 
	return(6.24); // Qy Shipman & Housman, 1979, corrected to protein??
      if(!strncmp(RESIDUE_NAME[FNum],"PHO",3))
	return(4.18); // Qy CLA*0.67
      if(!strncmp(RESIDUE_NAME[FNum],"BCL",3)) 
	return(6.13); // Qy Sauer,K. et al., J.Am.Chem.Soc., 1966,88,2681-2688
    }
  
  if(!strncmp(Transition,"Qx",2)) 
    {
      if(!strncmp(RESIDUE_NAME[FNum],"BCL",3)) 
	return(3.29); // Qx Sauer,K. et al., J.Am.Chem.Soc., 1966,88,2681-2688
    }
  
  if(!strncmp(RESIDUE_NAME[FNum],"RG1",3)) 
    if(!strncmp(Transition,"S2",2)) 
      return(13.0); // S2 Anderson,P.O., Photochem.Photobiol.,1991,54,353-360
  
  fprintf(stderr,"*** ERROR *** Unknown dipole moment %i:%s %s\n",
	  FNum + 1,RESIDUE_NAME[FNum],Transition);
  exit(0);
}


int ReadCubeInfo(char *Fname)
{
  FILE *Centres;
  int i;
  

  if (!(Centres=fopen(Fname,"rt")))
    {fprintf(stderr,"\n*** ERROR *** Centres file not found\n");exit(0);}
  i=0;
  while(fscanf(Centres,"%s",&FileName[i][0])!=EOF) {
    fscanf(Centres,"%s",&RESIDUE_NAME[i][0]);
    fscanf(Centres,"%s",&Transition[i][0]);
    i++;  
  }
  fclose(Centres);
  return(i);
}



void ReadGaussianCube(int FNum,int Det)
{
  // Function reads gausian cube file 
  // File name is taken from the array of filenames 
  // created by ReadCubeCentres function; FNum is the
  // number of cube file in this array
  FILE *GaussianCube;
  char line_buf[85];
  long i, j, ix, iy, iz, charged, maxit;
  double junk;  
 
 double correct, alpha, cutfactor, cut, abs_Q, prevsum,diffsum,prevdiffsum;
 double min_Q=10.0,max_Q=.0;

  if(!(GaussianCube=fopen(FileName[FNum],"rt"))){
    fprintf(stderr,"*** ERROR *** Cube file %s not found\n",FileName[FNum]);exit(0);}
  
  // Skip 2 first lines
  fgets(line_buf,82,GaussianCube);
  fgets(line_buf,82,GaussianCube);
  
  // Number of atoms and xyz of the origin (not center) of the cube
  fscanf(GaussianCube,"%d%lf%lf%lf",					\
	 &cube[Det].NumAtoms,&cube[Det].OriginX,&cube[Det].OriginY,&cube[Det].OriginZ);
  
  cube[Det].AtomicXYZ_Num = (__m128*)malloc(cube[Det].NumAtoms*sizeof(__m128));

  /***** Number of voxels and distance increment in XYZ  directions ******/
  // In the X axisY_axis_VecY
  fscanf(GaussianCube,"%d%lf%lf%lf",					\
	 &cube[Det].MX, &cube[Det].X_X, &cube[Det].X_Y, &cube[Det].X_Z);
  // In the Y axis
  fscanf(GaussianCube,"%d%lf%lf%lf",					\
	 &cube[Det].MY,&cube[Det].Y_X,&cube[Det].Y_Y,&cube[Det].Y_Z);
  // In the Z axis
  fscanf(GaussianCube,"%d%lf%lf%lf",					\
	 &cube[Det].MZ,&cube[Det].Z_X,&cube[Det].Z_Y,&cube[Det].Z_Z);
  
  cube[Det].NumOfVoxels = cube[Det].MX*cube[Det].MY*cube[Det].MZ;  
  int ATM;
  double X,Y,Z;

  // Read atoms and the cartesian coordinates of atomic centres
  // Don't comment this out !!! This will lead to reading coordinatates into charges!
  // I did it twice ... and wasted a lot of time figuring why charges are wrong.
  for(i=0;i<cube[Det].NumAtoms;i++) {
  	fscanf(GaussianCube,"%d%lf%lf%lf%lf", &ATM, &junk, &X, &Y, &Z);	cube[Det].AtomicXYZ_Num[i] = _mm_set_ps((float)ATM,(float)Z,(float)Y,(float)X);
  }
  
  // Convert cube distance units from Bohr to Angstrom
  cube[Det].OriginX *= Bohr2Ang;
  cube[Det].OriginY *= Bohr2Ang;
  cube[Det].OriginZ *= Bohr2Ang;
  cube[Det].X_Length = sqrt(pow(cube[Det].X_X,2) + pow(cube[Det].X_Y,2) + pow(cube[Det].X_Z,2));
  cube[Det].Y_Length = sqrt(pow(cube[Det].Y_X,2) + pow(cube[Det].Y_Y,2) + pow(cube[Det].Y_Z,2));
  cube[Det].Z_Length = sqrt(pow(cube[Det].Z_X,2) + pow(cube[Det].Z_Y,2) + pow(cube[Det].Z_Z,2));
  cube[Det].Vol=cube[Det].X_Length*cube[Det].Y_Length*cube[Det].Z_Length;
  cube[Det].X_X *= Bohr2Ang;
  cube[Det].X_Y *= Bohr2Ang;
  cube[Det].X_Z *= Bohr2Ang;
  cube[Det].Y_Y *= Bohr2Ang; 
  cube[Det].Y_X *= Bohr2Ang; 
  cube[Det].Y_Z *= Bohr2Ang; 
  cube[Det].Z_Z *= Bohr2Ang;
  cube[Det].Z_X *= Bohr2Ang;
  cube[Det].Z_Y *= Bohr2Ang;
  
  // Read transition density, accumulate sum, determine minimal charge (excluding zeros)
  cube[Det].sum=0; 
  charged=0;
  i=0;
  double q_charge;
  float tmp_vec[4][8]  __attribute__ ((aligned (32)));
  long v_element_count = 0;
  long v_count = 0;

  cube[Det].rX = aligned_alloc(32, (cube[Det].NumOfVoxels+8) * sizeof(float));
  cube[Det].rY = aligned_alloc(32, (cube[Det].NumOfVoxels+8) * sizeof(float));
  cube[Det].rZ = aligned_alloc(32, (cube[Det].NumOfVoxels+8) * sizeof(float));
  cube[Det].rQ = aligned_alloc(32, (cube[Det].NumOfVoxels+8) * sizeof(float));

  for (ix=0;ix<cube[Det].MX;ix++) 
    for (iy=0;iy<cube[Det].MY;iy++) 
      for (iz=0;iz<cube[Det].MZ;iz++) 
	{
	  fscanf(GaussianCube,"%lg",&q_charge);
          q_charge *= cube[Det].Vol;
	  // if cube is made by gaussian divide by sqrt(2) 
	  q_charge *= 0.707106781;
	  abs_Q=fabs(q_charge);
	  
	  if(abs_Q > Qcutoff) {
	    tmp_vec[0][v_element_count] = cube[Det].OriginX + ix*cube[Det].X_X + iy*cube[Det].Y_X + iz*cube[Det].Z_X;
	    tmp_vec[1][v_element_count] = cube[Det].OriginY + ix*cube[Det].X_Y + iy*cube[Det].Y_Y + iz*cube[Det].Z_Y;
	    tmp_vec[2][v_element_count] = cube[Det].OriginZ + ix*cube[Det].X_Z + iy*cube[Det].Y_Z + iz*cube[Det].Z_Z;
	    tmp_vec[3][v_element_count] = q_charge;
	    cube[Det].sum += q_charge;
	    charged++;
	    v_element_count++;
	    if ( v_element_count == 8 ) {
	     cube[Det].rX[v_count] = _mm256_set_ps(tmp_vec[0][7],tmp_vec[0][6],tmp_vec[0][5],tmp_vec[0][4],tmp_vec[0][3],tmp_vec[0][2],tmp_vec[0][1],tmp_vec[0][0]); 
	      cube[Det].rY[v_count] = _mm256_set_ps(tmp_vec[1][7],tmp_vec[1][6],tmp_vec[1][5],tmp_vec[1][4],tmp_vec[1][3],tmp_vec[1][2],tmp_vec[1][1],tmp_vec[1][0]);
	      cube[Det].rZ[v_count] = _mm256_set_ps(tmp_vec[2][7],tmp_vec[2][6],tmp_vec[2][5],tmp_vec[2][4],tmp_vec[2][3],tmp_vec[2][2],tmp_vec[2][1],tmp_vec[2][0]);
	      cube[Det].rQ[v_count] = _mm256_set_ps(tmp_vec[3][7],tmp_vec[3][6],tmp_vec[3][5],tmp_vec[3][4],tmp_vec[3][3],tmp_vec[3][2],tmp_vec[3][1],tmp_vec[3][0]);
	      v_count++;
	      v_element_count=0;
	      tmp_vec[0][0] = tmp_vec[0][1] = tmp_vec[0][2] = tmp_vec[0][3] = 0.0f;
	      tmp_vec[0][4] = tmp_vec[0][5] = tmp_vec[0][6] = tmp_vec[0][7] = 0.0f;
	      tmp_vec[1][0] = tmp_vec[1][1] = tmp_vec[1][2] = tmp_vec[1][3] = 0.0f;
	      tmp_vec[1][4] = tmp_vec[1][5] = tmp_vec[1][6] = tmp_vec[1][7] = 0.0f;
	      tmp_vec[2][0] = tmp_vec[2][1] = tmp_vec[2][2] = tmp_vec[2][3] = 0.0f;
	      tmp_vec[2][4] = tmp_vec[2][5] = tmp_vec[2][6] = tmp_vec[2][7] = 0.0f;
	      tmp_vec[3][0] = tmp_vec[3][1] = tmp_vec[3][2] = tmp_vec[3][3] = 0.0f;
	      tmp_vec[3][4] = tmp_vec[3][5] = tmp_vec[3][6] = tmp_vec[3][7] = 0.0f;
	    }
	    if(abs_Q < min_Q) min_Q=abs_Q;
	    if(abs_Q > max_Q) max_Q=abs_Q;
	  } 
	  i++;
	}

  cube[Det].last_v_element_count = v_element_count;

  if ( v_element_count !=0 ) {
	      cube[Det].rX[v_count] = _mm256_set_ps(tmp_vec[0][7],tmp_vec[0][6],tmp_vec[0][5],tmp_vec[0][4],tmp_vec[0][3],tmp_vec[0][2],tmp_vec[0][1],tmp_vec[0][0]); 
	      cube[Det].rY[v_count] = _mm256_set_ps(tmp_vec[1][7],tmp_vec[1][6],tmp_vec[1][5],tmp_vec[1][4],tmp_vec[1][3],tmp_vec[1][2],tmp_vec[1][1],tmp_vec[1][0]);
	      cube[Det].rZ[v_count] = _mm256_set_ps(tmp_vec[2][7],tmp_vec[2][6],tmp_vec[2][5],tmp_vec[2][4],tmp_vec[2][3],tmp_vec[2][2],tmp_vec[2][1],tmp_vec[2][0]);
	      cube[Det].rQ[v_count] = _mm256_set_ps(tmp_vec[3][7],tmp_vec[3][6],tmp_vec[3][5],tmp_vec[3][4],tmp_vec[3][3],tmp_vec[3][2],tmp_vec[3][1],tmp_vec[3][0]);
    v_count++;
  }
 
  cube[Det].v_count = v_count;
  // don't FORGET ABOUT V_count
  // The number of packed voxels
  cube[Det].NumOfVoxels=charged;
 
  fprintf(stderr,"Cube %i:\n",Det);
  fprintf(stderr,"Residual charge: %e  Minimal charge: %e\n" ,cube[Det].sum,min_Q);  
  fprintf(stderr,"Total voxels: %li  Charged voxels: %li\n" ,(long)cube[Det].MX * cube[Det].MY * cube[Det].MZ, cube[Det].NumOfVoxels);

  //  Before we do anything with the cubes, we must correct for the small 
  //    residual charge present.  This charge arises because the cube
  //    files only store five significant figures.
  //  Our original way to do this was with a constant correction.  This is
  //    not ideal because all of the very small charges end up taking on 
  //    roughly the value of the correction factor.  Thus, the relative 
  //    sizes of the charges get modified.  The newer method uses a linear
  //    correction factor such that each charge is modified only in the 
  //    last digit and below.  This preserves the relative sizes of 
  //    each of the charges.
  //  Below we'll preserve each of these as well as allowing no charge
  //    correction at all.

  
  // ************   Linear correction ****************
  correct = cube[Det].sum/cube[Det].NumOfVoxels;
  i = 0;
  prevsum = 0;
  cutfactor = 1e-5;
  maxit=100;
  alpha = 200.;
  cut = cutfactor*min_Q;
  diffsum = fabs(cube[Det].sum) - fabs(prevsum);
  prevdiffsum = fabs(diffsum) + 1;
  
  ix=0;
  float extract[8] __attribute__ ((aligned (32)));

  while((fabs(diffsum)>cut) && (ix < maxit))
    {
      min_Q=10.f;max_Q=0.f;
      if(fabs(diffsum)>fabs(prevdiffsum))
	alpha*=0.1;
      
      for (i=0;i<v_count;i++) 
	{
	  _mm256_store_ps(extract,cube[Det].rQ[i]);
	  for(j=0; j<8; j++) 
	    {
	      abs_Q=fabs(extract[j]);
	      if((abs_Q < min_Q) && (extract[j]!=.0)) min_Q = abs_Q;
	      if(abs_Q > max_Q) max_Q=abs_Q;
	    }
	  extract[0] = extract[1] = extract[2] = extract[3] = extract[4] = extract[5] = extract[6] = extract[7] = 0.0f;
	}
      
      cut=min_Q*cutfactor;
      cube[Det].sum=0.0;
      ix++;
      
      for (i=0;i<v_count;i++) {
	  _mm256_store_ps(extract,cube[Det].rQ[i]);
	  for(j=0; j<8; j++) {
	  	extract[j] -=alpha*correct*fabs(extract[j])/max_Q;
	  	cube[Det].sum+= extract[j];
	  }
	  cube[Det].rQ[i] = _mm256_set_ps(extract[7],extract[6],extract[5],extract[4],extract[3],extract[2],extract[1],extract[0]);
      }
      correct = cube[Det].sum/cube[Det].NumOfVoxels;
      prevdiffsum = diffsum;
      diffsum = fabs(cube[Det].sum) - fabs(prevsum);
      prevsum = cube[Det].sum;
    }
 
  fprintf(stderr,"Corrected charge: %e in %li iterrations\n",cube[Det].sum,ix);
  fflush(stderr);
  float x_extract[8] __attribute__ ((aligned (32)));
  float y_extract[8] __attribute__ ((aligned (32)));
  float z_extract[8] __attribute__ ((aligned (32)));
  // Find the center of 'charge' 
  vec_scale(cube[Det].center,0.0,cube[Det].center); 
  for (i=0;i<v_count;i++) {
      _mm256_store_ps(extract,cube[Det].rQ[i]);
      _mm256_store_ps(x_extract,cube[Det].rX[i]);
      _mm256_store_ps(y_extract,cube[Det].rY[i]);
      _mm256_store_ps(z_extract,cube[Det].rZ[i]);
      for(j=0; j<8; j++) {
      	cube[Det].abssum += fabs(extract[j]);
      	cube[Det].center[0] += fabs(extract[j])*x_extract[j];
      	cube[Det].center[1] += fabs(extract[j])*y_extract[j];
      	cube[Det].center[2] += fabs(extract[j])*z_extract[j];
      }
    }
  vec_scale(cube[Det].center,1/cube[Det].abssum,cube[Det].center);
  fprintf(stderr,"Center of charge is at:  [%f %f %f] Ang\n",cube[Det].center[0],cube[Det].center[1],cube[Det].center[2]); 
  fprintf(stderr,"Center of charge is at:  [%f %f %f] Bohr\n",cube[Det].center[0]/Bohr2Ang,cube[Det].center[1]/Bohr2Ang,cube[Det].center[2]/Bohr2Ang); 
  
  fclose(GaussianCube);
}



double CalcCoulombCoupling(int C1, int C2, int cache_size)
{
  // Returns Coulomb interaction between 2 TDC
  // TDC are  stored in global structures cube[0] and cube[1]
  long i, j, h, m, c2count;
  double R, VC,VCexp,VC_OVL,VC_OVLexp, kappa,VDD, VDDexp, exp_correct0, exp_correct1;
  int id, nthreads, cache_blocks;
  double dR[3], dRn[3], Dipole0[3], Dipole1[3], mu0, mu1, dmu0[3], dmu1[3];  
 __m256 r_vec, result, vcovlps, vcps;
 __m256 R_cutoff_check, R_CUTOFF;
 __m256 tmpQ[8], tmpX[8], tmpY[8], tmpZ[8], diff[8];

 float tmp_add[4][8] __attribute__ ((aligned (32)));
 float x_extract[8]  __attribute__ ((aligned (32)));

 Rcutoff =  0.25*sqrt(pow(fabs(cube[C1].Vol),2.0/3.0))+pow(fabs(cube[C2].Vol),2.0/3.0); 
 R_CUTOFF = _mm256_rcp_ps(_mm256_set1_ps(Rcutoff));
 VC = VC_OVL = 0;
 
 cache_blocks=cube[C2].v_count/cache_size;

 if(cache_blocks==0)
   cache_blocks++;
 c2count=cube[C2].v_count/(cache_blocks);

 fprintf(stderr,"Cube2: N vectors = %i\nCPU cache size = %i KB\nN blocks =  %i\n",cube[C2].v_count, cache_size,cache_blocks);
 
 // Take care of the last pack of cube 2, which may be incomplete
 // If any of X coordinates in this pack are zeros 
 // we set them to 1.0f to avoid division by zero.

 _mm256_store_ps(x_extract,cube[C2].rX[cube[C2].v_count-1]);
 for(i=cube[C2].last_v_element_count; i<8; i++) 
   x_extract[i] = 1.0f;
 cube[C2].rX[cube[C2].v_count-1] = _mm256_set_ps(x_extract[7],x_extract[6],x_extract[5],x_extract[4],x_extract[3],x_extract[2],x_extract[1],x_extract[0]);

 
#pragma omp parallel private(tmpQ,tmpX,tmpY,tmpZ, id,i,j,h,m,diff,r_vec,vcps,vcovlps,tmp_add,result,R_cutoff_check) reduction(+:VC,VC_OVL)
 {
   id = omp_get_thread_num();
   if(id==0)
     {
       nthreads = omp_get_num_threads();
       fprintf(stderr,"Running on %i processors\n",nthreads);fflush(stderr);
     }
   
     for(h=0;h<cache_blocks;h++) 
     {
      fprintf(stderr,"Block %li Thread: %i\r",h,id);fflush(stderr);
        for(i=id; i<cube[C1].v_count;i+=nthreads)
        {
             tmpQ[0] = _mm256_broadcast_ss(&cube[C1].rQ[i][0]);
             tmpQ[1] = _mm256_broadcast_ss(&cube[C1].rQ[i][1]);
             tmpQ[2] = _mm256_broadcast_ss(&cube[C1].rQ[i][2]);
             tmpQ[3] = _mm256_broadcast_ss(&cube[C1].rQ[i][3]);
             tmpQ[4] = _mm256_broadcast_ss(&cube[C1].rQ[i][4]);
             tmpQ[5] = _mm256_broadcast_ss(&cube[C1].rQ[i][5]);
             tmpQ[6] = _mm256_broadcast_ss(&cube[C1].rQ[i][6]);
             tmpQ[7] = _mm256_broadcast_ss(&cube[C1].rQ[i][7]);
             
	     tmpX[0] = _mm256_broadcast_ss(&cube[C1].rX[i][0]);
             tmpX[1] = _mm256_broadcast_ss(&cube[C1].rX[i][1]);
             tmpX[2] = _mm256_broadcast_ss(&cube[C1].rX[i][2]);
             tmpX[3] = _mm256_broadcast_ss(&cube[C1].rX[i][3]);
             tmpX[4] = _mm256_broadcast_ss(&cube[C1].rX[i][4]);
             tmpX[5] = _mm256_broadcast_ss(&cube[C1].rX[i][5]);
             tmpX[6] = _mm256_broadcast_ss(&cube[C1].rX[i][6]);
             tmpX[7] = _mm256_broadcast_ss(&cube[C1].rX[i][7]);
             
	     tmpY[0] = _mm256_broadcast_ss(&cube[C1].rY[i][0]);
             tmpY[1] = _mm256_broadcast_ss(&cube[C1].rY[i][1]);
             tmpY[2] = _mm256_broadcast_ss(&cube[C1].rY[i][2]);
             tmpY[3] = _mm256_broadcast_ss(&cube[C1].rY[i][3]);
             tmpY[4] = _mm256_broadcast_ss(&cube[C1].rY[i][4]);
             tmpY[5] = _mm256_broadcast_ss(&cube[C1].rY[i][5]);
             tmpY[6] = _mm256_broadcast_ss(&cube[C1].rY[i][6]);
             tmpY[7] = _mm256_broadcast_ss(&cube[C1].rY[i][7]);

             tmpZ[0] = _mm256_broadcast_ss(&cube[C1].rZ[i][0]);
             tmpZ[1] = _mm256_broadcast_ss(&cube[C1].rZ[i][1]);
             tmpZ[2] = _mm256_broadcast_ss(&cube[C1].rZ[i][2]);
             tmpZ[3] = _mm256_broadcast_ss(&cube[C1].rZ[i][3]);
             tmpZ[4] = _mm256_broadcast_ss(&cube[C1].rZ[i][4]);
             tmpZ[5] = _mm256_broadcast_ss(&cube[C1].rZ[i][5]);
             tmpZ[6] = _mm256_broadcast_ss(&cube[C1].rZ[i][6]);
             tmpZ[7] = _mm256_broadcast_ss(&cube[C1].rZ[i][7]);

           for(m=0; m<8; m++)
           {
              vcps = vcovlps = _mm256_setzero_ps();
              for(j=h*c2count; j<(h+1)*c2count; j++)
              {
                 diff[0] = _mm256_sub_ps(tmpX[m],cube[C2].rX[j]);
                 diff[1] = _mm256_sub_ps(tmpY[m],cube[C2].rY[j]);
                 diff[2] = _mm256_sub_ps(tmpZ[m],cube[C2].rZ[j]);
  
                 diff[0] = _mm256_mul_ps(diff[0],diff[0]);
                 diff[1] = _mm256_mul_ps(diff[1],diff[1]);
                 diff[2] = _mm256_mul_ps(diff[2],diff[2]);
 
                 r_vec = _mm256_add_ps(diff[0],diff[1]);
                 r_vec = _mm256_add_ps(r_vec,diff[2]);
                 r_vec = _mm256_rsqrt_ps(r_vec);
   
                 result = _mm256_mul_ps(tmpQ[m],cube[C2].rQ[j]);
                 result = _mm256_mul_ps(result,r_vec);
                 vcps = _mm256_add_ps(vcps,result);       // Accumulate coupling between all voxels
                 R_cutoff_check = _mm256_cmp_ps(r_vec,R_CUTOFF,18);
                 result = _mm256_and_ps(R_cutoff_check,result);
                 vcovlps = _mm256_add_ps(vcovlps,result); // Accumulate coupling excluding overlapped
              }
              _mm256_store_ps(tmp_add[1],vcps);
              _mm256_store_ps(tmp_add[0],vcovlps);
              VC += tmp_add[0][0] + tmp_add[0][1] + tmp_add[0][2] + tmp_add[0][3] + tmp_add[0][4] + tmp_add[0][5] + tmp_add[0][6] + tmp_add[0][7];
              VC_OVL += tmp_add[1][0] + tmp_add[1][1] + tmp_add[1][2] + tmp_add[1][3] + tmp_add[1][4] + tmp_add[1][5] + tmp_add[1][6] + tmp_add[1][7];
           }
	}      
     }	
     fprintf(stderr,"Processor %i done, VC = %15e\n",id,VC);
 } // End of parallel section
#pragma omp barrier
 

 Line(60);
  // Convert to eV
  VC=VC*1e10*HtoeV*echarge*echarge/(4*M_PI*ep0*JtoHartree);
  //Volume correction factor
  // VC=VC*cube[C1].Vol*cube[C2].Vol;

  VC_OVL=VC_OVL*1e10*HtoeV*echarge*echarge/(4*M_PI*ep0*JtoHartree);
  //Volume correction factor
  // VC_OVL=VC_OVL*cube[C1].Vol*cube[C2].Vol;
 
  // Dipole coupling 
  vec_diff(cube[1].center,cube[0].center,dR);
  R=sqrt(vec_dot(dR,dR));
  vec_scale(dR,1/R,dRn);
 
  CalcDipole(0, Dipole0);
  CalcDipole(1, Dipole1);

  mu0 = sqrt(vec_dot(Dipole0,Dipole0));
  mu1 = sqrt(vec_dot(Dipole1,Dipole1));

  vec_scale(Dipole0, 1/mu0, dmu0);
  vec_scale(Dipole1, 1/mu1, dmu1);

  kappa = vec_dot(dmu0,dmu1) - 3.0*vec_dot(dmu0,dRn)*vec_dot(dmu1,dRn);
  VDD=1.e10*mu0*mu1*echarge*echarge*kappa*HtoeV/(4.*JtoHartree*M_PI*ep0*R*R*R);
  fprintf(stderr,"Distance = %f\n",R);
  fprintf(stderr,"Orientation factor = %f\n",kappa);
  Line(60);
  fprintf(stderr,"Full Coulomb coupling w/o overlapped voxels: %e eV\n", VC);
  fprintf(stderr,"                                             %e cm-1\n\n", VC*eVtocm);
  fprintf(stderr,"Full Coulomb coupling w overlapped voxels:   %e eV\n", VC_OVL);
  fprintf(stderr,"                                             %e cm-1\n\n", VC_OVL*eVtocm );
  fprintf(stderr,"Dipole coupling:                             %e eV\n",VDD);
  fprintf(stderr,"                                             %e cm-1\n", VDD*eVtocm);
  Line(60);
  fprintf(stderr,"After scaling dipole moments to the experimental values:\n");

  exp_correct0=cube[0].Dipole/(mu0*eAngtoDebye);
  exp_correct1=cube[1].Dipole/(mu1*eAngtoDebye);

  VDDexp=VDD*exp_correct0* exp_correct1;
  VCexp=VC*exp_correct0* exp_correct1;
  VC_OVLexp=VC_OVL*exp_correct0* exp_correct1;

  fprintf(stderr,"Full Coulomb coupling w/o overlapped voxels: %e eV\n", VCexp);
  fprintf(stderr,"                                             %e cm-1\n\n", VCexp*eVtocm);
  fprintf(stderr,"Full Coulomb coupling w overlapped voxels:   %e eV\n", VC_OVLexp);
  fprintf(stderr,"                                             %e cm-1\n\n", VC_OVLexp*eVtocm);
  fprintf(stderr,"Dipole coupling:                             %e eV\n",VDDexp);
  fprintf(stderr,"                                             %e cm-1\n", VDDexp*eVtocm);
  Line(60);

  return VC;
}




/* DESCRIPTION: A FUNCTION TO CALCULATE TRANSITION  DIPOLE MOMENT AND RETURN SCALING FACTOR */
/* INPUT: index of the cube and experimental value of the dipole moment                     */
/* Function returns scaling factor:  (Experimental Dipole)/(Theoretical dipole)             */
double CalcDipole(int Det, double *DipoleXYZ)
{
  long  i,j;
  double Dipole;
  
  // Zero vector  
  vec_scale(DipoleXYZ,0.0,DipoleXYZ);
  float extract[8] __attribute__ ((aligned (32)));
  float x_extract[8] __attribute__ ((aligned (32)));
  float y_extract[8] __attribute__ ((aligned (32)));
  float z_extract[8] __attribute__ ((aligned (32)));

  //Calculate and accumulate cartesian components of the dipole moment
  for(i=0;i<cube[Det].v_count;i++) {
      _mm256_store_ps(extract,cube[Det].rQ[i]);
      _mm256_store_ps(x_extract,cube[Det].rX[i]);
      _mm256_store_ps(y_extract,cube[Det].rY[i]);
      _mm256_store_ps(z_extract,cube[Det].rZ[i]);
      for(j=0; j<8; j++) {
      	DipoleXYZ[0] -= x_extract[j]*extract[j];
      	DipoleXYZ[1] -= y_extract[j]*extract[j];
      	DipoleXYZ[2] -= z_extract[j]*extract[j];
      }
  }
  // Volume correction factor
  // ORCA cubes are already corrected
  //   vec_scale(DipoleXYZ,cube[Det].Vol,DipoleXYZ);

  // Find the value of the total dipole moment
  Dipole=sqrt(vec_dot(DipoleXYZ,DipoleXYZ));
  //Print the dipole moment info
  fprintf(stderr," Cube %i - Dipole Moment = %g e*Ang\n"
	 "[%g %g %g]\n",Det, Dipole, DipoleXYZ[0], DipoleXYZ[1], DipoleXYZ[2]);
  
  // Return scaling factor
  return (Dipole);
}


inline static void Line(int n)
{
  int i;
  for(i=0;i<n;i++)
    fprintf(stderr, "-");
  fprintf(stderr, "\n");
}

inline static double vec_dot(double *a, double *b)
{
  return(a[0]*b[0]+a[1]*b[1]+a[2]*b[2]);
}


inline static void vec_scale(double *a, double c, double *ac)
{
int i;
  for(i=0;i<3;i++)
    ac[i]=c*a[i];
}

inline static void vec_diff(double *a, double *b, double *diff)
{
  int i;
  for(i=0;i<3;i++)
    diff[i]=a[i]-b[i];
}
