/*
 ============================================================================
 Name        : sphere_tracing.cu
 Author      : gionuno
 Version     :
 Copyright   : 
 Description : sphere tracing
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <cstdlib>

#include "algebra.cuh"
#include "shape.cuh"

using namespace std;

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__device__ ray generate_ray(const vec & eye, const vec & view,double f,
							int nrows,int ncols,int subs,
							int idx,int jdx,int adx,int bdx,
							curandState * state)
{
	double fov = 2.*tan(M_PI*f/180.0);
	double asp = nrows>ncols? ncols*1.0/nrows : nrows*1.0/ncols;

	vec n = norm(view-eye);
	vec u = norm(YAXIS^n);
	vec v = norm(u^n);

	u = fov*u;
	v = fov*v;

	double r_x = 2.0*curand_uniform_double(state);
	double r_y = 2.0*curand_uniform_double(state);

	double d_x = r_x<1.0?sqrt(r_x)-1.0:1.0-sqrt(2.0-r_x);
	double d_y = r_y<1.0?sqrt(r_y)-1.0:1.0-sqrt(2.0-r_y);

	return ray(eye,
		       norm(n+asp*((idx+(adx+0.5+d_y)*1./subs)*1./nrows-0.5)*v+((jdx+(bdx+0.5+d_x)*1./subs)*1./ncols-0.5)*u)
		    );
}

__device__ vec pathtrace(
		int nshapes,shape * worldShapes,material * worldMaterials,
		const ray & r,curandState * state,int depth)
{
	info I;
	double t = 1.0-(intersect(nshapes,worldShapes,r,I)-MIN_TIME)/(MAX_TIME-MIN_TIME);

	if(I.idx < 0) return ZERO;

	//return 0.5*t*(I.n+ONES);
	samp newd = get_samp(worldMaterials[I.idx],r,I,state);
	if(2*depth <= MAX_DEPTH)
	{
		double p = max(max(newd.e),max(newd.w));
		if(curand_uniform_double(state) < p){
			newd.e /= p;
			newd.w /= p;
		}
		else return newd.e;
	}

	if(depth == 0) return newd.e;

	//return ZERO;
	return newd.e+newd.w%pathtrace(nshapes,worldShapes,worldMaterials,ray(I.x+I.eps*I.n,newd.d),state,depth-1);
	//*/
}

__global__ void init_seeds_kernel(curandState *state, int seed, int nrows,int ncols)
{
	int kdx = blockIdx.x*blockDim.x+threadIdx.x;
	//int jdx = blockIdx.y*blockDim.y+threadIdx.y;
	if(kdx < nrows*ncols)// && jdx < ncols)
	{
		curand_init(seed, kdx, 0, &state[kdx]);
	}
}

__global__ void bounce_kernel(
		double eye_x,double eye_y,double eye_z,
		double view_x,double view_y,double view_z,
		double f,
		int nshapes,shape * worldShapes, material * worldMaterials,
		int nrows,int ncols,
		curandState * state,
		vec * output,
		int samps,
		int subss,
		int depth)
{
	int kdx = blockIdx.x*blockDim.x+threadIdx.x;
	if(kdx < nrows*ncols)
	{
		int idx = kdx/ncols;
		int jdx = kdx%ncols;
		vec eye(eye_x,eye_y,eye_z);
		vec view(view_x,view_y,view_z);
		double inv_tsamps = 1.0/(subss*subss*samps);
		vec res = output[kdx];
		for(int adx=0;adx<subss;adx++)
		for(int bdx=0;bdx<subss;bdx++)
		//for(int s=0;s<samps;s++)
		{
			ray aux_r = generate_ray(eye,view,f,
					                 nrows,ncols,2,
					                 idx,jdx,adx,bdx,
					                 &state[kdx]);
			//res += inv_tsamps*vec(curand_uniform_double(&state[kdx]),curand_uniform_double(&state[kdx]),curand_uniform_double(&state[kdx]));
			res += inv_tsamps*clamp(pathtrace(nshapes,worldShapes,worldMaterials,aux_r,&state[kdx],depth),ZERO,ONES);
			//0.5*(aux_r.d+ONES);
		}
		output[kdx] = res;
	}
}

//scene devWorld;

int nshapes;
shape * hostShapes;
shape * devShapes;

material * hostMaterials;
material * devMaterials;

void init_world()
{
	nshapes = 8;
	hostShapes = new shape [nshapes];
	hostMaterials = new material [nshapes];

	//suelo
	hostShapes[0].type = 0;
	hostShapes[0].c = ZERO;
	hostShapes[0].n = YAXIS;
	hostMaterials[0].type = 1;
	hostMaterials[0].alb  = vec(0.95,0.95,0.95);
	//techo
	hostShapes[1].type = 0;
	hostShapes[1].c = 20.0*YAXIS;
	hostShapes[1].n = -YAXIS;
	hostMaterials[1].type = 1;
	hostMaterials[1].alb  = vec(0.95,0.95,0.95);

	//z_wall+
	hostShapes[2].type = 0;
	hostShapes[2].c = 20.0*ZAXIS;
	hostShapes[2].n = -ZAXIS;
	hostMaterials[2].type = 1;
	hostMaterials[2].alb  = vec(0.95,0.95,0.95);
	//z_wall-
	hostShapes[3].type = 0;
	hostShapes[3].c = -20.0*ZAXIS;
	hostShapes[3].n = ZAXIS;
	hostMaterials[3].type = 1;
	hostMaterials[3].alb  = vec(0.95,0.95,0.95);

	//x_wall+
	hostShapes[4].type = 0;
	hostShapes[4].c = 20.0*XAXIS;
	hostShapes[4].n = -XAXIS;
	hostMaterials[4].type = 1;
	hostMaterials[4].alb  = vec(237.0/255.0, 120.0/255.0, 18.0/255.0);

	//x_wall-
	hostShapes[5].type = 0;
	hostShapes[5].c = -20.0*XAXIS;
	hostShapes[5].n = XAXIS;
	hostMaterials[5].type = 1;
	hostMaterials[5].alb  = vec(18.0/255.0, 144.0/255.0, 224/255.0);

	//spec sphere
	hostShapes[6].type = 3;
	hostShapes[6].c = vec(10.0,5.0,0.0);
	hostShapes[6].a = 1.0;
	hostShapes[6].n = vec(4.0,1.0,0.0);
	hostMaterials[6].type = 2;
	hostMaterials[6].alb  = vec(0.95,0.95,0.95);

	//luz  sphere
	hostShapes[7].type = 2;
	hostShapes[7].c = vec(0.0,20.0,0.0);
	hostShapes[7].a = 1.0;
	hostShapes[7].n = vec(5.0,2.0,5.0);
	hostMaterials[7].type = 0;
	hostMaterials[7].alb  = vec(10.0,10.0,10.0);

	cudaMalloc((void **)&devShapes,sizeof(shape)*nshapes);
	cudaMalloc((void **)&devMaterials,sizeof(material)*nshapes);

	cudaMemcpy(devShapes,hostShapes,sizeof(shape)*nshapes, cudaMemcpyHostToDevice);
	cudaMemcpy(devMaterials,hostMaterials,sizeof(material)*nshapes, cudaMemcpyHostToDevice);

}

void free_world()
{
	delete [] hostShapes;
	cudaFree(devShapes);

	delete [] hostMaterials;
	cudaFree(devMaterials);
}

int main(void)
{
	srand(time(NULL));

	vec eye(0.0,5.0,-19.0);
	vec view(0.0,5.0,20.0);
	cout << "init_world()...";
	init_world();
	cout << "done" << endl;

	int nrows = 512;
	int ncols = 512;
	int samps = 16;
	int subss = 2;
	int depth = MAX_DEPTH;

	curandState *devStates;
	vec * devOutput;

	cudaMalloc((void ** )&devStates,(nrows*ncols*sizeof(curandState)));
	cudaMalloc((void ** )&devOutput,(nrows*ncols*sizeof(vec)));

	//dim3 thread_size(32, 32);
	//dim3 block_size(nrows*ncols/thread_size.x,ncols/thread_size.y);
	init_seeds_kernel<<<nrows*ncols/512,512>>>(devStates,rand(),nrows,ncols);
	cudaDeviceSynchronize();
	for(int s=0;s<samps;s++)
	{
		cout << s << endl;
		bounce_kernel<<<nrows*ncols/512,512>>>(
			  eye.x,eye.y,eye.z,
			  view.x,view.y,view.z,
			  45.0,
			  nshapes,devShapes,devMaterials,
			  nrows,ncols,
			  devStates,
			  devOutput,
			  samps,
			  subss,
			  depth);
		cudaDeviceSynchronize();
	}
	vec * hostOutput = new vec [nrows*ncols];
	cudaMemcpy(hostOutput,devOutput,sizeof(vec)*nrows*ncols,cudaMemcpyDeviceToHost);

	FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
	fprintf(f, "P3\n%d %d\n%d\n", ncols, nrows, 255);
	for (int i=0; i<nrows*ncols; i++)
	    fprintf(f,"%d %d %d ", (int)(255.0*clamp(hostOutput[i].x,0.0,1.0)+0.5),
	    			           (int)(255.0*clamp(hostOutput[i].y,0.0,1.0)+0.5),
	    			           (int)(255.0*clamp(hostOutput[i].z,0.0,1.0)+0.5));

	cout << "free_world()...";
	free_world();
	cout << "done" << endl;

	cudaFree(devStates);
	cudaFree(devOutput);
	delete [] hostOutput;

	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

