/*
 ============================================================================
 Name        : sphere_tracing.cu
 Description : sphere tracing
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <cstdlib>

#include "algebra.cuh"
#include "shape.cuh"

using namespace std;

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

__device__ vec itertrace(
		int nshapes,shape * worldShapes,material * worldMaterials,
		const ray & r,curandState * state)
{
	info I;
	double iter = intersect(nshapes,worldShapes,r,I)*1.0/(MAX_ITER);
	if(I.idx < 0) return ZERO;

	return iter*worldMaterials[I.idx].alb;
}

__device__ vec pathtrace(
		int nshapes,shape * worldShapes,material * worldMaterials,
		const ray & r,curandState * state)
{
	ray curr_r = r;

	vec acum_w = ONES;
	info I;
	for(int d = 0;d<MAX_DEPTH;d++)
	{
		I.s = MAX_TIME;
		I.idx = -1;

		double iter = intersect(nshapes,worldShapes,curr_r,I)*1.0/(MAX_ITER);
		if(I.idx < 0)
			return ZERO;
		material & m = worldMaterials[I.idx];
		if(m.type == 0)
			return acum_w % m.alb;

		samp newd = get_samp(m,curr_r,I,state);
		/*
		double p = max(newd.w);
		if(curand_uniform_double(state) >= p)
		{
			return ZERO;
			//return acum_s+(1e-1*newd.w%acum_w)/(1.0-p);
		}
		newd.w /= p;
		*/
		acum_w %= newd.w;

		curr_r.o = I.x+I.eps*I.n;
		curr_r.d = newd.d;
	}


	info J;
	vec rand_p = 2.0*worldShapes[nshapes-1].n%vec(curand_uniform_double(state)-0.5,
						  curand_uniform_double(state)-0.5,
						  curand_uniform_double(state)-0.5)+worldShapes[nshapes-1].c;
	vec rand_d = norm(rand_p - curr_r.o);

	double iter = intersect(nshapes,worldShapes,ray(curr_r.o,rand_d),J)*1.0/(MAX_ITER);

	if(J.idx < 0)
		return ZERO;
	material & m = worldMaterials[J.idx];
	if(m.type != 0) return ZERO;

	return /*(/len2(curr_r.o-J.x))*/max(-curr_r.d*rand_d,0.0)*(acum_w%m.alb);

	//;
	/*
	info I;
	double iter = intersect(nshapes,worldShapes,r,I)*1.0/(MAX_ITER);
	//double t = (I.s-MIN_TIME)/(50.0-MIN_TIME);

	if(I.idx < 0) return ZERO;

	//return iter*worldMaterials[I.idx].alb;//(I.n+ONES);
	samp newd = get_samp(worldMaterials[I.idx],r,I,state);
	if(max(newd.e)>0.0) return newd.e;

	if(depth <= MAX_ROULETTE)
	{
		double p = max(newd.w);
		if(curand_uniform(state) < p){
			newd.w /= p;
		}
		else return ZERO;
	}

	if(depth == 0) return newd.e;

	int B = (depth == 1 && worldMaterials[I.idx].type==1)? 2 : 1;
	for(int b=0;b<B;b++)
	{
		newd = get_samp(worldMaterials[I.idx],r,I,state);
		acum += (newd.w/(1.0*B))%pathtrace(nshapes,worldShapes,worldMaterials,ray(I.x+I.eps*I.n,newd.d),state,depth-1);
	}
	*/
}

__global__ void init_seeds_kernel(curandState *state, int seed, int nrows,int ncols)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int jdx = blockIdx.y*blockDim.y+threadIdx.y;
	if(idx < nrows && jdx < ncols)
	{
		curand_init(seed, ncols*idx+jdx, 0, &state[ncols*idx+jdx]);
	}
}

__global__ void bounce_kernel(
		double eye_x,double eye_y,double eye_z,
		double view_x,double view_y,double view_z,
		double f,
		const int nshapes,shape * worldShapes, material * worldMaterials,
		int nrows,int ncols,
		curandState * state,
		vec * output,
		int samps,
		int subss)
{

	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int jdx = blockIdx.y*blockDim.y+threadIdx.y;

	extern __shared__ int s[];
	shape    * shworldShapes = (shape*)s;
	material * shworldMaterials = (material *)&shworldShapes[nshapes];
	if(threadIdx.x < nshapes)
	{
		shworldShapes[threadIdx.x] = worldShapes[threadIdx.x];
		shworldMaterials[threadIdx.x] = worldMaterials[threadIdx.x];
	}
	__syncthreads();
	if(idx < nrows && jdx < ncols)
	{
		//int idx = kdx/ncols;
		//int jdx = kdx%ncols;
		vec eye(eye_x,eye_y,eye_z);
		vec view(view_x,view_y,view_z);
		double inv_tsamps = 1.0/(subss*subss*samps);
		vec res = output[ncols*idx+jdx];
		for(int adx=0;adx<subss;adx++)
		for(int bdx=0;bdx<subss;bdx++)
		//for(int s=0;s<samps;s++)
		{
			ray aux_r = generate_ray(eye,view,f,
					                 nrows,ncols,2,
					                 idx,jdx,adx,bdx,
					                 &state[ncols*idx+jdx]);
			//res += inv_tsamps*vec(curand_uniform_double(&state[kdx]),curand_uniform_double(&state[kdx]),curand_uniform_double(&state[kdx]));
			res += inv_tsamps*clamp(pathtrace(nshapes,shworldShapes,shworldMaterials,aux_r,&state[ncols*idx+jdx]),ZERO,ONES);
			//res += inv_tsamps*clamp(itertrace(nshapes,worldShapes,worldMaterials,aux_r,&state[ncols*idx+jdx]),ZERO,ONES);
		}
		output[ncols*idx+jdx] = res;
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
	nshapes = 12;
	hostShapes = new shape [nshapes];
	hostMaterials = new material [nshapes];

	//suelo
	hostShapes[0].type = 0;
	hostShapes[0].c = ZERO;
	hostShapes[0].n = YAXIS;
	hostMaterials[0].type = 1;
	hostMaterials[0].alb  = vec(0.99,0.99,0.99);
	hostMaterials[1].kd = 0.01;
	hostMaterials[1].ks = 0.99;
	hostMaterials[1].ex = 5.0;

	//techo
	hostShapes[1].type = 0;
	hostShapes[1].c = 20.0*YAXIS;
	hostShapes[1].n = -YAXIS;
	hostMaterials[1].type = 1;
	hostMaterials[1].alb  = vec(0.99,0.99,0.99);
	hostMaterials[1].kd = 0.9;
	hostMaterials[1].ks = 0.1;
	hostMaterials[1].ex = 1.0;
	//z_wall+
	hostShapes[2].type = 0;
	hostShapes[2].c = 20.0*ZAXIS;
	hostShapes[2].n = -ZAXIS;
	hostMaterials[2].type = 1;
	hostMaterials[2].alb  = vec(0.99,0.99,0.99);
	hostMaterials[2].kd = 0.9;
	hostMaterials[2].ks = 0.1;
	hostMaterials[2].ex = 1.0;
	//z_wall-
	hostShapes[3].type = 0;
	hostShapes[3].c = -20.0*ZAXIS;
	hostShapes[3].n = ZAXIS;
	hostMaterials[3].type = 1;
	hostMaterials[3].alb  = vec(0.99,0.99,0.99);
	hostMaterials[3].kd = 0.9;
	hostMaterials[3].ks = 0.1;
	hostMaterials[3].ex = 1.0;
	//x_wall+
	hostShapes[4].type = 0;
	hostShapes[4].c = 20.0*XAXIS;
	hostShapes[4].n = -XAXIS;
	hostMaterials[4].type = 1;
	hostMaterials[4].alb  = vec(237.0/255.0, 120.0/255.0, 18.0/255.0);
	hostMaterials[4].kd = 0.9;
	hostMaterials[4].ks = 0.1;
	hostMaterials[4].ex = 1.0;
	//x_wall-
	hostShapes[5].type = 0;
	hostShapes[5].c = -20.0*XAXIS;
	hostShapes[5].n = XAXIS;
	hostMaterials[5].type = 1;
	hostMaterials[5].alb  = vec(18.0/255.0, 120.0/255.0, 237/255.0);
	hostMaterials[5].kd = 0.9;
	hostMaterials[5].ks = 0.1;
	hostMaterials[5].ex = 1.0;
	//spec torus
	hostShapes[6].type = 3;
	hostShapes[6].c = vec(-10.0,5.0,0.0);
	hostShapes[6].a = 1.0;
	hostShapes[6].n = vec(4.0,1.0,0.0);
	hostMaterials[6].type = 2;
	hostMaterials[6].alb  = vec(255/255.0, 221/255.0, 50/255.0);

	//diffuse sphere
	hostShapes[7].type = 1;
	hostShapes[7].c = vec( 8.0,2.0,-6.0);
	hostShapes[7].a = 2.0;
	//hostShapes[7].n = vec(4.0,2.0,0.0);
	hostMaterials[7].type = 1;
	hostMaterials[7].alb  = vec(18.0/255.0, 237/255.0, 120.0/255.0);
	hostMaterials[7].kd = 0.5;
	hostMaterials[7].ks = 0.5;
	hostMaterials[7].ex = 50.0;

	//diffuse cylinder
	hostShapes[8].type = 4;
	hostShapes[8].c = vec( 8.0,0.0,10.0);
	hostShapes[8].a = 2.0;
	hostMaterials[8].type = 1;
	hostMaterials[8].alb  = vec(142.0/255.0, 121.0/255.0, 79.0/255.0);
	hostMaterials[8].kd = 0.8;
	hostMaterials[8].ks = 0.2;
	hostMaterials[8].ex = 2.0;

	//diffuse cylinder
	hostShapes[9].type = 4;
	hostShapes[9].c = vec(-8.0,0.0,10.0);
	hostShapes[9].a = 2.0;
	hostMaterials[9].type = 1;
	hostMaterials[9].alb  = vec(142.0/255.0, 121.0/255.0, 79.0/255.0);
	hostMaterials[9].kd = 0.8;
	hostMaterials[9].ks = 0.2;
	hostMaterials[9].ex = 2.0;

	//diffuse cylinder
	hostShapes[10].type = 4;
	hostShapes[10].c = vec(0.0,0.0,10.0);
	hostShapes[10].a = 2.0;
	hostMaterials[10].type = 1;
	hostMaterials[10].alb  = vec(142.0/255.0, 121.0/255.0, 79.0/255.0);
	hostMaterials[10].kd = 0.8;
	hostMaterials[10].ks = 0.2;
	hostMaterials[10].ex = 2.0;

	//luz  box
	hostShapes[11].type = 2;
	hostShapes[11].c = vec(0.0,20.0,0.0);
	hostShapes[11].a = 1.0;
	hostShapes[11].n = vec(10.0,0.5,2.0);
	hostMaterials[11].type = 0;
	hostMaterials[11].alb  = vec(1.0,1.0,1.0);

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
	int samps = 8;
	int subss = 4;

	curandState *devStates;
	vec * devOutput;

	cudaMalloc((void ** )&devStates,(nrows*ncols*sizeof(curandState)));
	cudaMalloc((void ** )&devOutput,(nrows*ncols*sizeof(vec)));

	dim3 thread_size(16, 16);
	dim3 block_size(nrows/thread_size.x,ncols/thread_size.y);
	init_seeds_kernel<<<block_size,thread_size>>>(devStates,rand(),nrows,ncols);
	cudaDeviceSynchronize();
	for(int s=0;s<samps;s++)
	{
		cout << s << endl;
		bounce_kernel<<<block_size,thread_size,(nshapes)*(sizeof(shape)+sizeof(material))>>>(
			  eye.x,eye.y,eye.z,
			  view.x,view.y,view.z,
			  45.0,
			  nshapes,devShapes,devMaterials,
			  nrows,ncols,
			  devStates,
			  devOutput,
			  samps,
			  subss);
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

